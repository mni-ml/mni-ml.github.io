import {
  Tensor, native,
  Module, Parameter,
  Linear, Embedding,
  softmax, gelu, layerNorm,
} from '@mni-ml/framework';
import type { BPETokenizer } from './bpe';
import type { CheckpointData, ModelConfig } from './model';
import type {
  BenchmarkProgress, RunTrace, StepTrace, ReplayToken,
} from './kvReplay';

export interface MlaModelConfig extends ModelConfig {
  dKv: number;
}

export interface MlaCheckpointData extends Omit<CheckpointData, 'config'> {
  config: MlaModelConfig;
}

export interface MlaRunOptions {
  prompt: string;
  maxNewTokens: number;
  temperature?: number;
  onProgress?: (progress: BenchmarkProgress) => void;
  onStep?: (run: RunTrace, step: StepTrace) => void;
  shouldStop?: () => boolean;
}

class LayerNorm extends Module {
  gamma!: Parameter<Tensor>;
  beta!: Parameter<Tensor>;

  constructor(dim: number) {
    super();
    this.gamma = new Parameter(
      Tensor.fromFloat32(new Float32Array(dim).fill(1.0), [dim]),
    );
    this.beta = new Parameter(
      Tensor.fromFloat32(new Float32Array(dim).fill(0.0), [dim]),
    );
  }

  forward(x: Tensor): Tensor {
    return layerNorm(x, this.gamma.value, this.beta.value, 1e-5);
  }
}

function ensurePromptTokens(tokenizer: BPETokenizer, prompt: string): number[] {
  const tokens = tokenizer.encode(prompt);
  if (tokens.length > 0) return tokens;
  if (tokenizer.eotToken >= 0) return [tokenizer.eotToken];
  throw new Error('prompt must encode to at least one token');
}

function yieldToBrowser(): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, 0));
}

function startNoGrad(): void {
  if (typeof native.noGradStart === 'function') native.noGradStart();
}

function endNoGrad(): void {
  if (typeof native.noGradEnd === 'function') native.noGradEnd();
}

function sampleNextToken(logits: Tensor, vocabSize: number, temperature: number): number {
  const data = logits.toFloat32();
  const offset = data.length - vocabSize;
  if (temperature <= 0) {
    let best = -Infinity;
    let bestIndex = 0;
    for (let i = 0; i < vocabSize; i++) {
      const value = data[offset + i];
      if (value > best) {
        best = value;
        bestIndex = i;
      }
    }
    return bestIndex;
  }

  const scaled = new Float32Array(vocabSize);
  let maxLogit = -Infinity;
  for (let i = 0; i < vocabSize; i++) {
    scaled[i] = data[offset + i] / temperature;
    if (scaled[i] > maxLogit) maxLogit = scaled[i];
  }

  let sum = 0;
  const probs = new Float32Array(vocabSize);
  for (let i = 0; i < vocabSize; i++) {
    probs[i] = Math.exp(scaled[i] - maxLogit);
    sum += probs[i];
  }
  let threshold = Math.random();
  let cumulative = 0;
  for (let i = 0; i < vocabSize; i++) {
    cumulative += probs[i] / sum;
    if (threshold < cumulative) return i;
  }
  return vocabSize - 1;
}

function buildReplayTokens(ids: number[], promptTokens: number, tokenizer: BPETokenizer): ReplayToken[] {
  return ids.map((id, index) => ({
    id,
    text: tokenizer.decodeToken(id),
    isPrompt: index < promptTokens,
  }));
}

function createToken(id: number, prompt: boolean, tokenizer: BPETokenizer): ReplayToken {
  return { id, text: tokenizer.decodeToken(id), isPrompt: prompt };
}

class MlaLatentCache {
  readonly dKv: number;
  readonly maxSeqLen: number;
  private len = 0;
  private readonly latents: Float32Array;

  constructor(dKv: number, maxSeqLen: number) {
    this.dKv = dKv;
    this.maxSeqLen = maxSeqLen;
    this.latents = new Float32Array(maxSeqLen * dKv);
  }

  length(): number {
    return this.len;
  }

  free(): void {
    this.len = 0;
  }

  append(latent: Float32Array): void {
    if (this.len >= this.maxSeqLen) {
      throw new Error(`mla latent cache is full (${this.len}/${this.maxSeqLen})`);
    }
    if (latent.length !== this.dKv) {
      throw new Error(`expected latent length ${this.dKv}, got ${latent.length}`);
    }
    this.latents.set(latent, this.len * this.dKv);
    this.len += 1;
  }

  asTensor(): Tensor {
    const view = this.latents.slice(0, this.len * this.dKv);
    return Tensor.fromFloat32(view, [1, this.len, this.dKv]);
  }
}

class MLACausalSelfAttention extends Module {
  nHead: number;
  nEmbd: number;
  headDim: number;
  dKv: number;
  scale: number;
  queryProj!: Linear;
  kvDownProj!: Linear;
  kvLatentNorm!: LayerNorm;
  keyUpProj!: Linear;
  valueUpProj!: Linear;
  outProj!: Linear;

  // Absorbed weights, populated by absorbWeights() after parameters are loaded.
  // The MLA inference trick: precompute Wq @ Wk^T and Wv @ Wo per head so that
  // attention runs entirely in the dKv-dim latent space and we never have to
  // re-up-project the cached latents to full K/V at every decode step.
  private wQkPerHead: Tensor[] = [];
  private bQkPerHead: Tensor[] = [];
  private wVoPerHead: Tensor[] = [];
  private bVoPerHead: Tensor[] = [];
  private absorbed = false;

  constructor(nEmbd: number, nHead: number, dKv: number) {
    super();
    if (nEmbd % nHead !== 0) {
      throw new Error(`nEmbd (${nEmbd}) must be divisible by nHead (${nHead})`);
    }
    this.nEmbd = nEmbd;
    this.nHead = nHead;
    this.headDim = nEmbd / nHead;
    this.dKv = dKv;
    this.scale = 1.0 / Math.sqrt(this.headDim);
    this.queryProj = new Linear(nEmbd, nEmbd);
    this.kvDownProj = new Linear(nEmbd, dKv);
    this.kvLatentNorm = new LayerNorm(dKv);
    this.keyUpProj = new Linear(dKv, nEmbd);
    this.valueUpProj = new Linear(dKv, nEmbd);
    this.outProj = new Linear(nEmbd, nEmbd);
  }

  absorbWeights(): void {
    const { nHead, nEmbd, headDim, dKv } = this;

    const Wq = this.queryProj.weight.value.toFloat32();   // [nEmbd, nEmbd]
    const bQ = this.queryProj.bias.value.toFloat32();     // [nEmbd]
    const Wk = this.keyUpProj.weight.value.toFloat32();   // [dKv, nEmbd]
    const Wv = this.valueUpProj.weight.value.toFloat32(); // [dKv, nEmbd]
    const bV = this.valueUpProj.bias.value.toFloat32();   // [nEmbd]
    const Wo = this.outProj.weight.value.toFloat32();     // [nEmbd, nEmbd]

    this.wQkPerHead = [];
    this.bQkPerHead = [];
    this.wVoPerHead = [];
    this.bVoPerHead = [];

    for (let h = 0; h < nHead; h++) {
      const headOffset = h * headDim;

      // wQk[h] = Wq[:, headSlice] @ Wk[:, headSlice]^T   shape [nEmbd, dKv]
      const wQkH = new Float32Array(nEmbd * dKv);
      for (let i = 0; i < nEmbd; i++) {
        for (let k = 0; k < dKv; k++) {
          let s = 0;
          for (let hd = 0; hd < headDim; hd++) {
            s += Wq[i * nEmbd + headOffset + hd] * Wk[k * nEmbd + headOffset + hd];
          }
          wQkH[i * dKv + k] = s;
        }
      }

      // bQk[h] = bQ[headSlice] @ Wk[:, headSlice]^T   shape [dKv]
      // (bK contributes only a constant per row, which softmax cancels)
      const bQkH = new Float32Array(dKv);
      for (let k = 0; k < dKv; k++) {
        let s = 0;
        for (let hd = 0; hd < headDim; hd++) {
          s += bQ[headOffset + hd] * Wk[k * nEmbd + headOffset + hd];
        }
        bQkH[k] = s;
      }

      // wVo[h] = Wv[:, headSlice] @ Wo[headSlice, :]   shape [dKv, nEmbd]
      const wVoH = new Float32Array(dKv * nEmbd);
      for (let r = 0; r < dKv; r++) {
        for (let j = 0; j < nEmbd; j++) {
          let s = 0;
          for (let hd = 0; hd < headDim; hd++) {
            s += Wv[r * nEmbd + headOffset + hd] * Wo[(headOffset + hd) * nEmbd + j];
          }
          wVoH[r * nEmbd + j] = s;
        }
      }

      // bVo[h] = bV[headSlice] @ Wo[headSlice, :]   shape [nEmbd]
      const bVoH = new Float32Array(nEmbd);
      for (let j = 0; j < nEmbd; j++) {
        let s = 0;
        for (let hd = 0; hd < headDim; hd++) {
          s += bV[headOffset + hd] * Wo[(headOffset + hd) * nEmbd + j];
        }
        bVoH[j] = s;
      }

      this.wQkPerHead.push(Tensor.fromFloat32(wQkH, [nEmbd, dKv]));
      this.bQkPerHead.push(Tensor.fromFloat32(bQkH, [1, dKv]));
      this.wVoPerHead.push(Tensor.fromFloat32(wVoH, [dKv, nEmbd]));
      this.bVoPerHead.push(Tensor.fromFloat32(bVoH, [1, nEmbd]));
    }

    this.absorbed = true;
  }

  forwardStep(x: Tensor, cache: MlaLatentCache): Tensor {
    if (!this.absorbed) {
      throw new Error('MLACausalSelfAttention.forwardStep called before absorbWeights()');
    }
    const [, seqLen] = x.shape;
    if (seqLen !== 1) {
      throw new Error(`mla attention expects seq_len=1, got ${seqLen}`);
    }

    const { nHead, nEmbd, dKv } = this;

    const cKvNew = this.kvLatentNorm.forward(this.kvDownProj.forward(x));
    cache.append(cKvNew.toFloat32());

    const len = cache.length();
    const cKv2D = cache.asTensor().view(len, dKv);     // [L, dKv]
    const cKvT = cKv2D.permute(1, 0).contiguous();      // [dKv, L]

    const xFlat = x.view(1, nEmbd);                     // [1, nEmbd]

    let outSum: Tensor | null = null;
    for (let h = 0; h < nHead; h++) {
      const qInter = xFlat
        .matmul(this.wQkPerHead[h])
        .add(this.bQkPerHead[h]);                       // [1, dKv]
      const scores = qInter.matmul(cKvT).mul(this.scale); // [1, L]
      const weights = softmax(scores, -1);                // [1, L]
      const tmp = weights.matmul(cKv2D);                  // [1, dKv]
      const outH = tmp
        .matmul(this.wVoPerHead[h])
        .add(this.bVoPerHead[h]);                       // [1, nEmbd]
      outSum = outSum === null ? outH : outSum.add(outH);
    }

    const final = outSum!.add(this.outProj.bias.value); // [1, nEmbd]
    return final.view(1, 1, nEmbd);
  }
}

class FeedForward extends Module {
  fc1!: Linear;
  fc2!: Linear;

  constructor(nEmbd: number) {
    super();
    this.fc1 = new Linear(nEmbd, 4 * nEmbd);
    this.fc2 = new Linear(4 * nEmbd, nEmbd);
  }

  forward(x: Tensor): Tensor {
    return this.fc2.forward(gelu(this.fc1.forward(x)));
  }
}

class TransformerBlock extends Module {
  ln1!: LayerNorm;
  attn!: MLACausalSelfAttention;
  ln2!: LayerNorm;
  ffn!: FeedForward;

  constructor(nEmbd: number, nHead: number, dKv: number) {
    super();
    this.ln1 = new LayerNorm(nEmbd);
    this.attn = new MLACausalSelfAttention(nEmbd, nHead, dKv);
    this.ln2 = new LayerNorm(nEmbd);
    this.ffn = new FeedForward(nEmbd);
  }

  forwardStep(x: Tensor, cache: MlaLatentCache): Tensor {
    x = x.add(this.attn.forwardStep(this.ln1.forward(x), cache));
    x = x.add(this.ffn.forward(this.ln2.forward(x)));
    return x;
  }
}

class MlaReplayMiniGPT extends Module {
  config: MlaModelConfig;
  vocabSize: number;
  tokenEmb!: Embedding;
  posEmb!: Embedding;
  lnFinal!: LayerNorm;
  headBias!: Parameter<Tensor>;
  [key: string]: any;

  constructor(vocabSize: number, config: MlaModelConfig) {
    super();
    this.config = config;
    this.vocabSize = vocabSize;
    this.tokenEmb = new Embedding(vocabSize, config.nEmbd);
    this.posEmb = new Embedding(config.blockSize, config.nEmbd);
    for (let i = 0; i < config.nLayer; i++) {
      this[`block${i}`] = new TransformerBlock(config.nEmbd, config.nHead, config.dKv);
    }
    this.lnFinal = new LayerNorm(config.nEmbd);
    this.headBias = new Parameter(Tensor.zeros([vocabSize]));
  }

  createCaches(): MlaLatentCache[] {
    return Array.from(
      { length: this.config.nLayer },
      () => new MlaLatentCache(this.config.dKv, this.config.blockSize),
    );
  }

  absorbWeights(): void {
    for (let i = 0; i < this.config.nLayer; i++) {
      const block = (this as any)[`block${i}`] as TransformerBlock;
      block.attn.absorbWeights();
    }
  }

  forwardToken(tokenId: number, position: number, caches: MlaLatentCache[]): Tensor {
    let x = this.tokenEmb.forward([[tokenId]]);
    x = x.add(this.posEmb.forward([[position]]));
    for (let i = 0; i < this.config.nLayer; i++) {
      x = (this as any)[`block${i}`].forwardStep(x, caches[i]);
    }
    x = this.lnFinal.forward(x);
    const weightT = this.tokenEmb.weight.value.permute(1, 0);
    return x.matmul(weightT).add(this.headBias.value);
  }
}

export function loadMlaModel(checkpoint: MlaCheckpointData, tokenizer: BPETokenizer): MlaReplayMiniGPT {
  if (checkpoint.config.dKv == null) {
    throw new Error('mla checkpoint is missing config.dKv');
  }
  const vocabSize = checkpoint.parameters['tokenEmb.weight']?.shape[0]
    ?? tokenizer.vocabSize;
  const model = new MlaReplayMiniGPT(vocabSize, checkpoint.config);
  for (const [name, param] of model.namedParameters()) {
    const saved = checkpoint.parameters[name];
    if (!saved) continue;
    (param as Parameter<Tensor>).update(
      Tensor.fromFloat32(new Float32Array(saved.data), saved.shape),
    );
  }
  model.eval();
  model.absorbWeights();
  return model;
}

function mlaCacheBytes(config: MlaModelConfig, cacheLen: number) {
  const bytesPerToken = config.nLayer * config.dKv * 4;
  return {
    usedBytes: cacheLen * bytesPerToken,
    capacityBytes: config.blockSize * bytesPerToken,
  };
}

function makeStepTrace(
  tokenizer: BPETokenizer,
  context: number[],
  promptTokens: number,
  config: MlaModelConfig,
  fields: Omit<StepTrace, 'tokens' | 'cacheBytesUsed' | 'cacheBytesCapacity' | 'cacheMode' | 'mode'>,
): StepTrace {
  const bytes = mlaCacheBytes(config, fields.cacheLen);
  return {
    mode: 'mla',
    cacheMode: 'mla',
    tokens: buildReplayTokens(context, promptTokens, tokenizer),
    cacheBytesUsed: bytes.usedBytes,
    cacheBytesCapacity: bytes.capacityBytes,
    ...fields,
  };
}

function buildRunTrace(
  tokenizer: BPETokenizer,
  config: MlaModelConfig,
  prompt: string,
  promptTokens: number,
  maxNewTokens: number,
  temperature: number,
  context: number[],
  prefillMs: number,
  decodeMs: number,
  steps: StepTrace[],
): RunTrace {
  const generatedTokens = Math.max(0, context.length - promptTokens);
  const totalMs = prefillMs + decodeMs;
  const lastStep = steps[steps.length - 1];
  const cacheLen = lastStep?.cacheLen ?? 0;
  const bytes = mlaCacheBytes(config, cacheLen);

  return {
    mode: 'mla',
    cacheMode: 'mla',
    prompt,
    promptTokens,
    generatedTokens,
    temperature,
    blockSize: config.blockSize,
    nLayer: config.nLayer,
    nHead: config.nHead,
    replayCapacity: promptTokens + maxNewTokens,
    totalMs,
    prefillMs,
    decodeMs,
    decodeMsPerToken: generatedTokens > 0 ? decodeMs / generatedTokens : 0,
    tokensPerSec: totalMs > 0 ? (generatedTokens * 1000) / totalMs : 0,
    cacheLen,
    cacheBytesUsed: bytes.usedBytes,
    cacheBytesCapacity: bytes.capacityBytes,
    text: tokenizer.decode(context),
    steps: steps.slice(),
  };
}

function triangularWork(size: number): number {
  return (size * (size + 1)) / 2;
}

export async function runMlaTrace(
  model: MlaReplayMiniGPT,
  tokenizer: BPETokenizer,
  options: MlaRunOptions,
): Promise<RunTrace> {
  const prompt = options.prompt;
  const maxNewTokens = options.maxNewTokens;
  const temperature = options.temperature ?? 0;
  const onProgress = options.onProgress;
  const onStep = options.onStep;
  const shouldStop = options.shouldStop;

  const promptIds = ensurePromptTokens(tokenizer, prompt);
  if (promptIds.length + maxNewTokens > model.config.blockSize) {
    throw new Error(
      `prompt_tokens + max_new_tokens must be <= blockSize (${promptIds.length + maxNewTokens} > ${model.config.blockSize})`,
    );
  }

  const caches = model.createCaches();
  const context = [...promptIds];
  const steps: StepTrace[] = [];
  let prefillMs = 0;
  let decodeMs = 0;

  model.eval();
  startNoGrad();
  try {
    onProgress?.({ label: 'Running MLA prefill...', current: 0, total: maxNewTokens + 1 });
    const prefillStart = performance.now();
    let logits: Tensor | null = null;
    for (let index = 0; index < promptIds.length; index++) {
      logits = model.forwardToken(promptIds[index], index, caches);
    }
    prefillMs = performance.now() - prefillStart;

    const prefillCacheLen = caches[0]?.length() ?? 0;
    steps.push(makeStepTrace(tokenizer, context, promptIds.length, model.config, {
      phase: 'prefill',
      stepIndex: 0,
      seqLen: context.length,
      focusIndex: context.length - 1,
      outputToken: null,
      cacheLen: prefillCacheLen,
      reusedPositions: 0,
      recomputedPositions: context.length,
      workUnits: triangularWork(context.length),
      stepMs: prefillMs,
      note: `prefill filled ${prefillCacheLen} latent positions across all layers before decode begins`,
    }));
    onStep?.(
      buildRunTrace(
        tokenizer, model.config, prompt, promptIds.length, maxNewTokens,
        temperature, context, prefillMs, decodeMs, steps,
      ),
      steps[steps.length - 1],
    );
    await yieldToBrowser();
    if (shouldStop?.()) {
      return buildRunTrace(
        tokenizer, model.config, prompt, promptIds.length, maxNewTokens,
        temperature, context, prefillMs, decodeMs, steps,
      );
    }

    if (!logits) {
      throw new Error('mla prefill did not produce logits');
    }

    for (let step = 0; step < maxNewTokens; step++) {
      const nextToken = sampleNextToken(logits, model.vocabSize, temperature);
      context.push(nextToken);
      const decodeStart = performance.now();
      logits = model.forwardToken(nextToken, promptIds.length + step, caches);
      const stepMs = performance.now() - decodeStart;
      decodeMs += stepMs;
      const cacheLen = caches[0]?.length() ?? 0;
      steps.push(makeStepTrace(tokenizer, context, promptIds.length, model.config, {
        phase: 'decode',
        stepIndex: step + 1,
        seqLen: context.length,
        focusIndex: context.length - 1,
        outputToken: createToken(nextToken, false, tokenizer),
        cacheLen,
        reusedPositions: Math.max(0, cacheLen - 1),
        recomputedPositions: 1,
        workUnits: cacheLen,
        stepMs,
        note: `this step reused ${Math.max(0, cacheLen - 1)} past latents (d_c=${model.config.dKv})`,
      }));
      onStep?.(
        buildRunTrace(
          tokenizer, model.config, prompt, promptIds.length, maxNewTokens,
          temperature, context, prefillMs, decodeMs, steps,
        ),
        steps[steps.length - 1],
      );
      onProgress?.({
        label: `Running MLA... ${step + 1}/${maxNewTokens}`,
        current: step + 1,
        total: maxNewTokens,
      });
      await yieldToBrowser();
      if (shouldStop?.()) break;
    }
  } finally {
    endNoGrad();
    for (const cache of caches) cache.free();
  }

  return buildRunTrace(
    tokenizer, model.config, prompt, promptIds.length, maxNewTokens,
    temperature, context, prefillMs, decodeMs, steps,
  );
}
