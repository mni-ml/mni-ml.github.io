import {
  Tensor, native,
  Module, Parameter,
  Linear, Embedding,
  softmax, gelu, layerNorm, flashAttention,
} from '@mni-ml/framework';
import type { BPETokenizer } from './bpe';
import type { CheckpointData, ModelConfig } from './model';

export type BenchmarkMode = 'baseline' | 'kv-fp32' | 'kv-int8';
export type CacheMode = 'none' | 'fp32' | 'int8';

export interface ReplayToken {
  id: number;
  text: string;
  isPrompt: boolean;
}

export interface StepTrace {
  mode: BenchmarkMode;
  cacheMode: CacheMode;
  phase: 'prefill' | 'decode';
  stepIndex: number;
  seqLen: number;
  focusIndex: number;
  tokens: ReplayToken[];
  outputToken: ReplayToken | null;
  cacheLen: number;
  cacheBytesUsed: number;
  cacheBytesCapacity: number;
  reusedPositions: number;
  recomputedPositions: number;
  workUnits: number;
  stepMs: number;
  note: string;
}

export interface RunTrace {
  mode: BenchmarkMode;
  cacheMode: CacheMode;
  prompt: string;
  promptTokens: number;
  generatedTokens: number;
  temperature: number;
  blockSize: number;
  nLayer: number;
  nHead: number;
  replayCapacity: number;
  totalMs: number;
  prefillMs: number;
  decodeMs: number;
  decodeMsPerToken: number;
  tokensPerSec: number;
  cacheLen: number;
  cacheBytesUsed: number;
  cacheBytesCapacity: number;
  text: string;
  steps: StepTrace[];
}

export interface BenchmarkSuite {
  prompt: string;
  temperature: number;
  maxNewTokens: number;
  runs: Record<BenchmarkMode, RunTrace>;
  outputsMatch: boolean;
}

export interface BenchmarkProgress {
  label: string;
  current: number;
  total: number;
}

export interface BenchmarkOptions {
  prompt: string;
  maxNewTokens: number;
  temperature?: number;
  onProgress?: (progress: BenchmarkProgress) => void;
}

export interface BenchmarkModeOptions extends BenchmarkOptions {
  mode: BenchmarkMode;
}

const MODE_ORDER: BenchmarkMode[] = ['baseline', 'kv-fp32', 'kv-int8'];
const EPS_SCALE = 1e-8;

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

const causalMaskCache = new Map<number, Tensor>();

function getCausalMask(size: number): Tensor {
  const cached = causalMaskCache.get(size);
  if (cached) return cached;
  const storage = new Float32Array(size * size);
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      storage[i * size + j] = j <= i ? 0.0 : -1e9;
    }
  }
  const mask = Tensor.fromFloat32(storage, [1, 1, size, size]);
  causalMaskCache.set(size, mask);
  return mask;
}

function reshapeForAttention(x: Tensor, batch: number, seqLen: number, nHead: number, headDim: number): Tensor {
  return x.view(batch, seqLen, nHead, headDim).permute(0, 2, 1, 3).contiguous();
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

function triangularWork(size: number): number {
  return (size * (size + 1)) / 2;
}

function cacheBytesForMode(config: ModelConfig, cacheLen: number, cacheMode: CacheMode) {
  if (cacheMode === 'none') {
    return { usedBytes: 0, capacityBytes: 0 };
  }
  const headDim = config.nEmbd / config.nHead;
  const rowsPerToken = config.nLayer * config.nHead;
  const valueBytesPerRow = cacheMode === 'int8' ? headDim : headDim * 4;
  const scaleBytesPerRow = cacheMode === 'int8' ? 4 : 0;
  const bytesPerToken = 2 * rowsPerToken * (valueBytesPerRow + scaleBytesPerRow);
  return {
    usedBytes: cacheLen * bytesPerToken,
    capacityBytes: config.blockSize * bytesPerToken,
  };
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

function expectStepTensor(name: string, tensor: Tensor, numHeads: number, headDim: number): Float32Array {
  const shape = tensor.shape;
  if (
    shape.length !== 4
    || shape[0] !== 1
    || shape[1] !== numHeads
    || shape[2] !== 1
    || shape[3] !== headDim
  ) {
    throw new Error(
      `${name} expected [1, ${numHeads}, 1, ${headDim}], got [${shape.join(', ')}]`,
    );
  }
  return tensor.toFloat32();
}

function quantizeRowI8(source: Float32Array, sourceOffset: number, target: Int8Array, targetOffset: number, length: number): number {
  let maxAbs = 0;
  for (let i = 0; i < length; i++) {
    maxAbs = Math.max(maxAbs, Math.abs(source[sourceOffset + i]));
  }
  const scale = maxAbs > 0 ? maxAbs / 127 : EPS_SCALE;
  for (let i = 0; i < length; i++) {
    const quantized = Math.round(source[sourceOffset + i] / scale);
    target[targetOffset + i] = Math.max(-127, Math.min(127, quantized));
  }
  return scale;
}

class BrowserKvCache {
  readonly numHeads: number;
  readonly headDim: number;
  readonly maxSeqLen: number;
  readonly quantized: boolean;
  private len = 0;
  private readonly keysFp32: Float32Array | null;
  private readonly valuesFp32: Float32Array | null;
  private readonly keysI8: Int8Array | null;
  private readonly valuesI8: Int8Array | null;
  private readonly keyScales: Float32Array | null;
  private readonly valueScales: Float32Array | null;

  constructor(numHeads: number, headDim: number, maxSeqLen: number, quantized: boolean) {
    this.numHeads = numHeads;
    this.headDim = headDim;
    this.maxSeqLen = maxSeqLen;
    this.quantized = quantized;
    const valueCapacity = numHeads * maxSeqLen * headDim;
    const scaleCapacity = numHeads * maxSeqLen;
    this.keysFp32 = quantized ? null : new Float32Array(valueCapacity);
    this.valuesFp32 = quantized ? null : new Float32Array(valueCapacity);
    this.keysI8 = quantized ? new Int8Array(valueCapacity) : null;
    this.valuesI8 = quantized ? new Int8Array(valueCapacity) : null;
    this.keyScales = quantized ? new Float32Array(scaleCapacity) : null;
    this.valueScales = quantized ? new Float32Array(scaleCapacity) : null;
  }

  length(): number {
    return this.len;
  }

  free(): void {
    this.len = 0;
  }

  decodeStep(q: Tensor, k: Tensor, v: Tensor, scale: number): Tensor {
    const qData = expectStepTensor('q', q, this.numHeads, this.headDim);
    const kData = expectStepTensor('k', k, this.numHeads, this.headDim);
    const vData = expectStepTensor('v', v, this.numHeads, this.headDim);
    this.append(kData, vData);
    const currentLen = this.len;
    const out = new Float32Array(this.numHeads * this.headDim);

    for (let head = 0; head < this.numHeads; head++) {
      const qOffset = head * this.headDim;
      const scores = new Float32Array(currentLen);
      let maxScore = -Infinity;

      for (let pos = 0; pos < currentLen; pos++) {
        let dot = 0;
        const base = ((head * this.maxSeqLen) + pos) * this.headDim;
        if (this.quantized) {
          const keyScale = this.keyScales![head * this.maxSeqLen + pos];
          for (let d = 0; d < this.headDim; d++) {
            dot += qData[qOffset + d] * this.keysI8![base + d] * keyScale;
          }
        } else {
          for (let d = 0; d < this.headDim; d++) {
            dot += qData[qOffset + d] * this.keysFp32![base + d];
          }
        }
        scores[pos] = dot * scale;
        if (scores[pos] > maxScore) maxScore = scores[pos];
      }

      let sum = 0;
      for (let pos = 0; pos < currentLen; pos++) {
        const weight = Math.exp(scores[pos] - maxScore);
        scores[pos] = weight;
        sum += weight;
      }

      const invSum = sum > 0 ? 1 / sum : 0;
      for (let pos = 0; pos < currentLen; pos++) {
        const weight = scores[pos] * invSum;
        const base = ((head * this.maxSeqLen) + pos) * this.headDim;
        if (this.quantized) {
          const valueScale = this.valueScales![head * this.maxSeqLen + pos];
          for (let d = 0; d < this.headDim; d++) {
            out[qOffset + d] += weight * this.valuesI8![base + d] * valueScale;
          }
        } else {
          for (let d = 0; d < this.headDim; d++) {
            out[qOffset + d] += weight * this.valuesFp32![base + d];
          }
        }
      }
    }

    return Tensor.fromFloat32(out, [1, this.numHeads, 1, this.headDim]);
  }

  private append(kData: Float32Array, vData: Float32Array): void {
    if (this.len >= this.maxSeqLen) {
      throw new Error(`kv cache is full (${this.len}/${this.maxSeqLen})`);
    }
    const position = this.len;
    for (let head = 0; head < this.numHeads; head++) {
      const sourceOffset = head * this.headDim;
      const targetOffset = ((head * this.maxSeqLen) + position) * this.headDim;
      if (this.quantized) {
        this.keyScales![head * this.maxSeqLen + position] = quantizeRowI8(
          kData,
          sourceOffset,
          this.keysI8!,
          targetOffset,
          this.headDim,
        );
        this.valueScales![head * this.maxSeqLen + position] = quantizeRowI8(
          vData,
          sourceOffset,
          this.valuesI8!,
          targetOffset,
          this.headDim,
        );
      } else {
        this.keysFp32!.set(kData.subarray(sourceOffset, sourceOffset + this.headDim), targetOffset);
        this.valuesFp32!.set(vData.subarray(sourceOffset, sourceOffset + this.headDim), targetOffset);
      }
    }
    this.len += 1;
  }
}

class CausalSelfAttention extends Module {
  nHead: number;
  headDim: number;
  scale: number;
  queryProj!: Linear;
  keyProj!: Linear;
  valueProj!: Linear;
  outProj!: Linear;

  constructor(nEmbd: number, nHead: number) {
    super();
    this.nHead = nHead;
    this.headDim = nEmbd / nHead;
    this.scale = 1 / Math.sqrt(this.headDim);
    this.queryProj = new Linear(nEmbd, nEmbd);
    this.keyProj = new Linear(nEmbd, nEmbd);
    this.valueProj = new Linear(nEmbd, nEmbd);
    this.outProj = new Linear(nEmbd, nEmbd);
  }

  forward(x: Tensor): Tensor {
    const [batch, seqLen, embd] = x.shape;
    let q = this.queryProj.forward(x);
    let k = this.keyProj.forward(x);
    let v = this.valueProj.forward(x);
    q = q.view(batch, seqLen, this.nHead, this.headDim).permute(0, 2, 1, 3).contiguous().view(batch * this.nHead, seqLen, this.headDim);
    k = k.view(batch, seqLen, this.nHead, this.headDim).permute(0, 2, 1, 3).contiguous().view(batch * this.nHead, seqLen, this.headDim);
    v = v.view(batch, seqLen, this.nHead, this.headDim).permute(0, 2, 1, 3).contiguous().view(batch * this.nHead, seqLen, this.headDim);

    let out: Tensor;
    if (typeof native.flashAttention === 'function') {
      out = flashAttention(q, k, v, this.scale, true);
    } else {
      let scores = q.matmul(k.permute(0, 2, 1)).mul(this.scale);
      const mask = getCausalMask(seqLen).view(seqLen, seqLen);
      scores = scores.add(mask);
      const probs = softmax(scores, -1);
      out = probs.matmul(v);
    }

    out = out.view(batch, this.nHead, seqLen, this.headDim).permute(0, 2, 1, 3).contiguous().view(batch, seqLen, embd);
    return this.outProj.forward(out);
  }

  forwardStep(x: Tensor, cache: BrowserKvCache): Tensor {
    const [batch, seqLen, embd] = x.shape;
    if (seqLen !== 1) {
      throw new Error(`kv attention expects seq_len=1, got ${seqLen}`);
    }
    const q = reshapeForAttention(this.queryProj.forward(x), batch, seqLen, this.nHead, this.headDim);
    const k = reshapeForAttention(this.keyProj.forward(x), batch, seqLen, this.nHead, this.headDim);
    const v = reshapeForAttention(this.valueProj.forward(x), batch, seqLen, this.nHead, this.headDim);
    const out = cache.decodeStep(q, k, v, this.scale)
      .permute(0, 2, 1, 3)
      .contiguous()
      .view(batch, seqLen, embd);
    return this.outProj.forward(out);
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
  attn!: CausalSelfAttention;
  ln2!: LayerNorm;
  ffn!: FeedForward;

  constructor(nEmbd: number, nHead: number) {
    super();
    this.ln1 = new LayerNorm(nEmbd);
    this.attn = new CausalSelfAttention(nEmbd, nHead);
    this.ln2 = new LayerNorm(nEmbd);
    this.ffn = new FeedForward(nEmbd);
  }

  forward(x: Tensor): Tensor {
    x = x.add(this.attn.forward(this.ln1.forward(x)));
    x = x.add(this.ffn.forward(this.ln2.forward(x)));
    return x;
  }

  forwardStep(x: Tensor, cache: BrowserKvCache): Tensor {
    x = x.add(this.attn.forwardStep(this.ln1.forward(x), cache));
    x = x.add(this.ffn.forward(this.ln2.forward(x)));
    return x;
  }
}

class ReplayMiniGPT extends Module {
  config: ModelConfig;
  vocabSize: number;
  tokenEmb!: Embedding;
  posEmb!: Embedding;
  lnFinal!: LayerNorm;
  headBias!: Parameter<Tensor>;
  [key: string]: any;

  constructor(vocabSize: number, config: ModelConfig) {
    super();
    this.config = config;
    this.vocabSize = vocabSize;
    this.tokenEmb = new Embedding(vocabSize, config.nEmbd);
    this.posEmb = new Embedding(config.blockSize, config.nEmbd);
    for (let i = 0; i < config.nLayer; i++) {
      this[`block${i}`] = new TransformerBlock(config.nEmbd, config.nHead);
    }
    this.lnFinal = new LayerNorm(config.nEmbd);
    this.headBias = new Parameter(Tensor.zeros([vocabSize]));
  }

  forward(indices: number[][]): Tensor {
    const batch = indices.length;
    const seqLen = indices[0].length;
    let x = this.tokenEmb.forward(indices);
    const posIndices: number[][] = [];
    for (let b = 0; b < batch; b++) {
      posIndices.push(Array.from({ length: seqLen }, (_, index) => index));
    }
    x = x.add(this.posEmb.forward(posIndices));
    for (let i = 0; i < this.config.nLayer; i++) {
      x = this[`block${i}`].forward(x);
    }
    x = this.lnFinal.forward(x);
    const weightT = this.tokenEmb.weight.value.permute(1, 0);
    return x.matmul(weightT).add(this.headBias.value);
  }

  createCaches(cacheMode: Exclude<CacheMode, 'none'>): BrowserKvCache[] {
    if (cacheMode !== 'fp32' && cacheMode !== 'int8') {
      throw new Error(`unsupported cache mode: ${cacheMode}`);
    }
    const headDim = this.config.nEmbd / this.config.nHead;
    const quantized = cacheMode === 'int8';
    return Array.from(
      { length: this.config.nLayer },
      () => new BrowserKvCache(this.config.nHead, headDim, this.config.blockSize, quantized),
    );
  }

  forwardToken(tokenId: number, position: number, caches: BrowserKvCache[]): Tensor {
    let x = this.tokenEmb.forward([[tokenId]]);
    x = x.add(this.posEmb.forward([[position]]));
    for (let i = 0; i < this.config.nLayer; i++) {
      x = this[`block${i}`].forwardStep(x, caches[i]);
    }
    x = this.lnFinal.forward(x);
    const weightT = this.tokenEmb.weight.value.permute(1, 0);
    return x.matmul(weightT).add(this.headBias.value);
  }
}

export function loadReplayModel(checkpoint: CheckpointData, tokenizer: BPETokenizer): ReplayMiniGPT {
  const vocabSize = checkpoint.parameters['tokenEmb.weight']?.shape[0]
    ?? tokenizer.vocabSize;
  const model = new ReplayMiniGPT(vocabSize, checkpoint.config);
  for (const [name, param] of model.namedParameters()) {
    const saved = checkpoint.parameters[name];
    if (!saved) continue;
    param.update(Tensor.fromFloat32(new Float32Array(saved.data), saved.shape));
  }
  model.eval();
  return model;
}

function makeStepTrace(
  tokenizer: BPETokenizer,
  context: number[],
  promptTokens: number,
  config: ModelConfig,
  mode: BenchmarkMode,
  cacheMode: CacheMode,
  fields: Omit<StepTrace, 'tokens' | 'cacheBytesUsed' | 'cacheBytesCapacity' | 'cacheMode' | 'mode'>,
): StepTrace {
  const bytes = cacheBytesForMode(config, fields.cacheLen, cacheMode);
  return {
    mode,
    cacheMode,
    tokens: buildReplayTokens(context, promptTokens, tokenizer),
    cacheBytesUsed: bytes.usedBytes,
    cacheBytesCapacity: bytes.capacityBytes,
    ...fields,
  };
}

function makeNote(mode: BenchmarkMode, phase: 'prefill' | 'decode', seqLen: number, reusedPositions: number, cacheLen: number, cacheMode: CacheMode): string {
  if (phase === 'prefill') {
    if (mode === 'baseline') {
      return `prefill computed the ${seqLen}-token prompt once with a full causal pass`;
    }
    return `prefill filled ${cacheLen} cached positions across all layers before decode begins`;
  }
  if (mode === 'baseline') {
    return `this step recomputed the whole ${seqLen}-token prefix`;
  }
  const dtype = cacheMode === 'int8' ? 'int8' : 'fp32';
  return `this step reused ${reusedPositions} past positions from a ${dtype} kv cache`;
}

type RunSeed = {
  mode: BenchmarkMode;
  cacheMode: CacheMode;
  prefillMs: number;
  decodeMs: number;
};

async function runBaselineTrace(
  model: ReplayMiniGPT,
  tokenizer: BPETokenizer,
  prompt: string,
  maxNewTokens: number,
  temperature: number,
  onProgress?: (progress: BenchmarkProgress) => void,
): Promise<RunTrace> {
  const promptIds = ensurePromptTokens(tokenizer, prompt);
  if (promptIds.length + maxNewTokens > model.config.blockSize) {
    throw new Error(
      `prompt_tokens + max_new_tokens must be <= blockSize (${promptIds.length + maxNewTokens} > ${model.config.blockSize})`,
    );
  }

  const context = [...promptIds];
  const steps: StepTrace[] = [];
  const runSeed: RunSeed = {
    mode: 'baseline',
    cacheMode: 'none',
    prefillMs: 0,
    decodeMs: 0,
  };

  model.eval();
  startNoGrad();
  try {
    onProgress?.({ label: 'Running baseline prefill...', current: 0, total: maxNewTokens + 1 });
    const prefillStart = performance.now();
    let logits = model.forward([context]);
    runSeed.prefillMs = performance.now() - prefillStart;
    steps.push(makeStepTrace(tokenizer, context, promptIds.length, model.config, runSeed.mode, runSeed.cacheMode, {
      phase: 'prefill',
      stepIndex: 0,
      seqLen: context.length,
      focusIndex: context.length - 1,
      outputToken: null,
      cacheLen: 0,
      reusedPositions: 0,
      recomputedPositions: context.length,
      workUnits: triangularWork(context.length),
      stepMs: runSeed.prefillMs,
      note: makeNote('baseline', 'prefill', context.length, 0, 0, 'none'),
    }));
    await yieldToBrowser();

    for (let step = 0; step < maxNewTokens; step++) {
      const nextToken = sampleNextToken(logits, model.vocabSize, temperature);
      context.push(nextToken);
      const decodeStart = performance.now();
      logits = model.forward([context]);
      const stepMs = performance.now() - decodeStart;
      runSeed.decodeMs += stepMs;
    steps.push(makeStepTrace(tokenizer, context, promptIds.length, model.config, runSeed.mode, runSeed.cacheMode, {
        phase: 'decode',
        stepIndex: step + 1,
        seqLen: context.length,
        focusIndex: context.length - 1,
        outputToken: createToken(nextToken, false, tokenizer),
        cacheLen: 0,
        reusedPositions: 0,
        recomputedPositions: context.length,
        workUnits: triangularWork(context.length),
        stepMs,
        note: makeNote('baseline', 'decode', context.length, 0, 0, 'none'),
      }));
      onProgress?.({ label: `Running baseline... ${step + 1}/${maxNewTokens}`, current: step + 1, total: maxNewTokens });
      await yieldToBrowser();
    }
  } finally {
    endNoGrad();
  }

  const totalMs = runSeed.prefillMs + runSeed.decodeMs;
  return {
    mode: 'baseline',
    cacheMode: 'none',
    prompt,
    promptTokens: promptIds.length,
    generatedTokens: maxNewTokens,
    temperature,
    blockSize: model.config.blockSize,
    nLayer: model.config.nLayer,
    nHead: model.config.nHead,
    replayCapacity: promptIds.length + maxNewTokens,
    totalMs,
    prefillMs: runSeed.prefillMs,
    decodeMs: runSeed.decodeMs,
    decodeMsPerToken: maxNewTokens > 0 ? runSeed.decodeMs / maxNewTokens : 0,
    tokensPerSec: totalMs > 0 ? (maxNewTokens * 1000) / totalMs : 0,
    cacheLen: 0,
    cacheBytesUsed: 0,
    cacheBytesCapacity: 0,
    text: tokenizer.decode(context),
    steps,
  };
}

async function runKvTrace(
  model: ReplayMiniGPT,
  tokenizer: BPETokenizer,
  prompt: string,
  maxNewTokens: number,
  temperature: number,
  cacheMode: Exclude<CacheMode, 'none'>,
  onProgress?: (progress: BenchmarkProgress) => void,
): Promise<RunTrace> {
  const promptIds = ensurePromptTokens(tokenizer, prompt);
  if (promptIds.length + maxNewTokens > model.config.blockSize) {
    throw new Error(
      `prompt_tokens + max_new_tokens must be <= blockSize (${promptIds.length + maxNewTokens} > ${model.config.blockSize})`,
    );
  }

  const mode: BenchmarkMode = cacheMode === 'int8' ? 'kv-int8' : 'kv-fp32';
  const caches = model.createCaches(cacheMode);
  const context = [...promptIds];
  const steps: StepTrace[] = [];
  const runSeed: RunSeed = {
    mode,
    cacheMode,
    prefillMs: 0,
    decodeMs: 0,
  };

  model.eval();
  startNoGrad();
  try {
    onProgress?.({ label: `Running ${mode} prefill...`, current: 0, total: maxNewTokens + 1 });
    const prefillStart = performance.now();
    let logits: Tensor | null = null;
    for (let index = 0; index < promptIds.length; index++) {
      logits = model.forwardToken(promptIds[index], index, caches);
    }
    runSeed.prefillMs = performance.now() - prefillStart;
    const prefillCacheLen = caches[0]?.length() ?? 0;
    steps.push(makeStepTrace(tokenizer, context, promptIds.length, model.config, runSeed.mode, runSeed.cacheMode, {
      phase: 'prefill',
      stepIndex: 0,
      seqLen: context.length,
      focusIndex: context.length - 1,
      outputToken: null,
      cacheLen: prefillCacheLen,
      reusedPositions: 0,
      recomputedPositions: context.length,
      workUnits: triangularWork(context.length),
      stepMs: runSeed.prefillMs,
      note: makeNote(mode, 'prefill', context.length, 0, prefillCacheLen, cacheMode),
    }));
    await yieldToBrowser();

    if (!logits) {
      throw new Error('kv prefill did not produce logits');
    }

    for (let step = 0; step < maxNewTokens; step++) {
      const nextToken = sampleNextToken(logits, model.vocabSize, temperature);
      context.push(nextToken);
      const decodeStart = performance.now();
      logits = model.forwardToken(nextToken, promptIds.length + step, caches);
      const stepMs = performance.now() - decodeStart;
      runSeed.decodeMs += stepMs;
      const cacheLen = caches[0]?.length() ?? 0;
      steps.push(makeStepTrace(tokenizer, context, promptIds.length, model.config, runSeed.mode, runSeed.cacheMode, {
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
        note: makeNote(mode, 'decode', context.length, Math.max(0, cacheLen - 1), cacheLen, cacheMode),
      }));
      onProgress?.({ label: `Running ${mode}... ${step + 1}/${maxNewTokens}`, current: step + 1, total: maxNewTokens });
      await yieldToBrowser();
    }
  } finally {
    endNoGrad();
    for (const cache of caches) cache.free();
  }

  const totalMs = runSeed.prefillMs + runSeed.decodeMs;
  const cacheLen = promptIds.length + maxNewTokens;
  const bytes = cacheBytesForMode(model.config, cacheLen, cacheMode);
  return {
    mode,
    cacheMode,
    prompt,
    promptTokens: promptIds.length,
    generatedTokens: maxNewTokens,
    temperature,
    blockSize: model.config.blockSize,
    nLayer: model.config.nLayer,
    nHead: model.config.nHead,
    replayCapacity: promptIds.length + maxNewTokens,
    totalMs,
    prefillMs: runSeed.prefillMs,
    decodeMs: runSeed.decodeMs,
    decodeMsPerToken: maxNewTokens > 0 ? runSeed.decodeMs / maxNewTokens : 0,
    tokensPerSec: totalMs > 0 ? (maxNewTokens * 1000) / totalMs : 0,
    cacheLen,
    cacheBytesUsed: bytes.usedBytes,
    cacheBytesCapacity: bytes.capacityBytes,
    text: tokenizer.decode(context),
    steps,
  };
}

export async function runBenchmarkSuite(
  model: ReplayMiniGPT,
  tokenizer: BPETokenizer,
  options: BenchmarkOptions,
): Promise<BenchmarkSuite> {
  const prompt = options.prompt;
  const maxNewTokens = options.maxNewTokens;
  const temperature = options.temperature ?? 0;
  const runs = {} as Record<BenchmarkMode, RunTrace>;

  for (const mode of MODE_ORDER) {
    if (mode === 'baseline') {
      runs[mode] = await runBaselineTrace(model, tokenizer, prompt, maxNewTokens, temperature, options.onProgress);
    } else if (mode === 'kv-fp32') {
      runs[mode] = await runKvTrace(model, tokenizer, prompt, maxNewTokens, temperature, 'fp32', options.onProgress);
    } else {
      runs[mode] = await runKvTrace(model, tokenizer, prompt, maxNewTokens, temperature, 'int8', options.onProgress);
    }
  }

  const baselineText = runs.baseline.text;
  const outputsMatch = MODE_ORDER.every(mode => runs[mode].text === baselineText);
  return {
    prompt,
    temperature,
    maxNewTokens,
    runs,
    outputsMatch,
  };
}

export async function runBenchmarkMode(
  model: ReplayMiniGPT,
  tokenizer: BPETokenizer,
  options: BenchmarkModeOptions,
): Promise<RunTrace> {
  const prompt = options.prompt;
  const maxNewTokens = options.maxNewTokens;
  const temperature = options.temperature ?? 0;

  if (options.mode === 'baseline') {
    return runBaselineTrace(model, tokenizer, prompt, maxNewTokens, temperature, options.onProgress);
  }
  if (options.mode === 'kv-fp32') {
    return runKvTrace(model, tokenizer, prompt, maxNewTokens, temperature, 'fp32', options.onProgress);
  }
  return runKvTrace(model, tokenizer, prompt, maxNewTokens, temperature, 'int8', options.onProgress);
}
