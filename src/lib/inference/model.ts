import {
  Tensor, Module, Parameter, Linear, Embedding,
  gelu, dropout, layerNorm, flashAttention,
} from '@mni-ml/framework';

class LayerNorm extends Module {
  gamma!: Parameter<Tensor>;
  beta!: Parameter<Tensor>;
  dim: number;

  constructor(dim: number) {
    super();
    this.dim = dim;
    this.gamma = new Parameter(
      Tensor.fromFloat32(new Float32Array(dim).fill(1.0), [dim])
    );
    this.beta = new Parameter(
      Tensor.fromFloat32(new Float32Array(dim).fill(0.0), [dim])
    );
  }

  forward(x: Tensor): Tensor {
    return layerNorm(x, this.gamma.value, this.beta.value, 1e-5);
  }
}

class CausalSelfAttention extends Module {
  nHead: number;
  headDim: number;
  scale: number;
  dropoutRate: number;
  queryProj!: Linear;
  keyProj!: Linear;
  valueProj!: Linear;
  outProj!: Linear;

  constructor(nEmbd: number, nHead: number, dropoutRate: number) {
    super();
    this.nHead = nHead;
    this.headDim = nEmbd / nHead;
    this.scale = 1.0 / Math.sqrt(this.headDim);
    this.dropoutRate = dropoutRate;
    this.queryProj = new Linear(nEmbd, nEmbd);
    this.keyProj = new Linear(nEmbd, nEmbd);
    this.valueProj = new Linear(nEmbd, nEmbd);
    this.outProj = new Linear(nEmbd, nEmbd);
  }

  forward(x: Tensor): Tensor {
    const [B, S, E] = x.shape;
    const { nHead, headDim, scale } = this;
    let q = this.queryProj.forward(x);
    let k = this.keyProj.forward(x);
    let v = this.valueProj.forward(x);
    q = q.view(B, S, nHead, headDim).permute(0, 2, 1, 3).contiguous().view(B * nHead, S, headDim);
    k = k.view(B, S, nHead, headDim).permute(0, 2, 1, 3).contiguous().view(B * nHead, S, headDim);
    v = v.view(B, S, nHead, headDim).permute(0, 2, 1, 3).contiguous().view(B * nHead, S, headDim);
    let out = flashAttention(q, k, v, scale, true);
    out = out.view(B, nHead, S, headDim).permute(0, 2, 1, 3).contiguous().view(B, S, E);
    return this.outProj.forward(out);
  }
}

class FeedForward extends Module {
  dropoutRate: number;
  fc1!: Linear;
  fc2!: Linear;

  constructor(nEmbd: number, dropoutRate: number) {
    super();
    this.dropoutRate = dropoutRate;
    this.fc1 = new Linear(nEmbd, 4 * nEmbd);
    this.fc2 = new Linear(4 * nEmbd, nEmbd);
  }

  forward(x: Tensor): Tensor {
    let h = this.fc1.forward(x);
    h = gelu(h);
    h = this.fc2.forward(h);
    return dropout(h, this.dropoutRate, !this.training);
  }
}

class TransformerBlock extends Module {
  ln1!: LayerNorm;
  attn!: CausalSelfAttention;
  ln2!: LayerNorm;
  ffn!: FeedForward;

  constructor(nEmbd: number, nHead: number, dropoutRate: number) {
    super();
    this.ln1 = new LayerNorm(nEmbd);
    this.attn = new CausalSelfAttention(nEmbd, nHead, dropoutRate);
    this.ln2 = new LayerNorm(nEmbd);
    this.ffn = new FeedForward(nEmbd, dropoutRate);
  }

  forward(x: Tensor): Tensor {
    x = x.add(this.attn.forward(this.ln1.forward(x)));
    x = x.add(this.ffn.forward(this.ln2.forward(x)));
    return x;
  }
}

export interface ModelConfig {
  nEmbd: number;
  nHead: number;
  nLayer: number;
  blockSize: number;
  dropoutRate?: number;
  dKv?: number;
}

export class MiniGPT extends Module {
  config: ModelConfig;
  vocabSize: number;
  tokenEmb!: Embedding;
  posEmb!: Embedding;
  lnFinal!: LayerNorm;
  headBias!: Parameter<Tensor>;
  [key: string]: any;

  constructor(vocabSize: number, config: ModelConfig) {
    super();
    const { nEmbd, nHead, nLayer, blockSize, dropoutRate = 0 } = config;
    this.config = config;
    this.vocabSize = vocabSize;
    this.tokenEmb = new Embedding(vocabSize, nEmbd);
    this.posEmb = new Embedding(blockSize, nEmbd);
    for (let i = 0; i < nLayer; i++) {
      (this as any)[`block${i}`] = new TransformerBlock(nEmbd, nHead, dropoutRate);
    }
    this.lnFinal = new LayerNorm(nEmbd);
    this.headBias = new Parameter(Tensor.zeros([vocabSize]));
  }

  forward(indices: number[][]): Tensor {
    const batch = indices.length;
    const seqLen = indices[0].length;
    let x = this.tokenEmb.forward(indices);
    const posIndices: number[][] = [];
    for (let b = 0; b < batch; b++) {
      posIndices.push(Array.from({ length: seqLen }, (_, i) => i));
    }
    x = x.add(this.posEmb.forward(posIndices));
    for (let i = 0; i < this.config.nLayer; i++) {
      x = (this as any)[`block${i}`].forward(x);
    }
    x = this.lnFinal.forward(x);
    const wT = this.tokenEmb.weight.value.permute(1, 0);
    return x.matmul(wT).add(this.headBias.value);
  }
}

export interface CheckpointData {
  config: ModelConfig;
  parameters: Record<string, { shape: number[]; data: number[] }>;
  tokenizerPath?: string;
}

export function loadCheckpoint(
  checkpoint: CheckpointData,
  onProgress?: (loaded: number, total: number) => void,
): MiniGPT {
  const vocabSize = checkpoint.parameters['tokenEmb.weight']?.shape[0]
    ?? checkpoint.config.blockSize;
  const model = new MiniGPT(vocabSize, checkpoint.config);
  const namedParams = model.namedParameters();
  const total = namedParams.length;
  let loaded = 0;

  for (const [name, param] of namedParams) {
    const saved = checkpoint.parameters[name];
    if (saved) {
      const t = Tensor.fromFloat32(new Float32Array(saved.data), saved.shape);
      param.update(t);
    }
    loaded++;
    onProgress?.(loaded, total);
  }

  model.eval();
  return model;
}

export interface TopKToken {
  id: number;
  prob: number;
  text: string;
}

export function getTopKTokens(
  model: MiniGPT,
  context: number[],
  temperature: number = 0.8,
  topK: number = 10,
): { logits: Float32Array; topTokens: TopKToken[] } {
  const blockSize = model.config.blockSize;
  const ctxWindow = context.slice(-blockSize);
  const logits = model.forward([ctxWindow]);
  const seqLen = ctxWindow.length;
  const vocabSize = model.vocabSize;

  const logitsData = logits.toFloat32();
  const offset = (seqLen - 1) * vocabSize;
  const lastLogits = new Float32Array(vocabSize);
  for (let v = 0; v < vocabSize; v++) {
    lastLogits[v] = logitsData[offset + v] / temperature;
  }

  const maxLogit = lastLogits.reduce((a, b) => Math.max(a, b), -Infinity);
  const exps = new Float32Array(vocabSize);
  let sumExps = 0;
  for (let v = 0; v < vocabSize; v++) {
    exps[v] = Math.exp(lastLogits[v] - maxLogit);
    sumExps += exps[v];
  }

  const probs = new Float32Array(vocabSize);
  if (sumExps > 0 && isFinite(sumExps)) {
    for (let v = 0; v < vocabSize; v++) probs[v] = exps[v] / sumExps;
  }

  // Find top-K
  const indices = Array.from({ length: vocabSize }, (_, i) => i);
  indices.sort((a, b) => probs[b] - probs[a]);

  const topTokens: TopKToken[] = indices.slice(0, topK).map(idx => ({
    id: idx,
    prob: probs[idx],
    text: '',
  }));

  return { logits: probs, topTokens };
}
