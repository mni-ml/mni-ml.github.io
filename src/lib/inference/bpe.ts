/**
 * Browser-compatible BPE tokenizer.
 * Loads a HuggingFace tokenizers JSON object (ByteLevel pre-tokenizer).
 */

function buildByteEncoder(): { encoder: Record<number, string>; decoder: Record<string, number> } {
  const bs: number[] = [];
  for (let i = 0x21; i <= 0x7e; i++) bs.push(i);
  for (let i = 0xa1; i <= 0xac; i++) bs.push(i);
  for (let i = 0xae; i <= 0xff; i++) bs.push(i);

  const cs = [...bs];
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n++;
    }
  }

  const encoder: Record<number, string> = {};
  const decoder: Record<string, number> = {};
  for (let i = 0; i < bs.length; i++) {
    const ch = String.fromCodePoint(cs[i]);
    encoder[bs[i]] = ch;
    decoder[ch] = bs[i];
  }
  return { encoder, decoder };
}

const { encoder: BYTE_ENC, decoder: BYTE_DEC } = buildByteEncoder();

const GPT2_PAT =
  /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

export interface TokenizerJSON {
  model: {
    vocab: Record<string, number>;
    merges: string[] | string[][];
  };
  added_tokens?: Array<{ content: string; id: number }>;
}

export class BPETokenizer {
  vocab: Record<string, number>;
  idToToken: Record<number, string>;
  vocabSize: number;
  private bpeRanks: Record<string, number>;
  private addedByLength: Array<{ content: string; id: number }>;

  constructor(raw: TokenizerJSON) {
    const model = raw.model;
    this.vocab = { ...model.vocab };
    this.idToToken = {};
    for (const [tok, id] of Object.entries(this.vocab)) {
      this.idToToken[id] = tok;
    }

    this.addedByLength = [];
    if (raw.added_tokens) {
      for (const at of raw.added_tokens) {
        if (at.id !== undefined && at.content) {
          this.vocab[at.content] = at.id;
          this.idToToken[at.id] = at.content;
          this.addedByLength.push({ content: at.content, id: at.id });
        }
      }
      this.addedByLength.sort((a, b) => b.content.length - a.content.length);
    }

    const maxId = Math.max(...Object.keys(this.idToToken).map(Number));
    this.vocabSize = Math.max(Object.keys(this.vocab).length, maxId + 1);

    this.bpeRanks = {};
    if (model.merges) {
      for (let i = 0; i < model.merges.length; i++) {
        const m = model.merges[i];
        const key = Array.isArray(m) ? m.join(' ') : m;
        if (typeof key === 'string' && key.startsWith('#')) continue;
        this.bpeRanks[key] = i;
      }
    }
  }

  encode(text: string): number[] {
    if (!text) return [];
    for (const { content, id } of this.addedByLength) {
      if (text === content) return [id];
    }
    const matches = text.match(GPT2_PAT);
    if (!matches) return [];

    const ids: number[] = [];
    const encoder = new TextEncoder();

    for (const word of matches) {
      const bytes = encoder.encode(word);
      const unicoded = Array.from(bytes).map(b => BYTE_ENC[b]).join('');
      let pieces = [...unicoded];
      pieces = this.bpe(pieces);
      for (const piece of pieces) {
        const id = this.vocab[piece];
        if (id !== undefined) ids.push(id);
      }
    }
    return ids;
  }

  private bpe(pieces: string[]): string[] {
    while (pieces.length > 1) {
      let bestIdx = -1;
      let bestRank = Infinity;
      for (let i = 0; i < pieces.length - 1; i++) {
        const key = pieces[i] + ' ' + pieces[i + 1];
        const rank = this.bpeRanks[key];
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          bestIdx = i;
        }
      }
      if (bestIdx === -1) break;
      const merged = pieces[bestIdx] + pieces[bestIdx + 1];
      pieces = [...pieces.slice(0, bestIdx), merged, ...pieces.slice(bestIdx + 2)];
    }
    return pieces;
  }

  decode(ids: number[]): string {
    const strs = ids.map(id => this.idToToken[id] || '');
    const joined = strs.join('');
    const bytes: number[] = [];
    for (const ch of joined) {
      const b = BYTE_DEC[ch];
      if (b !== undefined) bytes.push(b);
    }
    return new TextDecoder().decode(new Uint8Array(bytes));
  }

  decodeToken(id: number): string {
    const tok = this.idToToken[id] || '';
    const bytes: number[] = [];
    for (const ch of tok) {
      const b = BYTE_DEC[ch];
      if (b !== undefined) bytes.push(b);
    }
    return new TextDecoder().decode(new Uint8Array(bytes));
  }

  get eotToken(): number {
    return this.vocab['<|endoftext|>'] ?? -1;
  }
}
