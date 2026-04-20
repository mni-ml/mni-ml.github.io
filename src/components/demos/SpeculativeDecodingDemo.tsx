import { useState, useRef, useCallback, useEffect } from 'react';
import { BPETokenizer, type TokenizerJSON } from '../../lib/inference/bpe';
import {
  MiniGPT, loadCheckpoint,
  type CheckpointData,
} from '../../lib/inference/model';

const TARGET_MODEL_URL =
  'https://media.githubusercontent.com/media/mni-ml/transformer/main/out/model-final.json';
const DRAFT_MODEL_URL =
  'https://media.githubusercontent.com/media/mni-ml/transformer/main/out/draft-final.json';
const TOKENIZER_URL =
  'https://raw.githubusercontent.com/mni-ml/transformer/main/out/tokenizer.json';

const KNOWN_TARGET_SIZE = 251_000_000;
const KNOWN_DRAFT_SIZE = 20_000_000;

const DEFAULT_PROMPT = 'There once was';
const MAX_NEW_TOKENS = 256;
const SPEC_K = 4;

type Phase =
  | 'idle'
  | 'downloading-tokenizer'
  | 'downloading-draft'
  | 'downloading-target'
  | 'parsing-draft'
  | 'parsing-target'
  | 'ready'
  | 'generating'
  | 'error';

interface TokenSpan {
  id: number;
  text: string;
  isPrompt: boolean;
  isPending: boolean;
}

interface Metrics {
  ttftMs: number | null;
  itlMs: number | null;
  throughput: number | null;
  totalMs: number | null;
  tokensGenerated: number;
  acceptRate: number | null;
}

const EMPTY_METRICS: Metrics = {
  ttftMs: null,
  itlMs: null,
  throughput: null,
  totalMs: null,
  tokensGenerated: 0,
  acceptRate: null,
};

function softmaxAtPosition(
  logitsFlat: Float32Array,
  pos: number,
  vocabSize: number,
  temperature: number,
): Float32Array {
  const offset = pos * vocabSize;
  const out = new Float32Array(vocabSize);

  if (temperature <= 1e-3) {
    let maxIdx = 0;
    let maxVal = logitsFlat[offset];
    for (let v = 1; v < vocabSize; v++) {
      const x = logitsFlat[offset + v];
      if (x > maxVal) {
        maxVal = x;
        maxIdx = v;
      }
    }
    out[maxIdx] = 1;
    return out;
  }

  let max = -Infinity;
  for (let v = 0; v < vocabSize; v++) {
    const x = logitsFlat[offset + v] / temperature;
    out[v] = x;
    if (x > max) max = x;
  }
  let sum = 0;
  for (let v = 0; v < vocabSize; v++) {
    out[v] = Math.exp(out[v] - max);
    sum += out[v];
  }
  if (!isFinite(sum) || sum <= 0) {
    out.fill(1 / vocabSize);
    return out;
  }
  for (let v = 0; v < vocabSize; v++) out[v] /= sum;
  return out;
}

function sampleFromProbs(probs: Float32Array): number {
  const r = Math.random();
  let cum = 0;
  for (let v = 0; v < probs.length; v++) {
    cum += probs[v];
    if (r < cum) return v;
  }
  return probs.length - 1;
}

function forwardLogitsFlat(model: MiniGPT, ctxWindow: number[]): Float32Array {
  const logits = model.forward([ctxWindow]);
  return logits.toFloat32();
}

export default function SpeculativeDecodingDemo() {
  const [phase, setPhase] = useState<Phase>('idle');
  const [progress, setProgress] = useState({ loaded: 0, total: 0, label: '' });
  const [error, setError] = useState('');

  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [temperature, setTemperature] = useState(0.1);
  const [useSpec, setUseSpec] = useState(true);

  const [tokens, setTokens] = useState<TokenSpan[]>([]);
  const [metrics, setMetrics] = useState<Metrics>(EMPTY_METRICS);

  const targetRef = useRef<MiniGPT | null>(null);
  const draftRef = useRef<MiniGPT | null>(null);
  const tokenizerRef = useRef<BPETokenizer | null>(null);
  const cancelRef = useRef(false);
  const outputRef = useRef<HTMLDivElement>(null);

  const isGenerating = phase === 'generating';

  useEffect(() => {
    if (outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight;
    }
  }, [tokens]);

  const downloadJson = useCallback(
    async (
      url: string,
      knownSize: number,
      label: string,
    ): Promise<any> => {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`Fetch failed (${res.status}): ${url}`);
      const contentLength = res.headers.get('content-length');
      const parsed = contentLength ? parseInt(contentLength, 10) : 0;
      const total = parsed > 1_000_000 ? parsed : knownSize;
      const reader = res.body?.getReader();
      if (!reader) throw new Error('ReadableStream not supported');

      const chunks: Uint8Array[] = [];
      let received = 0;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        received += value.length;
        setProgress({
          loaded: received,
          total,
          label: `${label} ${(received / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(0)} MB`,
        });
      }

      const decoder = new TextDecoder();
      let jsonStr = '';
      for (const c of chunks) jsonStr += decoder.decode(c, { stream: true });
      jsonStr += decoder.decode();
      return JSON.parse(jsonStr);
    },
    [],
  );

  const handleStart = useCallback(async () => {
    try {
      setError('');

      setPhase('downloading-tokenizer');
      setProgress({ loaded: 0, total: 0, label: 'Downloading tokenizer...' });
      const tokRes = await fetch(TOKENIZER_URL);
      if (!tokRes.ok) throw new Error(`Tokenizer fetch failed: ${tokRes.status}`);
      const tokJson: TokenizerJSON = await tokRes.json();
      tokenizerRef.current = new BPETokenizer(tokJson);

      setPhase('downloading-draft');
      const draftJson = await downloadJson(
        DRAFT_MODEL_URL,
        KNOWN_DRAFT_SIZE,
        'Downloading draft model...',
      );
      setPhase('parsing-draft');
      setProgress({ loaded: 0, total: 0, label: 'Loading draft weights...' });
      await new Promise(r => setTimeout(r, 20));
      draftRef.current = loadCheckpoint(draftJson as CheckpointData);

      setPhase('downloading-target');
      const targetJson = await downloadJson(
        TARGET_MODEL_URL,
        KNOWN_TARGET_SIZE,
        'Downloading target model...',
      );
      setPhase('parsing-target');
      setProgress({ loaded: 0, total: 0, label: 'Loading target weights...' });
      await new Promise(r => setTimeout(r, 20));
      targetRef.current = loadCheckpoint(targetJson as CheckpointData);

      setPhase('ready');
    } catch (e: any) {
      setError(e.message || 'Unknown error');
      setPhase('error');
    }
  }, [downloadJson]);

  const generateStandard = useCallback(
    async (
      target: MiniGPT,
      tokenizer: BPETokenizer,
      promptIds: number[],
      temp: number,
    ) => {
      const blockSize = target.config.blockSize;
      const vocabSize = target.vocabSize;
      const eot = tokenizer.eotToken;
      const context = [...promptIds];

      const t0 = performance.now();
      let firstTokenTime = 0;
      const tokenTimes: number[] = [];

      for (let i = 0; i < MAX_NEW_TOKENS; i++) {
        if (cancelRef.current) break;
        const ctxWindow = context.slice(-blockSize);
        const logits = forwardLogitsFlat(target, ctxWindow);
        const probs = softmaxAtPosition(
          logits,
          ctxWindow.length - 1,
          vocabSize,
          temp,
        );
        const tok = sampleFromProbs(probs);
        context.push(tok);

        const now = performance.now();
        if (i === 0) firstTokenTime = now - t0;
        tokenTimes.push(now);

        setTokens(prev => [
          ...prev,
          { id: tok, text: tokenizer.decodeToken(tok), isPrompt: false, isPending: false },
        ]);

        const generated = i + 1;
        const totalMs = now - t0;
        setMetrics({
          ttftMs: firstTokenTime,
          itlMs:
            generated > 1
              ? (totalMs - firstTokenTime) / (generated - 1)
              : null,
          throughput: totalMs > 0 ? (generated / totalMs) * 1000 : null,
          totalMs,
          tokensGenerated: generated,
          acceptRate: null,
        });

        await new Promise(r => setTimeout(r, 0));
        if (eot >= 0 && tok === eot) break;
      }
    },
    [],
  );

  const generateSpeculative = useCallback(
    async (
      target: MiniGPT,
      draft: MiniGPT,
      tokenizer: BPETokenizer,
      promptIds: number[],
      temp: number,
    ) => {
      const blockSize = target.config.blockSize;
      const vocabSize = target.vocabSize;
      const eot = tokenizer.eotToken;
      const context = [...promptIds];

      const t0 = performance.now();
      let firstTokenTime = 0;
      let totalGenerated = 0;
      let totalProposed = 0;
      let totalAccepted = 0;
      let stopOnEot = false;

      const recordMetrics = (now: number) => {
        if (totalGenerated === 1) firstTokenTime = now - t0;
        const totalMs = now - t0;
        setMetrics({
          ttftMs: firstTokenTime,
          itlMs:
            totalGenerated > 1
              ? (totalMs - firstTokenTime) / (totalGenerated - 1)
              : null,
          throughput:
            totalMs > 0 ? (totalGenerated / totalMs) * 1000 : null,
          totalMs,
          tokensGenerated: totalGenerated,
          acceptRate:
            totalProposed > 0 ? totalAccepted / totalProposed : null,
        });
      };

      while (totalGenerated < MAX_NEW_TOKENS && !stopOnEot) {
        if (cancelRef.current) break;
        const remaining = MAX_NEW_TOKENS - totalGenerated;
        const k = Math.min(SPEC_K, Math.max(1, remaining));
        const draftStartIdx = promptIds.length + totalGenerated;

        const drafts: number[] = [];
        const draftProbs: Float32Array[] = [];

        for (let i = 0; i < k; i++) {
          if (cancelRef.current) break;
          const ctxWin = [...context, ...drafts].slice(-blockSize);
          const logitsFlat = forwardLogitsFlat(draft, ctxWin);
          const probs = softmaxAtPosition(
            logitsFlat,
            ctxWin.length - 1,
            vocabSize,
            temp,
          );
          const tok = sampleFromProbs(probs);
          drafts.push(tok);
          draftProbs.push(probs);

          setTokens(prev => [
            ...prev,
            {
              id: tok,
              text: tokenizer.decodeToken(tok),
              isPrompt: false,
              isPending: true,
            },
          ]);

          await new Promise(r => setTimeout(r, 0));
        }

        if (cancelRef.current) break;

        const ctxAll = [...context, ...drafts].slice(-blockSize);
        const W = ctxAll.length;
        const targetLogits = forwardLogitsFlat(target, ctxAll);

        let rejectedThisIter = false;
        let processed = 0;
        for (let i = 0; i < k; i++) {
          const pos = W - k - 1 + i;
          let confirmedTok: number;
          let didReject = false;

          if (pos < 0) {
            const tProbs = softmaxAtPosition(
              targetLogits,
              W - 1,
              vocabSize,
              temp,
            );
            confirmedTok = sampleFromProbs(tProbs);
            didReject = true;
          } else {
            const tProbs = softmaxAtPosition(targetLogits, pos, vocabSize, temp);
            const dProbs = draftProbs[i];
            const tok = drafts[i];
            const pt = tProbs[tok];
            const pd = Math.max(dProbs[tok], 1e-10);
            const ratio = pt / pd;
            const r = Math.random();
            totalProposed++;
            if (r < Math.min(1, ratio)) {
              totalAccepted++;
              confirmedTok = tok;
            } else {
              const adj = new Float32Array(vocabSize);
              let sum = 0;
              for (let v = 0; v < vocabSize; v++) {
                const x = Math.max(0, tProbs[v] - dProbs[v]);
                adj[v] = x;
                sum += x;
              }
              if (sum > 0) {
                for (let v = 0; v < vocabSize; v++) adj[v] /= sum;
              } else {
                adj.set(tProbs);
              }
              confirmedTok = sampleFromProbs(adj);
              didReject = true;
            }
          }

          const tokenIdx = draftStartIdx + i;
          if (didReject) {
            const cutLen = tokenIdx + 1;
            const txt = tokenizer.decodeToken(confirmedTok);
            setTokens(prev => {
              const next = prev.slice(0, cutLen);
              next[tokenIdx] = {
                id: confirmedTok,
                text: txt,
                isPrompt: false,
                isPending: false,
              };
              return next;
            });
          } else {
            setTokens(prev =>
              prev.map((t, j) =>
                j === tokenIdx ? { ...t, isPending: false } : t,
              ),
            );
          }

          context.push(confirmedTok);
          totalGenerated++;
          processed++;
          recordMetrics(performance.now());

          if (eot >= 0 && confirmedTok === eot) {
            stopOnEot = true;
            break;
          }
          if (didReject) {
            rejectedThisIter = true;
            break;
          }
        }

        if (
          !rejectedThisIter &&
          !stopOnEot &&
          processed === k &&
          totalGenerated < MAX_NEW_TOKENS
        ) {
          const bonusProbs = softmaxAtPosition(
            targetLogits,
            W - 1,
            vocabSize,
            temp,
          );
          const bonusTok = sampleFromProbs(bonusProbs);
          const txt = tokenizer.decodeToken(bonusTok);
          setTokens(prev => [
            ...prev,
            { id: bonusTok, text: txt, isPrompt: false, isPending: false },
          ]);
          context.push(bonusTok);
          totalGenerated++;
          recordMetrics(performance.now());
          if (eot >= 0 && bonusTok === eot) stopOnEot = true;
        }

        await new Promise(r => setTimeout(r, 0));
      }

      setTokens(prev => prev.filter(t => !t.isPending));
    },
    [],
  );

  const handleGenerate = useCallback(async () => {
    const tokenizer = tokenizerRef.current;
    const target = targetRef.current;
    const draft = draftRef.current;
    if (!tokenizer || !target || !draft) return;

    cancelRef.current = false;
    setMetrics(EMPTY_METRICS);

    const promptIds = tokenizer.encode(prompt);
    setTokens(
      promptIds.map(id => ({
        id,
        text: tokenizer.decodeToken(id),
        isPrompt: true,
        isPending: false,
      })),
    );

    setPhase('generating');
    try {
      if (useSpec) {
        await generateSpeculative(target, draft, tokenizer, promptIds, temperature);
      } else {
        await generateStandard(target, tokenizer, promptIds, temperature);
      }
    } catch (e: any) {
      setError(`Generation error: ${e.message}`);
      setPhase('error');
      return;
    }
    setPhase('ready');
  }, [prompt, temperature, useSpec, generateSpeculative, generateStandard]);

  const handleStop = useCallback(() => {
    cancelRef.current = true;
  }, []);

  const isLoadingPhase =
    phase === 'downloading-tokenizer' ||
    phase === 'downloading-draft' ||
    phase === 'downloading-target' ||
    phase === 'parsing-draft' ||
    phase === 'parsing-target';

  const formatMs = (n: number | null) =>
    n === null ? '—' : n < 10 ? n.toFixed(2) + ' ms' : n.toFixed(0) + ' ms';
  const formatTok = (n: number | null) =>
    n === null ? '—' : n.toFixed(1) + ' tok/s';
  const formatPct = (n: number | null) =>
    n === null ? '—' : (n * 100).toFixed(0) + '%';

  return (
    <div className="demo-root">
      <div className="demo-header">
        <h1 className="demo-title">Speculative Decoding Demo</h1>
        <p className="demo-desc">
          A 12M-parameter target LLM accelerated by a small draft model, both
          running entirely in your browser using{' '}
          <a href="https://github.com/mni-ml/framework">@mni-ml/framework</a>.
          Toggle speculative decoding on or off and compare the throughput.
        </p>
      </div>

      {phase === 'idle' && (
        <div className="demo-card demo-idle">
          <p className="demo-muted">
            This demo downloads the target model (~250 MB) and a small draft
            model (~20 MB), then runs inference entirely in your browser.
          </p>
          <button onClick={handleStart} className="demo-btn">
            Load Models &amp; Start
          </button>
        </div>
      )}

      {isLoadingPhase && (
        <div className="demo-card demo-loading">
          <div className="demo-muted">{progress.label}</div>
          <div className="demo-progress-track">
            <div
              className={`demo-progress-bar${
                phase === 'parsing-draft' ||
                phase === 'parsing-target' ||
                !progress.total
                  ? ' demo-progress-pulse'
                  : ''
              }`}
              style={{
                width:
                  progress.total > 0
                    ? `${(progress.loaded / progress.total) * 100}%`
                    : phase === 'parsing-draft' || phase === 'parsing-target'
                    ? '100%'
                    : '0%',
              }}
            />
          </div>
          {progress.total > 0 && (
            <div className="demo-muted" style={{ marginTop: 6 }}>
              {((progress.loaded / progress.total) * 100).toFixed(0)}%
            </div>
          )}
        </div>
      )}

      {phase === 'error' && (
        <div className="demo-card demo-error">
          <div className="demo-error-text">{error}</div>
          <button
            onClick={() => {
              setPhase('idle');
              setError('');
            }}
            className="demo-btn"
          >
            Try Again
          </button>
        </div>
      )}

      {(phase === 'ready' || phase === 'generating') && (
        <>
          <div className="demo-section">
            <div className="demo-label" style={{ marginBottom: 8 }}>
              Prompt
            </div>
            <textarea
              className="demo-prompt"
              value={prompt}
              onChange={e => setPrompt(e.target.value)}
              disabled={isGenerating}
              rows={2}
            />
          </div>

          <div className="demo-section demo-controls">
            <label className="demo-toggle">
              <input
                type="checkbox"
                checked={useSpec}
                onChange={e => setUseSpec(e.target.checked)}
                disabled={isGenerating}
              />
              <span>Speculative decoding</span>
            </label>

            <div className="demo-slider-wrap">
              <span className="demo-slider-label">
                Temperature:{' '}
                <span className="demo-slider-value">
                  {temperature.toFixed(2)}
                </span>
              </span>
              <input
                type="range"
                min="0"
                max="0.5"
                step="0.05"
                value={temperature}
                onChange={e => setTemperature(parseFloat(e.target.value))}
                disabled={isGenerating}
                className="demo-slider"
              />
            </div>

            {isGenerating ? (
              <button onClick={handleStop} className="demo-btn">
                Stop
              </button>
            ) : (
              <button
                onClick={handleGenerate}
                className="demo-btn demo-btn-primary"
                disabled={!prompt.trim()}
              >
                Generate
              </button>
            )}
          </div>

          <div className="demo-section">
            <div className="demo-label" style={{ marginBottom: 8 }}>
              Output
            </div>
            <div ref={outputRef} className="demo-output">
              {tokens.length === 0 ? (
                <span className="demo-muted-inline">
                  Click Generate to produce up to {MAX_NEW_TOKENS} tokens.
                </span>
              ) : (
                tokens.map((t, i) => (
                  <span
                    key={i}
                    className={
                      t.isPrompt
                        ? 'demo-tok-prompt'
                        : t.isPending
                        ? 'demo-tok-pending'
                        : 'demo-tok-gen'
                    }
                    title={`token ${t.id}${t.isPending ? ' (draft)' : ''}`}
                  >
                    {t.text}
                  </span>
                ))
              )}
              {isGenerating && <span className="demo-cursor">▊</span>}
            </div>
          </div>

          <div className="demo-section">
            <div className="demo-label" style={{ marginBottom: 8 }}>
              Metrics
            </div>
            <div className="demo-metrics">
              <div className="demo-metric">
                <div className="demo-metric-label">Acceptance rate</div>
                <div className="demo-metric-value">
                  {formatPct(metrics.acceptRate)}
                </div>
              </div>
              <div className="demo-metric">
                <div className="demo-metric-label">ITL</div>
                <div className="demo-metric-value">
                  {formatMs(metrics.itlMs)}
                </div>
              </div>
              <div className="demo-metric">
                <div className="demo-metric-label">Throughput</div>
                <div className="demo-metric-value">
                  {formatTok(metrics.throughput)}
                </div>
              </div>
              <div className="demo-metric">
                <div className="demo-metric-label">Tokens</div>
                <div className="demo-metric-value">
                  {metrics.tokensGenerated || '—'}
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      <style>{`
        .demo-root {
          max-width: var(--content-width);
          margin: 0 auto;
          padding: 36px 48px;
        }

        .demo-header {
          margin-bottom: 32px;
          padding-bottom: 20px;
          border-bottom: 0.5px solid var(--bdr);
        }

        .demo-title {
          font-size: 22px;
          font-weight: 500;
          letter-spacing: -0.03em;
          margin: 0 0 6px;
        }

        .demo-desc {
          font-size: 12.5px;
          color: var(--muted);
          margin: 0;
          line-height: 1.75;
        }

        .demo-desc a {
          color: var(--acc);
          text-decoration: none;
        }

        .demo-desc a:hover {
          text-decoration: underline;
          text-underline-offset: 3px;
        }

        .demo-card {
          border: 0.5px solid var(--bdr);
          border-radius: 6px;
          padding: 32px 24px;
          text-align: center;
        }

        .demo-error {
          border-color: #5c2020;
          background: #1a0d0d;
        }

        .demo-error-text {
          color: #f87171;
          font-size: 12px;
          margin-bottom: 12px;
          word-break: break-word;
        }

        .demo-muted {
          font-size: 12px;
          color: var(--muted);
          margin-bottom: 14px;
        }

        .demo-muted-inline {
          font-size: 12px;
          color: var(--muted);
        }

        .demo-btn {
          font-family: var(--font-mono);
          font-size: 11px;
          color: var(--muted);
          background: none;
          border: 0.5px solid var(--bdr);
          border-radius: 4px;
          padding: 5px 12px;
          cursor: pointer;
          transition: color 0.2s, border-color 0.2s;
        }

        .demo-btn:hover {
          color: var(--acc);
          border-color: var(--acc);
        }

        .demo-btn:disabled {
          opacity: 0.4;
          cursor: not-allowed;
        }

        .demo-btn-primary {
          color: var(--acc);
          border-color: var(--acc);
        }

        .demo-progress-track {
          width: 100%;
          max-width: 320px;
          height: 3px;
          background: var(--bdr);
          border-radius: 2px;
          overflow: hidden;
          margin: 0 auto;
        }

        .demo-progress-bar {
          height: 100%;
          background: var(--acc);
          border-radius: 2px;
          transition: width 0.3s ease;
        }

        .demo-progress-pulse {
          animation: demo-pulse 1.5s ease-in-out infinite;
        }

        .demo-section {
          margin-bottom: 16px;
        }

        .demo-label {
          font-size: 10px;
          color: var(--muted);
          text-transform: uppercase;
          letter-spacing: 0.1em;
        }

        .demo-prompt {
          width: 100%;
          background: none;
          border: 0.5px solid var(--bdr);
          border-radius: 6px;
          padding: 10px 12px;
          font-family: var(--font-mono);
          font-size: 12.5px;
          color: var(--txt);
          resize: vertical;
          line-height: 1.6;
        }

        .demo-prompt:focus {
          outline: none;
          border-color: var(--acc);
        }

        .demo-controls {
          display: flex;
          flex-wrap: wrap;
          align-items: center;
          gap: 18px;
        }

        .demo-toggle {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          font-size: 12px;
          color: var(--txt);
          cursor: pointer;
          user-select: none;
        }

        .demo-toggle input {
          margin: 0;
          accent-color: var(--acc);
        }

        .demo-slider-wrap {
          display: inline-flex;
          align-items: center;
          gap: 12px;
          font-size: 12px;
          color: var(--muted);
        }

        .demo-slider-label {
          font-family: var(--font-mono);
          font-size: 11px;
          color: var(--muted);
        }

        .demo-slider-value {
          color: var(--txt);
          display: inline-block;
          min-width: 48px;
        }

        .demo-slider {
          -webkit-appearance: none;
          appearance: none;
          width: 140px;
          height: 2px;
          background: var(--bdr);
          border-radius: 1px;
          outline: none;
          cursor: pointer;
          margin: 0;
          padding: 0;
        }

        .demo-slider:disabled {
          opacity: 0.4;
          cursor: not-allowed;
        }

        .demo-slider::-webkit-slider-runnable-track {
          height: 2px;
          background: var(--bdr);
          border-radius: 1px;
        }

        .demo-slider::-moz-range-track {
          height: 2px;
          background: var(--bdr);
          border-radius: 1px;
        }

        .demo-slider::-webkit-slider-thumb {
          -webkit-appearance: none;
          appearance: none;
          width: 12px;
          height: 12px;
          margin-top: -5px;
          border-radius: 50%;
          background: var(--acc);
          border: none;
          cursor: pointer;
          transition: transform 0.15s;
        }

        .demo-slider::-moz-range-thumb {
          width: 12px;
          height: 12px;
          border-radius: 50%;
          background: var(--acc);
          border: none;
          cursor: pointer;
          transition: transform 0.15s;
        }

        .demo-slider:hover::-webkit-slider-thumb {
          transform: scale(1.15);
        }

        .demo-slider:hover::-moz-range-thumb {
          transform: scale(1.15);
        }

        .demo-output {
          border: 0.5px solid var(--bdr);
          border-radius: 6px;
          padding: 14px 16px;
          min-height: 8rem;
          max-height: 18rem;
          overflow-y: auto;
          font-size: 12.5px;
          line-height: 1.75;
          white-space: pre-wrap;
          word-break: break-word;
        }

        .demo-tok-prompt { color: var(--muted); }
        .demo-tok-gen { color: var(--txt); transition: color 0.15s ease; }
        .demo-tok-pending {
          color: var(--acc);
          opacity: 0.45;
          font-style: italic;
          transition: opacity 0.15s ease;
        }

        .demo-cursor {
          color: var(--acc);
          animation: demo-pulse 1s ease-in-out infinite;
        }

        .demo-metrics {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
          gap: 6px;
        }

        .demo-metric {
          border: 0.5px solid var(--bdr);
          border-radius: 6px;
          padding: 10px 12px;
        }

        .demo-metric-label {
          font-size: 10px;
          color: var(--muted);
          text-transform: uppercase;
          letter-spacing: 0.1em;
          margin-bottom: 4px;
        }

        .demo-metric-value {
          font-family: var(--font-mono);
          font-size: 14px;
          color: var(--txt);
        }

        @keyframes demo-pulse {
          0%, 100% { opacity: 0.5; }
          50% { opacity: 1; }
        }

        @media (max-width: 768px) {
          .demo-root {
            padding: 24px 16px;
          }
          .demo-controls {
            flex-direction: column;
            align-items: stretch;
            gap: 12px;
          }
          .demo-slider {
            width: 100%;
          }
        }
      `}</style>
    </div>
  );
}
