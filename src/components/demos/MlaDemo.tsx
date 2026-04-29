import { useCallback, useEffect, useRef, useState } from 'react';
import { BPETokenizer, type TokenizerJSON } from '../../lib/inference/bpe';
import type { CheckpointData } from '../../lib/inference/model';
import {
  loadReplayModel,
  runBenchmarkMode,
  type RunTrace,
  type StepTrace,
} from '../../lib/inference/kvReplay';
import {
  loadMlaModel,
  runMlaTrace,
  type MlaCheckpointData,
} from '../../lib/inference/mlaReplay';

const MHA_MODEL_URL =
  'https://media.githubusercontent.com/media/mni-ml/transformer/main/out/model-final.json';
const MLA_MODEL_URL =
  'https://media.githubusercontent.com/media/mni-ml/transformer/main/out/model-mla-final.json';
const TOKENIZER_URL =
  'https://raw.githubusercontent.com/mni-ml/transformer/main/out/tokenizer.json';

const DEFAULT_PROMPT = 'Once upon a time';
const MAX_GENERATED_TOKENS = 256;
const DEFAULT_TEMPERATURE = 0;
const MAX_TEMPERATURE = 0.5;

type Phase = 'idle' | 'downloading' | 'parsing' | 'running' | 'ready' | 'error';
type AttentionMode = 'mha' | 'mla';

function clampTemperature(value: number): number {
  return Math.max(0, Math.min(MAX_TEMPERATURE, value));
}

function formatMs(value: number | null): string {
  if (value == null) return '—';
  return `${value.toFixed(1)} ms`;
}

function formatRate(value: number | null): string {
  if (value == null) return '—';
  return `${value.toFixed(2)} tok/s`;
}

function formatBytes(bytes: number): string {
  if (bytes <= 0) return '0 b';
  if (bytes < 1024) return `${bytes} b`;
  const units = ['kb', 'mb', 'gb'];
  let value = bytes;
  let unitIndex = -1;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(2)} ${units[unitIndex]}`;
}

function MetricsSection({ run, step }: { run: RunTrace; step: StepTrace }) {
  const cards = [
    { label: 'Cache used', value: formatBytes(step.cacheBytesUsed), highlight: true },
    { label: 'Total', value: formatMs(run.totalMs) },
    { label: 'ITL', value: formatMs(run.decodeMsPerToken) },
    { label: 'Throughput', value: formatRate(run.tokensPerSec) },
  ];

  return (
    <div className="mla-metrics">
      {cards.map(card => (
        <div className={`mla-metric${card.highlight ? ' is-highlight' : ''}`} key={card.label}>
          <div className="mla-metric-label">{card.label}</div>
          <div className="mla-metric-value">{card.value}</div>
        </div>
      ))}
    </div>
  );
}

interface ProgressState {
  loaded: number;
  total: number;
  label: string;
}

interface DownloadResult {
  json: any;
  bytes: number;
}

async function downloadJson(
  url: string,
  label: string,
  onProgress: (p: ProgressState) => void,
): Promise<DownloadResult> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch failed (${res.status}): ${url}`);
  const contentLength = res.headers.get('content-length');
  const total = contentLength ? parseInt(contentLength, 10) : 0;
  const reader = res.body?.getReader();
  if (!reader) throw new Error('ReadableStream not supported');

  const chunks: Uint8Array[] = [];
  let received = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    const progressLabel = total > 0
      ? `${label} ${(received / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(1)} MB`
      : `${label} ${(received / 1e6).toFixed(1)} MB`;
    onProgress({ loaded: received, total, label: progressLabel });
  }

  const decoder = new TextDecoder();
  let json = '';
  for (const chunk of chunks) json += decoder.decode(chunk, { stream: true });
  json += decoder.decode();
  return { json: JSON.parse(json), bytes: received };
}

export default function MlaDemo() {
  const [phase, setPhase] = useState<Phase>('idle');
  const [progress, setProgress] = useState<ProgressState>({ loaded: 0, total: 0, label: '' });
  const [error, setError] = useState('');
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [temperature, setTemperature] = useState(clampTemperature(DEFAULT_TEMPERATURE));
  const [useMla, setUseMla] = useState(true);
  const [run, setRun] = useState<RunTrace | null>(null);

  const mhaModelRef = useRef<ReturnType<typeof loadReplayModel> | null>(null);
  const mlaModelRef = useRef<ReturnType<typeof loadMlaModel> | null>(null);
  const tokenizerRef = useRef<BPETokenizer | null>(null);
  const outputRef = useRef<HTMLDivElement | null>(null);
  const activeRunIdRef = useRef(0);
  const stopRequestedRef = useRef(false);

  const isBenchmarkRunning = phase === 'running';
  const isLoadingPhase = phase === 'downloading' || phase === 'parsing';

  const requestModeSwitch = useCallback((nextUseMla: boolean) => {
    setUseMla(nextUseMla);
    if (isBenchmarkRunning) {
      stopRequestedRef.current = true;
      setProgress(current => ({
        ...current,
        label: 'Stopping after the current decode step to switch modes...',
      }));
    }
  }, [isBenchmarkRunning]);

  const ensureTokenizer = useCallback(async () => {
    if (tokenizerRef.current) return tokenizerRef.current;
    setPhase('downloading');
    setProgress({ loaded: 0, total: 0, label: 'Downloading tokenizer...' });
    const tokRes = await fetch(TOKENIZER_URL);
    if (!tokRes.ok) throw new Error(`tokenizer fetch failed: ${tokRes.status}`);
    const tokJson: TokenizerJSON = await tokRes.json();
    const tokenizer = new BPETokenizer(tokJson);
    tokenizerRef.current = tokenizer;
    return tokenizer;
  }, []);

  const ensureMhaModel = useCallback(async () => {
    if (mhaModelRef.current) return mhaModelRef.current;
    const tokenizer = await ensureTokenizer();
    setPhase('downloading');
    const { json } = await downloadJson(
      MHA_MODEL_URL,
      'Downloading MHA model...',
      setProgress,
    );
    setPhase('parsing');
    setProgress({ loaded: 0, total: 0, label: 'Loading MHA weights into runtime...' });
    await new Promise(resolve => setTimeout(resolve, 20));
    const model = loadReplayModel(json as CheckpointData, tokenizer);
    mhaModelRef.current = model;
    return model;
  }, [ensureTokenizer]);

  const ensureMlaModel = useCallback(async () => {
    if (mlaModelRef.current) return mlaModelRef.current;
    const tokenizer = await ensureTokenizer();
    setPhase('downloading');
    const { json } = await downloadJson(
      MLA_MODEL_URL,
      'Downloading MLA model...',
      setProgress,
    );
    setPhase('parsing');
    setProgress({ loaded: 0, total: 0, label: 'Loading MLA weights into runtime...' });
    await new Promise(resolve => setTimeout(resolve, 20));
    const model = loadMlaModel(json as MlaCheckpointData, tokenizer);
    mlaModelRef.current = model;
    return model;
  }, [ensureTokenizer]);

  const runSelectedMode = useCallback(async () => {
    const tokenizer = await ensureTokenizer();

    let blockSize: number;
    let runner: (
      maxNewTokens: number,
      runId: number,
    ) => Promise<RunTrace>;

    if (useMla) {
      const model = await ensureMlaModel();
      blockSize = model.config.blockSize;
      runner = async (maxNewTokens, runId) => runMlaTrace(model, tokenizer, {
        prompt,
        maxNewTokens,
        temperature: clampTemperature(temperature),
        onProgress: runProgress => {
          setProgress({
            loaded: runProgress.current,
            total: runProgress.total,
            label: runProgress.label,
          });
        },
        onStep: partialRun => {
          if (activeRunIdRef.current !== runId) return;
          setRun(partialRun);
        },
        shouldStop: () => activeRunIdRef.current !== runId || stopRequestedRef.current,
      });
    } else {
      const model = await ensureMhaModel();
      blockSize = model.config.blockSize;
      runner = async (maxNewTokens, runId) => runBenchmarkMode(model, tokenizer, {
        mode: 'kv-fp32',
        prompt,
        maxNewTokens,
        temperature: clampTemperature(temperature),
        onProgress: runProgress => {
          setProgress({
            loaded: runProgress.current,
            total: runProgress.total,
            label: runProgress.label,
          });
        },
        onStep: partialRun => {
          if (activeRunIdRef.current !== runId) return;
          setRun(partialRun);
        },
        shouldStop: () => activeRunIdRef.current !== runId || stopRequestedRef.current,
      });
    }

    const promptTokens = tokenizer.encode(prompt);
    const maxNewTokens = Math.min(
      MAX_GENERATED_TOKENS,
      Math.max(0, blockSize - promptTokens.length),
    );
    if (maxNewTokens <= 0) {
      throw new Error(`prompt is too long for this ${blockSize}-token context window`);
    }

    const runId = activeRunIdRef.current + 1;
    activeRunIdRef.current = runId;
    stopRequestedRef.current = false;
    setRun(null);
    setPhase('running');
    setProgress({ loaded: 0, total: maxNewTokens, label: 'Running prefill...' });

    const nextRun = await runner(maxNewTokens, runId);
    if (activeRunIdRef.current !== runId) return;
    setRun(nextRun);
    setPhase('ready');
  }, [ensureMhaModel, ensureMlaModel, ensureTokenizer, prompt, temperature, useMla]);

  const handleStart = useCallback(async () => {
    try {
      setError('');
      await ensureTokenizer();
      await ensureMhaModel();
      await ensureMlaModel();
      setPhase('ready');
    } catch (err: any) {
      setError(err?.message || 'Unknown error');
      setPhase('error');
    }
  }, [ensureMhaModel, ensureMlaModel, ensureTokenizer]);

  const visibleRun = run;
  const visibleStepTrace = visibleRun
    ? visibleRun.steps[visibleRun.steps.length - 1] ?? null
    : null;

  useEffect(() => {
    if (!outputRef.current || !visibleStepTrace) return;
    outputRef.current.scrollTop = outputRef.current.scrollHeight;
  }, [visibleStepTrace]);

  return (
    <div className="mla-root">
      <div className="mla-header">
        <h1 className="mla-title">Multi-head Latent Attention Demo</h1>
        <p className="mla-desc">
          Compare the KV cache footprint of standard multi-head attention against multi-head
          latent attention (MLA), running entirely in your browser using{' '}
          <a href="https://github.com/mni-ml/framework">@mni-ml/framework</a>. MLA stores a single
          low-rank latent per token instead of full K/V vectors.
        </p>
      </div>

      {phase === 'idle' && (
        <div className="mla-card">
          <p className="mla-muted">
            Downloads both the MHA and MLA checkpoints, then decodes with a KV cache so you
            can compare per-token cache memory. The toggle below switches between models
            instantly once both are loaded.
          </p>
          <button onClick={handleStart} className="demo-btn">
            Load Models &amp; Start
          </button>
        </div>
      )}

      {(isLoadingPhase || (phase === 'running' && !visibleRun)) && (
        <div className="mla-card">
          <div className="mla-muted">{progress.label}</div>
          <div className="mla-progress-track">
            <div
              className={`mla-progress-bar${phase === 'parsing' || !progress.total ? ' is-pulse' : ''}`}
              style={{
                width: progress.total > 0
                  ? `${(progress.loaded / progress.total) * 100}%`
                  : phase === 'parsing'
                    ? '100%'
                    : '0%',
              }}
            />
          </div>
          {progress.total > 0 && (
            <div className="mla-muted" style={{ marginTop: 6 }}>
              {((progress.loaded / progress.total) * 100).toFixed(0)}%
            </div>
          )}
        </div>
      )}

      {phase === 'error' && (
        <div className="mla-card mla-card-error">
          <div className="mla-error-text">{error}</div>
          <button
            onClick={() => {
              setError('');
              setPhase('idle');
            }}
            className="demo-btn"
          >
            try again
          </button>
        </div>
      )}

      {(visibleRun || phase === 'running' || phase === 'ready') && (
        <>
          <div className="mla-section">
            <div className="mla-label" style={{ marginBottom: 8 }}>
              Prompt
            </div>
            <textarea
              className="mla-prompt"
              value={prompt}
              onChange={event => setPrompt(event.target.value)}
              disabled={isBenchmarkRunning}
              rows={2}
              spellCheck={false}
            />
          </div>

          <div className="mla-section mla-controls">
            <div className="mla-controls-main">
              <div className="mla-toggle-group">
                <label className="mla-toggle">
                  <input
                    type="checkbox"
                    checked={useMla}
                    onChange={event => requestModeSwitch(event.target.checked)}
                    disabled={isLoadingPhase}
                  />
                  <span>multi-head latent attention</span>
                </label>
              </div>

              <div className="mla-slider-wrap">
                <span className="mla-slider-label">
                  Temperature:{' '}
                  <span className="mla-slider-value">
                    {temperature.toFixed(2)}
                  </span>
                </span>
                <input
                  type="range"
                  min="0"
                  max={MAX_TEMPERATURE}
                  step="0.05"
                  value={temperature}
                  onChange={event => setTemperature(clampTemperature(parseFloat(event.target.value)))}
                  disabled={isBenchmarkRunning}
                  className="mla-slider"
                />
              </div>

              {isBenchmarkRunning ? (
                <button
                  className="demo-btn"
                  onClick={() => {
                    stopRequestedRef.current = true;
                    setProgress(current => ({
                      ...current,
                      label: 'Stopping after the current decode step...',
                    }));
                  }}
                >
                  Stop
                </button>
              ) : (
                <button
                  className="demo-btn"
                  onClick={() => {
                    runSelectedMode().catch((err: any) => {
                      setError(err?.message || 'Unknown error');
                      setPhase('error');
                    });
                  }}
                  disabled={!prompt.trim() || isLoadingPhase}
                >
                  Generate
                </button>
              )}
            </div>
          </div>

          <div className="mla-section">
            <div className="mla-label" style={{ marginBottom: 8 }}>
              Output{' '}
              <span className="mla-mode-pill">
                {useMla ? 'mla' : 'mha + fp32 kv'}
              </span>
            </div>
            <div ref={outputRef} className="mla-output">
              {visibleStepTrace ? (
                visibleStepTrace.tokens.map((token, index) => (
                  <span
                    key={`${token.id}-${index}`}
                    className={[
                      token.isPrompt ? 'mla-token-prompt' : 'mla-token-gen',
                      index === visibleStepTrace.focusIndex ? 'mla-token-focus' : '',
                    ].join(' ').trim()}
                    title={`token ${token.id}`}
                  >
                    {token.text}
                  </span>
                ))
              ) : (
                <span className="mla-muted-inline">
                  Click Generate to produce up to {MAX_GENERATED_TOKENS} tokens.
                </span>
              )}
            </div>
          </div>

          <div className="mla-section">
            <div className="mla-label" style={{ marginBottom: 8 }}>
              Metrics
            </div>
            {visibleRun && visibleStepTrace ? (
              <MetricsSection run={visibleRun} step={visibleStepTrace} />
            ) : (
              <div className="mla-metrics">
                <div className="mla-metric is-highlight">
                  <div className="mla-metric-label">Cache used</div>
                  <div className="mla-metric-value">—</div>
                </div>
                <div className="mla-metric">
                  <div className="mla-metric-label">Total</div>
                  <div className="mla-metric-value">—</div>
                </div>
                <div className="mla-metric">
                  <div className="mla-metric-label">ITL</div>
                  <div className="mla-metric-value">—</div>
                </div>
                <div className="mla-metric">
                  <div className="mla-metric-label">Throughput</div>
                  <div className="mla-metric-value">—</div>
                </div>
              </div>
            )}
          </div>
        </>
      )}

      <style>{`
        .mla-root {
          max-width: 48rem;
          margin: 0 auto;
          padding: 36px 48px;
        }

        .mla-header {
          margin-bottom: 32px;
          padding-bottom: 20px;
          border-bottom: 0.5px solid var(--bdr);
        }

        .mla-title {
          font-size: 22px;
          font-weight: 500;
          letter-spacing: -0.03em;
          margin: 0 0 6px;
        }

        .mla-desc {
          font-size: 12.5px;
          color: var(--muted);
          margin: 0;
          line-height: 1.75;
        }

        .mla-desc a {
          color: var(--acc);
          text-decoration: none;
        }

        .mla-desc a:hover {
          text-decoration: underline;
          text-underline-offset: 3px;
        }

        .mla-desc code {
          font-family: var(--font-mono);
          font-size: 11.5px;
          color: var(--txt);
        }

        .mla-card,
        .mla-metric {
          border: 0.5px solid var(--bdr);
          border-radius: 6px;
          background: rgba(19, 25, 35, 0.48);
        }

        .mla-card {
          padding: 32px 24px;
          text-align: center;
        }

        .mla-card-error {
          border-color: #5c2020;
          background: #1a0d0d;
        }

        .mla-error-text {
          color: #f87171;
          font-size: 12px;
          margin-bottom: 12px;
          word-break: break-word;
        }

        .mla-muted {
          font-size: 12px;
          color: var(--muted);
          margin-bottom: 14px;
        }

        .mla-muted-inline {
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

        .mla-progress-track {
          width: 100%;
          max-width: 320px;
          height: 3px;
          background: var(--bdr);
          border-radius: 2px;
          overflow: hidden;
          margin: 0 auto;
        }

        .mla-progress-bar {
          height: 100%;
          background: var(--acc);
          border-radius: 2px;
          transition: width 0.3s ease;
        }

        .mla-progress-bar.is-pulse {
          animation: mla-pulse 1.5s ease-in-out infinite;
        }

        .mla-section {
          margin-bottom: 16px;
        }

        .mla-label,
        .mla-metric-label {
          font-size: 10px;
          color: var(--muted);
          text-transform: uppercase;
          letter-spacing: 0.1em;
        }

        .mla-mode-pill {
          font-family: var(--font-mono);
          font-size: 10px;
          color: var(--acc);
          border: 0.5px solid var(--acc);
          border-radius: 4px;
          padding: 1px 6px;
          margin-left: 6px;
          letter-spacing: 0.05em;
          text-transform: lowercase;
        }

        .mla-prompt {
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

        .mla-prompt:focus {
          outline: none;
          border-color: var(--acc);
        }

        .mla-controls {
          display: grid;
          gap: 12px;
        }

        .mla-controls-main {
          display: flex;
          flex-wrap: wrap;
          align-items: center;
          gap: 18px;
        }

        .mla-toggle-group {
          display: grid;
          gap: 6px;
          min-width: 9.5rem;
        }

        .mla-toggle {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          font-size: 12px;
          color: var(--txt);
          cursor: pointer;
          user-select: none;
        }

        .mla-toggle input {
          margin: 0;
          accent-color: var(--acc);
        }

        .mla-slider-wrap {
          display: inline-flex;
          align-items: center;
          gap: 12px;
          font-size: 12px;
          color: var(--muted);
          flex-wrap: wrap;
        }

        .mla-slider-label {
          font-family: var(--font-mono);
          font-size: 11px;
          color: var(--muted);
        }

        .mla-slider-value {
          color: var(--txt);
          display: inline-block;
          min-width: 56px;
        }

        .mla-slider {
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

        .mla-slider:disabled {
          opacity: 0.4;
          cursor: not-allowed;
        }

        .mla-slider::-webkit-slider-runnable-track {
          height: 2px;
          background: var(--bdr);
          border-radius: 1px;
        }

        .mla-slider::-moz-range-track {
          height: 2px;
          background: var(--bdr);
          border-radius: 1px;
        }

        .mla-slider::-webkit-slider-thumb {
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

        .mla-slider::-moz-range-thumb {
          width: 12px;
          height: 12px;
          border-radius: 50%;
          background: var(--acc);
          border: none;
          cursor: pointer;
          transition: transform 0.15s;
        }

        .mla-slider:hover::-webkit-slider-thumb {
          transform: scale(1.15);
        }

        .mla-slider:hover::-moz-range-thumb {
          transform: scale(1.15);
        }

        .mla-output {
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

        .mla-token-prompt {
          color: var(--muted);
        }

        .mla-token-gen {
          color: var(--txt);
        }

        .mla-token-focus {
          background: rgba(56, 189, 248, 0.12);
          border-radius: 3px;
          padding: 1px 2px;
        }

        .mla-metrics {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
          gap: 6px;
        }

        .mla-metric {
          padding: 10px 12px;
        }

        .mla-metric.is-highlight {
          border-color: rgba(56, 189, 248, 0.45);
          background: rgba(56, 189, 248, 0.06);
        }

        .mla-metric-value {
          font-family: var(--font-mono);
          font-size: 14px;
          color: var(--txt);
          margin-top: 4px;
        }

        @keyframes mla-pulse {
          0%, 100% { opacity: 0.5; }
          50% { opacity: 1; }
        }

        @media (max-width: 768px) {
          .mla-root {
            padding: 24px 16px;
          }

          .mla-controls-main {
            flex-direction: column;
            align-items: stretch;
          }

          .mla-slider {
            width: 100%;
          }
        }
      `}</style>
    </div>
  );
}
