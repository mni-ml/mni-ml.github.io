import { useCallback, useEffect, useRef, useState } from 'react';
import { BPETokenizer, type TokenizerJSON } from '../../lib/inference/bpe';
import type { CheckpointData } from '../../lib/inference/model';
import {
  loadReplayModel,
  runBenchmarkMode,
  type BenchmarkMode,
  type RunTrace,
  type StepTrace,
} from '../../lib/inference/kvReplay';

const MODEL_URL =
  'https://media.githubusercontent.com/media/mni-ml/transformer/main/out/model-final.json';
const TOKENIZER_URL =
  'https://raw.githubusercontent.com/mni-ml/transformer/main/out/tokenizer.json';
const KNOWN_MODEL_SIZE = 251_000_000;

const DEFAULT_PROMPT = 'Once upon a time';
const MAX_GENERATED_TOKENS = 256;
const DEFAULT_TEMPERATURE = 0;
const MAX_TEMPERATURE = 0.5;
const MEMORY_GRID_SIZE = 12;
const MEMORY_GRID_CELLS = MEMORY_GRID_SIZE * MEMORY_GRID_SIZE;

type Phase = 'idle' | 'downloading' | 'parsing' | 'running' | 'ready' | 'error';

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
    { label: 'Total', value: formatMs(run.totalMs) },
    { label: 'ITL', value: formatMs(run.decodeMsPerToken) },
    { label: 'Throughput', value: formatRate(run.tokensPerSec) },
    {
      label: 'Cache used',
      value: run.cacheMode === 'none' ? '0 b' : formatBytes(step.cacheBytesUsed),
    },
  ];

  return (
    <div className="kv-metrics">
      {cards.map(card => (
        <div className="kv-metric" key={card.label}>
          <div className="kv-metric-label">{card.label}</div>
          <div className="kv-metric-value">{card.value}</div>
        </div>
      ))}
    </div>
  );
}

function DecodeFlowPane({
  run,
  step,
  previousStep,
}: {
  run: RunTrace;
  step: StepTrace;
  previousStep: StepTrace | null;
}) {
  const previousCacheLen = previousStep?.cacheLen ?? 0;
  const tokenOrdinal = step.focusIndex + 1;
  const headline = step.phase === 'prefill'
    ? `prompt prefill complete`
    : `token ${tokenOrdinal} gets processed`;
  const leftLabel = step.phase === 'prefill'
    ? `${step.seqLen} prompt tokens`
    : `token ${tokenOrdinal}`;
  const rightLabel = run.cacheMode === 'none'
    ? 'recompute prefix'
    : 'append new K/V entry';
  const cacheLine = run.cacheMode === 'none'
    ? 'cache stays disabled in baseline mode'
    : `cache length ${previousCacheLen} → ${step.cacheLen}`;

  return (
    <section className="kv-pane">
      <div className="kv-pane-header">
        <span className="kv-pane-title">decode step</span>
        <span className="kv-pane-meta">
          {step.phase === 'prefill' ? 'prefill' : `step ${step.stepIndex}`}
        </span>
      </div>

      <div className="kv-flow-headline">{headline}</div>

      <div className="kv-flow-strip">
        <div className="kv-flow-node">{leftLabel}</div>
        <div className="kv-flow-arrow">→</div>
        <div className="kv-flow-node is-action">{rightLabel}</div>
      </div>

      <div className="kv-flow-detail">{cacheLine}</div>
      <div className="kv-flow-note">{step.note}</div>
    </section>
  );
}

function MemorySquarePane({
  run,
  step,
  previousStep,
}: {
  run: RunTrace;
  step: StepTrace;
  previousStep: StepTrace | null;
}) {
  if (run.cacheMode === 'none') {
    return (
      <section className="kv-pane">
        <div className="kv-pane-header">
          <span className="kv-pane-title">kv memory</span>
          <span className="kv-pane-meta">disabled</span>
        </div>
        <div className="kv-memory-empty">
          baseline keeps no persistent kv allocation, so the memory square stays empty.
        </div>
      </section>
    );
  }

  const previousBytes = previousStep?.cacheBytesUsed ?? 0;
  const previousCacheLen = previousStep?.cacheLen ?? 0;
  const fillRatio = run.cacheBytesCapacity > 0
    ? step.cacheBytesUsed / run.cacheBytesCapacity
    : 0;
  const filledCells = Math.max(0, Math.min(
    MEMORY_GRID_CELLS,
    Math.round(fillRatio * MEMORY_GRID_CELLS),
  ));

  return (
    <section className="kv-pane">
      <div className="kv-pane-header">
        <span className="kv-pane-title">kv memory</span>
        <span className="kv-pane-meta">{run.cacheMode}</span>
      </div>

      <div className="kv-memory-metrics">
        <div className="kv-memory-stat">
          <span className="kv-memory-label">cache length</span>
          <span className="kv-memory-value">{previousCacheLen} → {step.cacheLen}</span>
        </div>
        <div className="kv-memory-stat">
          <span className="kv-memory-label">memory</span>
          <span className="kv-memory-value">
            {formatBytes(previousBytes)} → {formatBytes(step.cacheBytesUsed)}
          </span>
        </div>
      </div>

      <div
        className="kv-memory-square"
        style={{ gridTemplateColumns: `repeat(${MEMORY_GRID_SIZE}, minmax(0, 1fr))` }}
      >
        {Array.from({ length: MEMORY_GRID_CELLS }, (_, index) => (
          <div
            key={index}
            className={`kv-memory-cell${index < filledCells ? ' is-filled' : ''}`}
          />
        ))}
      </div>

      <div className="kv-memory-caption">
        {formatBytes(step.cacheBytesUsed)} of {formatBytes(run.cacheBytesCapacity)}
      </div>
    </section>
  );
}

export default function KvCacheDemo() {
  const [phase, setPhase] = useState<Phase>('idle');
  const [progress, setProgress] = useState({ loaded: 0, total: 0, label: '' });
  const [error, setError] = useState('');
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [temperature, setTemperature] = useState(clampTemperature(DEFAULT_TEMPERATURE));
  const [kvCacheEnabled, setKvCacheEnabled] = useState(true);
  const [kvQuantEnabled, setKvQuantEnabled] = useState(false);
  const [run, setRun] = useState<RunTrace | null>(null);

  const modelRef = useRef<ReturnType<typeof loadReplayModel> | null>(null);
  const tokenizerRef = useRef<BPETokenizer | null>(null);
  const outputRef = useRef<HTMLDivElement | null>(null);
  const activeRunIdRef = useRef(0);
  const stopRequestedRef = useRef(false);

  const isBenchmarkRunning = phase === 'running';
  const selectedMode: BenchmarkMode = !kvCacheEnabled
    ? 'baseline'
    : kvQuantEnabled
      ? 'kv-int8'
      : 'kv-fp32';

  const requestModeSwitch = useCallback((nextCacheEnabled: boolean, nextQuantEnabled: boolean) => {
    setKvCacheEnabled(nextCacheEnabled);
    setKvQuantEnabled(nextQuantEnabled);

    if (isBenchmarkRunning) {
      stopRequestedRef.current = true;
      setProgress(current => ({
        ...current,
        label: 'Stopping after the current decode step to switch modes...',
      }));
    }
  }, [isBenchmarkRunning]);

  const loadArtifacts = useCallback(async () => {
    setPhase('downloading');
    setProgress({ loaded: 0, total: 0, label: 'Downloading tokenizer...' });

    const tokRes = await fetch(TOKENIZER_URL);
    if (!tokRes.ok) throw new Error(`Tokenizer fetch failed: ${tokRes.status}`);
    const tokJson: TokenizerJSON = await tokRes.json();
    const tokenizer = new BPETokenizer(tokJson);
    tokenizerRef.current = tokenizer;

    setProgress({ loaded: 0, total: 0, label: 'Downloading model (~250 MB)...' });
    const modelRes = await fetch(MODEL_URL);
    if (!modelRes.ok) throw new Error(`Model fetch failed: ${modelRes.status}`);

    const contentLength = modelRes.headers.get('content-length');
    const parsedLength = contentLength ? parseInt(contentLength, 10) : 0;
    const totalBytes = parsedLength > 1_000_000 ? parsedLength : KNOWN_MODEL_SIZE;
    const reader = modelRes.body?.getReader();
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
        total: totalBytes,
        label: `Downloading model... ${(received / 1e6).toFixed(1)} / ${(totalBytes / 1e6).toFixed(0)} MB`,
      });
    }

    setPhase('parsing');
    setProgress({ loaded: 0, total: 0, label: 'Parsing checkpoint JSON...' });
    await new Promise(resolve => setTimeout(resolve, 20));

    const decoder = new TextDecoder();
    let json = '';
    for (const chunk of chunks) json += decoder.decode(chunk, { stream: true });
    json += decoder.decode();
    const checkpoint: CheckpointData = JSON.parse(json);

    setProgress({ loaded: 0, total: 0, label: 'Loading weights into the browser runtime...' });
    await new Promise(resolve => setTimeout(resolve, 20));
    modelRef.current = loadReplayModel(checkpoint, tokenizer);
  }, []);

  const runSelectedMode = useCallback(async () => {
    const model = modelRef.current;
    const tokenizer = tokenizerRef.current;
    if (!model || !tokenizer) throw new Error('model is not loaded yet');

    const promptTokens = tokenizer.encode(prompt);
    const maxNewTokens = Math.min(
      MAX_GENERATED_TOKENS,
      Math.max(0, model.config.blockSize - promptTokens.length),
    );
    if (maxNewTokens <= 0) {
      throw new Error(`prompt is too long for this ${model.config.blockSize}-token context window`);
    }

    const runTemperature = clampTemperature(temperature);
    const runId = activeRunIdRef.current + 1;
    activeRunIdRef.current = runId;
    stopRequestedRef.current = false;
    setRun(null);
    setPhase('running');
    setProgress({ loaded: 0, total: maxNewTokens, label: `Running prefill...` });

    const nextRun = await runBenchmarkMode(model, tokenizer, {
      mode: selectedMode,
      prompt,
      maxNewTokens,
      temperature: runTemperature,
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

    if (activeRunIdRef.current !== runId) return;
    setRun(nextRun);
    setPhase('ready');
  }, [prompt, selectedMode, temperature]);

  const handleLoadModel = useCallback(async () => {
    try {
      setError('');
      if (!modelRef.current || !tokenizerRef.current) {
        await loadArtifacts();
      }
      setPhase('ready');
    } catch (err: any) {
      setError(err?.message || 'Unknown error');
      setPhase('error');
    }
  }, [loadArtifacts]);

  const visibleRun = run;
  const visibleStepTrace = visibleRun
    ? visibleRun.steps[visibleRun.steps.length - 1] ?? null
    : null;
  const previousStepTrace = visibleRun && visibleRun.steps.length > 1
    ? visibleRun.steps[visibleRun.steps.length - 2]
    : null;

  useEffect(() => {
    if (!outputRef.current || !visibleStepTrace) return;
    outputRef.current.scrollTop = outputRef.current.scrollHeight;
  }, [visibleStepTrace]);

  return (
    <div className="kv-root">
      <div className="kv-header">
        <h1 className="kv-title">KV Cache Demo</h1>
        <p className="kv-desc">
          A 12M-parameter target LLM run with baseline recomputation, fp32 KV cache, or int8 KV
          cache, entirely in your browser using <a href="https://github.com/mni-ml/framework">@mni-ml/framework</a>.
        </p>
      </div>

      {phase === 'idle' && (
        <div className="kv-card">
          <p className="kv-muted">
            This demo downloads the model (~250 MB), and runs inference entirely in your browser with KV cache optimization and quantization.
          </p>
          <button onClick={handleLoadModel} className="demo-btn">
            Load Model & Start
          </button>
        </div>
      )}

      {(phase === 'downloading' || phase === 'parsing' || (phase === 'running' && !visibleRun)) && (
        <div className="kv-card">
          <div className="kv-muted">{progress.label}</div>
          <div className="kv-progress-track">
            <div
              className={`kv-progress-bar${phase === 'parsing' || !progress.total ? ' is-pulse' : ''}`}
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
            <div className="kv-muted" style={{ marginTop: 6 }}>
              {((progress.loaded / progress.total) * 100).toFixed(0)}%
            </div>
          )}
        </div>
      )}

      {phase === 'error' && (
        <div className="kv-card kv-card-error">
          <div className="kv-error-text">{error}</div>
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
          <div className="kv-section">
            <div className="kv-label" style={{ marginBottom: 8 }}>
              Prompt
            </div>
            <textarea
              className="kv-prompt"
              value={prompt}
              onChange={event => setPrompt(event.target.value)}
              disabled={isBenchmarkRunning}
              rows={2}
              spellCheck={false}
            />
          </div>

          <div className="kv-section kv-controls">
            <div className="kv-controls-main">
              <div className="kv-toggle-group">
                <label className="kv-toggle">
                  <input
                    type="checkbox"
                    checked={kvCacheEnabled}
                    onChange={event => {
                      const enabled = event.target.checked;
                      requestModeSwitch(enabled, enabled ? kvQuantEnabled : false);
                    }}
                    disabled={phase === 'downloading' || phase === 'parsing'}
                  />
                  <span>kv cache optimization</span>
                </label>

                <label className="kv-toggle">
                  <input
                    type="checkbox"
                    checked={kvQuantEnabled}
                    onChange={event => {
                      const enabled = event.target.checked;
                      requestModeSwitch(enabled ? true : kvCacheEnabled, enabled);
                    }}
                    disabled={phase === 'downloading' || phase === 'parsing'}
                  />
                  <span>kv quantization</span>
                </label>
              </div>

              <div className="kv-slider-wrap">
                <span className="kv-slider-label">
                  Temperature:{' '}
                  <span className="kv-slider-value">
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
                  className="kv-slider"
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
                  disabled={!prompt.trim() || phase === 'downloading' || phase === 'parsing'}
                >
                  Generate
                </button>
              )}
            </div>
          </div>

          <div className="kv-section">
            <div className="kv-label" style={{ marginBottom: 8 }}>
              Output
            </div>
            <div ref={outputRef} className="kv-output">
              {visibleStepTrace ? (
                visibleStepTrace.tokens.map((token, index) => (
                  <span
                    key={`${token.id}-${index}`}
                    className={[
                      token.isPrompt ? 'kv-token-prompt' : 'kv-token-gen',
                      index === visibleStepTrace.focusIndex ? 'kv-token-focus' : '',
                    ].join(' ').trim()}
                    title={`token ${token.id}`}
                  >
                    {token.text}
                  </span>
                ))
              ) : (
                <span className="kv-muted-inline">
                  Click Generate to produce up to {MAX_GENERATED_TOKENS} tokens.
                </span>
              )}
            </div>
          </div>

          <div className="kv-section">
            <div className="kv-label" style={{ marginBottom: 8 }}>
              Metrics
            </div>
            {visibleRun && visibleStepTrace ? (
              <MetricsSection run={visibleRun} step={visibleStepTrace} />
            ) : (
              <div className="kv-metrics">
                <div className="kv-metric">
                  <div className="kv-metric-label">Total</div>
                  <div className="kv-metric-value">—</div>
                </div>
                <div className="kv-metric">
                  <div className="kv-metric-label">Decode</div>
                  <div className="kv-metric-value">—</div>
                </div>
                <div className="kv-metric">
                  <div className="kv-metric-label">Throughput</div>
                  <div className="kv-metric-value">—</div>
                </div>
                <div className="kv-metric">
                  <div className="kv-metric-label">Cache used</div>
                  <div className="kv-metric-value">—</div>
                </div>
              </div>
            )}
          </div>

          <div className="kv-stage-grid">
            {/* <DecodeFlowPane
              run={visibleRun}
              step={visibleStepTrace}
              previousStep={previousStepTrace}
            /> */}
            {/* <MemorySquarePane
              run={visibleRun}
              step={visibleStepTrace}
              previousStep={previousStepTrace}
            /> */}
          </div>
        </>
      )}

      <style>{`
        .kv-root {
          max-width: 48rem;
          margin: 0 auto;
          padding: 36px 48px;
        }

        .kv-header {
          margin-bottom: 32px;
          padding-bottom: 20px;
          border-bottom: 0.5px solid var(--bdr);
        }

        .kv-title {
          font-size: 22px;
          font-weight: 500;
          letter-spacing: -0.03em;
          margin: 0 0 6px;
        }

        .kv-desc {
          font-size: 12.5px;
          color: var(--muted);
          margin: 0;
          line-height: 1.75;
        }

        .kv-desc a {
          color: var(--acc);
          text-decoration: none;
        }

        .kv-desc a:hover {
          text-decoration: underline;
          text-underline-offset: 3px;
        }

        .kv-card,
        .kv-pane,
        .kv-metric {
          border: 0.5px solid var(--bdr);
          border-radius: 6px;
          background: rgba(19, 25, 35, 0.48);
        }

        .kv-card {
          padding: 32px 24px;
          text-align: center;
        }

        .kv-card-error {
          border-color: #5c2020;
          background: #1a0d0d;
        }

        .kv-error-text {
          color: #f87171;
          font-size: 12px;
          margin-bottom: 12px;
          word-break: break-word;
        }

        .kv-muted {
          font-size: 12px;
          color: var(--muted);
          margin-bottom: 14px;
        }

        .kv-muted-inline {
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

        .kv-progress-track {
          width: 100%;
          max-width: 320px;
          height: 3px;
          background: var(--bdr);
          border-radius: 2px;
          overflow: hidden;
          margin: 0 auto;
        }

        .kv-progress-bar {
          height: 100%;
          background: var(--acc);
          border-radius: 2px;
          transition: width 0.3s ease;
        }

        .kv-progress-bar.is-pulse {
          animation: kv-pulse 1.5s ease-in-out infinite;
        }

        .kv-section {
          margin-bottom: 16px;
        }

        .kv-label,
        .kv-metric-label,
        .kv-memory-label {
          font-size: 10px;
          color: var(--muted);
          text-transform: uppercase;
          letter-spacing: 0.1em;
        }

        .kv-prompt {
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

        .kv-prompt:focus {
          outline: none;
          border-color: var(--acc);
        }

        .kv-controls {
          display: grid;
          gap: 12px;
        }

        .kv-controls-main {
          display: flex;
          flex-wrap: wrap;
          align-items: center;
          gap: 18px;
        }

        .kv-toggle-group {
          display: grid;
          gap: 6px;
          min-width: 9.5rem;
        }

        .kv-toggle {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          font-size: 12px;
          color: var(--txt);
          cursor: pointer;
          user-select: none;
        }

        .kv-toggle input {
          margin: 0;
          accent-color: var(--acc);
        }

        .kv-slider-wrap {
          display: inline-flex;
          align-items: center;
          gap: 12px;
          font-size: 12px;
          color: var(--muted);
          flex-wrap: wrap;
        }

        .kv-slider-label {
          font-family: var(--font-mono);
          font-size: 11px;
          color: var(--muted);
        }

        .kv-slider-value {
          color: var(--txt);
          display: inline-block;
          min-width: 56px;
        }

        .kv-slider {
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

        .kv-slider:disabled {
          opacity: 0.4;
          cursor: not-allowed;
        }

        .kv-slider::-webkit-slider-runnable-track {
          height: 2px;
          background: var(--bdr);
          border-radius: 1px;
        }

        .kv-slider::-moz-range-track {
          height: 2px;
          background: var(--bdr);
          border-radius: 1px;
        }

        .kv-slider::-webkit-slider-thumb {
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

        .kv-slider::-moz-range-thumb {
          width: 12px;
          height: 12px;
          border-radius: 50%;
          background: var(--acc);
          border: none;
          cursor: pointer;
          transition: transform 0.15s;
        }

        .kv-slider:hover::-webkit-slider-thumb {
          transform: scale(1.15);
        }

        .kv-slider:hover::-moz-range-thumb {
          transform: scale(1.15);
        }

        .kv-output {
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

        .kv-token-prompt {
          color: var(--muted);
        }

        .kv-token-gen {
          color: var(--txt);
        }

        .kv-token-focus {
          background: rgba(56, 189, 248, 0.12);
          border-radius: 3px;
          padding: 1px 2px;
        }

        .kv-metrics {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
          gap: 6px;
        }

        .kv-metric {
          padding: 10px 12px;
        }

        .kv-metric-value {
          font-family: var(--font-mono);
          font-size: 14px;
          color: var(--txt);
          margin-top: 4px;
        }

        .kv-generate-toolbar {
          display: flex;
          flex-wrap: wrap;
          align-items: center;
          gap: 16px;
        }

        .kv-progress-inline {
          color: var(--muted);
          font-size: 12px;
        }

        .kv-progress-copy {
          font-family: var(--font-mono);
        }

        .kv-stage-grid {
          display: grid;
          grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
          gap: 16px;
        }

        .kv-pane {
          padding: 14px;
        }

        .kv-pane-header {
          display: flex;
          justify-content: space-between;
          gap: 12px;
          margin-bottom: 10px;
        }

        .kv-pane-title {
          color: var(--txt);
        }

        .kv-pane-meta {
          color: var(--muted);
          font-size: 11px;
        }

        .kv-flow-headline {
          color: var(--txt);
          margin-bottom: 12px;
          font-size: 14px;
        }

        .kv-flow-strip {
          display: grid;
          grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
          gap: 10px;
          align-items: center;
          margin-bottom: 12px;
        }

        .kv-flow-node {
          border: 0.5px solid var(--bdr);
          border-radius: 6px;
          padding: 12px 10px;
          color: var(--txt);
          text-align: center;
          min-height: 3.5rem;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .kv-flow-node.is-action {
          background: rgba(56, 189, 248, 0.08);
          border-color: rgba(56, 189, 248, 0.3);
        }

        .kv-flow-arrow {
          color: var(--acc);
          font-size: 16px;
        }

        .kv-flow-detail,
        .kv-flow-note {
          color: var(--muted);
          font-size: 12px;
        }

        .kv-flow-detail {
          margin-bottom: 6px;
        }

        .kv-memory-metrics {
          display: grid;
          gap: 10px;
          margin-bottom: 12px;
        }

        .kv-memory-stat {
          display: flex;
          justify-content: space-between;
          gap: 12px;
          align-items: baseline;
        }

        .kv-memory-value {
          color: var(--txt);
          font-size: 12px;
        }

        .kv-memory-square {
          display: grid;
          gap: 3px;
          aspect-ratio: 1;
          margin-bottom: 10px;
        }

        .kv-memory-cell {
          border-radius: 2px;
          border: 0.5px solid rgba(26, 37, 53, 0.9);
          background: transparent;
          transition: background 0.16s ease, border-color 0.16s ease;
        }

        .kv-memory-cell.is-filled {
          background: rgba(56, 189, 248, 0.45);
          border-color: rgba(56, 189, 248, 0.26);
        }

        .kv-memory-caption,
        .kv-memory-empty {
          color: var(--muted);
          font-size: 12px;
          line-height: 1.7;
        }

        @keyframes kv-pulse {
          0%, 100% { opacity: 0.5; }
          50% { opacity: 1; }
        }

        @media (max-width: 768px) {
          .kv-root {
            padding: 24px 16px;
          }

          .kv-controls-main,
          .kv-stage-grid,
          .kv-flow-strip,
          .kv-generate-toolbar {
            grid-template-columns: 1fr;
            flex-direction: column;
            align-items: stretch;
          }

          .kv-slider {
            width: 100%;
          }

          .kv-memory-stat {
            flex-direction: column;
            gap: 4px;
          }
        }
      `}</style>
    </div>
  );
}
