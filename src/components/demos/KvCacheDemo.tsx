import { useCallback, useEffect, useRef, useState } from 'react';
import { BPETokenizer, type TokenizerJSON } from '../../lib/inference/bpe';
import type { CheckpointData } from '../../lib/inference/model';
import {
  loadReplayModel,
  runBenchmarkSuite,
  type BenchmarkMode,
  type BenchmarkSuite,
  type RunTrace,
  type StepTrace,
} from '../../lib/inference/kvReplay';

const MODEL_URL =
  'https://media.githubusercontent.com/media/mni-ml/transformer/main/out/model-final.json';
const TOKENIZER_URL =
  'https://raw.githubusercontent.com/mni-ml/transformer/main/out/tokenizer.json';
const KNOWN_MODEL_SIZE = 251_000_000;

const DEFAULT_PROMPT = 'Once upon a time';
const DEFAULT_TOKENS = 32;
const DEFAULT_TEMPERATURE = 0;
const TOKEN_OPTIONS = [8, 16, 32, 64];
const MODE_ORDER: BenchmarkMode[] = ['baseline', 'kv-fp32', 'kv-int8'];

type Phase = 'idle' | 'downloading' | 'parsing' | 'running' | 'ready' | 'error';

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

function modeLabel(mode: BenchmarkMode): string {
  if (mode === 'kv-fp32') return 'kv fp32';
  if (mode === 'kv-int8') return 'kv int8';
  return 'baseline';
}

function formatReplayValue(step: StepTrace): string {
  if (step.phase === 'prefill') return 'prefill';
  return `decode ${step.stepIndex}`;
}

function MatrixPane({ run, step }: { run: RunTrace; step: StepTrace }) {
  const size = Math.min(step.seqLen, run.replayCapacity);
  const cells = [];

  for (let row = 0; row < size; row++) {
    for (let col = 0; col < size; col++) {
      let className = 'kv-matrix-cell';
      if (col > row) {
        className += ' is-empty';
      } else if (step.phase === 'prefill') {
        className += ' is-active';
      } else if (run.cacheMode === 'none') {
        className += row === size - 1 ? ' is-focus' : ' is-active';
      } else {
        className += row === size - 1 ? ' is-focus' : ' is-history';
      }

      cells.push(
        <div
          key={`${row}-${col}`}
          className={className}
          title={`row ${row + 1}, col ${col + 1}`}
        />,
      );
    }
  }

  return (
    <section className="kv-pane">
      <div className="kv-pane-header">
        <span className="kv-pane-title">causal work</span>
        <span className="kv-pane-meta">{size} × {size}</span>
      </div>
      <div
        className="kv-matrix"
        style={{ gridTemplateColumns: `repeat(${size || 1}, minmax(0, 1fr))` }}
      >
        {cells}
      </div>
    </section>
  );
}

function CachePane({ run, step }: { run: RunTrace; step: StepTrace }) {
  if (run.cacheMode === 'none') {
    return (
      <section className="kv-pane">
        <div className="kv-pane-header">
          <span className="kv-pane-title">kv cache</span>
          <span className="kv-pane-meta">disabled</span>
        </div>
        <div className="kv-cache-empty">
          baseline keeps no persistent kv memory and recomputes the prefix every decode step
        </div>
      </section>
    );
  }

  const displayCapacity = run.replayCapacity;
  const filled = Math.min(step.cacheLen, displayCapacity);
  const laneCells = Array.from({ length: displayCapacity }, (_, index) => index < filled);

  return (
    <section className="kv-pane">
      <div className="kv-pane-header">
        <span className="kv-pane-title">kv cache</span>
        <span className="kv-pane-meta">{run.cacheMode}</span>
      </div>
      <div className={`kv-cache-stack cache-${run.cacheMode}`}>
        {Array.from({ length: run.nLayer }, (_, layer) => (
          <div className="kv-cache-layer" key={layer}>
            <div className="kv-cache-label">L{layer + 1}</div>
            <div className="kv-cache-lane-wrap">
              {(['K', 'V'] as const).map(kind => (
                <div className="kv-cache-lane-row" key={kind}>
                  <span className="kv-cache-kind">{kind}</span>
                  <div className="kv-cache-lane">
                    {laneCells.map((isFilled, index) => (
                      <div
                        key={`${layer}-${kind}-${index}`}
                        className={`kv-cache-cell${isFilled ? ' is-filled' : ''}`}
                        title={`layer ${layer + 1}, ${kind}, token ${index + 1}, dtype=${run.cacheMode}`}
                      />
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function MetricsSection({ run, step }: { run: RunTrace; step: StepTrace }) {
  const cards = [
    { label: 'Total', value: formatMs(run.totalMs) },
    { label: 'Decode', value: formatMs(run.decodeMsPerToken) },
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

export default function KvCacheDemo() {
  const [phase, setPhase] = useState<Phase>('idle');
  const [progress, setProgress] = useState({ loaded: 0, total: 0, label: '' });
  const [error, setError] = useState('');
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [maxNewTokens, setMaxNewTokens] = useState(DEFAULT_TOKENS);
  const [temperature, setTemperature] = useState(DEFAULT_TEMPERATURE);
  const [selectedMode, setSelectedMode] = useState<BenchmarkMode>('kv-fp32');
  const [suite, setSuite] = useState<BenchmarkSuite | null>(null);
  const [replayStep, setReplayStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  const modelRef = useRef<ReturnType<typeof loadReplayModel> | null>(null);
  const tokenizerRef = useRef<BPETokenizer | null>(null);

  const isRunning = phase === 'running';

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

  const runBenchmarks = useCallback(async () => {
    const model = modelRef.current;
    const tokenizer = tokenizerRef.current;
    if (!model || !tokenizer) throw new Error('model is not loaded yet');

    setIsPlaying(false);
    setReplayStep(0);
    setPhase('running');
    setProgress({ loaded: 0, total: maxNewTokens, label: 'Running baseline prefill...' });

    const nextSuite = await runBenchmarkSuite(model, tokenizer, {
      prompt,
      maxNewTokens,
      temperature,
      onProgress: setProgress,
    });

    setSuite(nextSuite);
    setReplayStep(0);
    setPhase('ready');
  }, [maxNewTokens, prompt, temperature]);

  const handleLoadAndRun = useCallback(async () => {
    try {
      setError('');
      if (!modelRef.current || !tokenizerRef.current) {
        await loadArtifacts();
      }
      await runBenchmarks();
    } catch (err: any) {
      setError(err?.message || 'Unknown error');
      setPhase('error');
    }
  }, [loadArtifacts, runBenchmarks]);

  useEffect(() => {
    if (!suite || !isPlaying) return undefined;
    const run = suite.runs[selectedMode];
    const maxStep = run.steps.length - 1;
    if (replayStep >= maxStep) {
      setIsPlaying(false);
      return undefined;
    }
    const timer = window.setTimeout(() => {
      setReplayStep(current => Math.min(current + 1, maxStep));
    }, 650);
    return () => window.clearTimeout(timer);
  }, [isPlaying, replayStep, selectedMode, suite]);

  const selectedRun = suite?.runs[selectedMode] ?? null;
  const selectedStep = selectedRun?.steps[replayStep] ?? null;

  return (
    <div className="kv-root">
      <div className="kv-header">
        <h1 className="kv-title">KV Cache Replay</h1>
        <p className="kv-desc">
          Replay the same transformer run three ways: baseline recomputation, KV cache with fp32
          storage, and KV cache with int8 storage. The browser measures the run, then replays the
          work and memory story one step at a time.
        </p>
      </div>

      {phase === 'idle' && (
        <div className="kv-card">
          <p className="kv-muted">
            This demo downloads the 12M transformer checkpoint (~250 MB), runs the benchmark in your
            browser, and replays prefill plus decode step by step.
          </p>
          <button type="button" onClick={handleLoadAndRun} className="kv-btn">
            load model and start
          </button>
        </div>
      )}

      {(phase === 'downloading' || phase === 'parsing' || (phase === 'running' && !suite)) && (
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
            type="button"
            onClick={() => {
              setError('');
              setPhase('idle');
            }}
            className="kv-btn"
          >
            try again
          </button>
        </div>
      )}

      {(suite || phase === 'running') && (
        <>
          <div className="kv-section">
            <div className="kv-label" style={{ marginBottom: 8 }}>
              Prompt
            </div>
            <textarea
              className="kv-prompt"
              value={prompt}
              onChange={event => setPrompt(event.target.value)}
              disabled={isRunning}
              rows={2}
              spellCheck={false}
            />
          </div>

          <div className="kv-section kv-controls">
            <div className="kv-control-group">
              <span className="kv-control-label">mode</span>
              <div className="kv-mode-group">
                {MODE_ORDER.map(mode => (
                  <button
                    type="button"
                    key={mode}
                    className={`kv-mode-btn${selectedMode === mode ? ' is-active' : ''}`}
                    onClick={() => {
                      setSelectedMode(mode);
                      setReplayStep(0);
                      setIsPlaying(false);
                    }}
                  >
                    {modeLabel(mode)}
                  </button>
                ))}
              </div>
            </div>

            <div className="kv-control-group">
              <span className="kv-control-label">generated tokens</span>
              <div className="kv-chip-row">
                {TOKEN_OPTIONS.map(value => (
                  <button
                    type="button"
                    key={value}
                    className={`kv-chip${maxNewTokens === value ? ' is-active' : ''}`}
                    onClick={() => setMaxNewTokens(value)}
                    disabled={isRunning}
                  >
                    {value}
                  </button>
                ))}
              </div>
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
                max="1"
                step="0.05"
                value={temperature}
                onChange={event => setTemperature(parseFloat(event.target.value))}
                disabled={isRunning}
                className="kv-slider"
              />
            </div>

            <button
              type="button"
              onClick={handleLoadAndRun}
              className="kv-btn kv-btn-primary"
              disabled={!prompt.trim() || isRunning}
            >
              {isRunning ? 'running…' : 'run benchmark'}
            </button>
          </div>

          {isRunning && suite && (
            <div className="kv-status">
              {progress.label}
            </div>
          )}

          {suite && selectedRun && selectedStep && (
            <>
              <div className="kv-section">
                <div className="kv-label" style={{ marginBottom: 8 }}>
                  Output
                </div>
                <div className="kv-output">
                  {selectedStep.tokens.map((token, index) => (
                    <span
                      key={`${token.id}-${index}`}
                      className={[
                        token.isPrompt ? 'kv-token-prompt' : 'kv-token-gen',
                        index === selectedStep.focusIndex ? 'kv-token-focus' : '',
                      ].join(' ').trim()}
                      title={`token ${token.id}`}
                    >
                      {token.text}
                    </span>
                  ))}
                </div>
              </div>

              <div className="kv-section">
                <div className="kv-label" style={{ marginBottom: 8 }}>
                  Metrics
                </div>
                <MetricsSection run={selectedRun} step={selectedStep} />
              </div>

              <div className="kv-section">
                <div className="kv-label" style={{ marginBottom: 8 }}>
                  Replay
                </div>
                <div className="kv-replay-toolbar">
                  <div className="kv-replay-buttons">
                    <button
                      type="button"
                      className="kv-btn"
                      onClick={() => {
                        setIsPlaying(false);
                        setReplayStep(0);
                      }}
                    >
                      reset
                    </button>
                    <button
                      type="button"
                      className="kv-btn"
                      onClick={() => {
                        setIsPlaying(false);
                        setReplayStep(step => Math.max(step - 1, 0));
                      }}
                    >
                      prev
                    </button>
                    <button
                      type="button"
                      className="kv-btn"
                      onClick={() => setIsPlaying(value => !value)}
                    >
                      {isPlaying ? 'pause' : 'play'}
                    </button>
                    <button
                      type="button"
                      className="kv-btn"
                      onClick={() => {
                        setIsPlaying(false);
                        setReplayStep(step => Math.min(step + 1, selectedRun.steps.length - 1));
                      }}
                    >
                      next
                    </button>
                  </div>

                  <div className="kv-slider-wrap kv-slider-wrap-wide">
                    <span className="kv-slider-label">
                      Position:{' '}
                      <span className="kv-slider-value">
                        {formatReplayValue(selectedStep)}
                      </span>
                    </span>
                    <input
                      type="range"
                      min="0"
                      max={selectedRun.steps.length - 1}
                      step="1"
                      value={replayStep}
                      onChange={event => {
                        setIsPlaying(false);
                        setReplayStep(parseInt(event.target.value, 10));
                      }}
                      className="kv-slider kv-slider-wide"
                    />
                  </div>
                </div>

                <div className="kv-live-note">{selectedStep.note}</div>
              </div>

              <div className="kv-stage-grid">
                <MatrixPane run={selectedRun} step={selectedStep} />
                <CachePane run={selectedRun} step={selectedStep} />
              </div>
            </>
          )}
        </>
      )}

      <style>{`
        .kv-root {
          max-width: var(--content-width);
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

        .kv-btn,
        .kv-mode-btn,
        .kv-chip {
          font-family: var(--font-mono);
          font-size: 11px;
          color: var(--muted);
          background: none;
          border: 0.5px solid var(--bdr);
          border-radius: 4px;
          padding: 5px 12px;
          cursor: pointer;
          transition: color 0.2s, border-color 0.2s, background 0.2s;
        }

        .kv-btn:hover,
        .kv-mode-btn:hover,
        .kv-chip:hover {
          color: var(--acc);
          border-color: var(--acc);
        }

        .kv-btn:disabled,
        .kv-chip:disabled {
          opacity: 0.4;
          cursor: not-allowed;
        }

        .kv-btn-primary,
        .kv-mode-btn.is-active,
        .kv-chip.is-active {
          color: var(--acc);
          border-color: var(--acc);
        }

        .kv-mode-btn.is-active,
        .kv-chip.is-active {
          background: rgba(56, 189, 248, 0.08);
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
        .kv-control-label,
        .kv-metric-label {
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
          display: flex;
          flex-wrap: wrap;
          align-items: center;
          gap: 18px;
        }

        .kv-control-group {
          display: grid;
          gap: 6px;
        }

        .kv-mode-group,
        .kv-chip-row,
        .kv-replay-buttons {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          flex-wrap: wrap;
        }

        .kv-slider-wrap {
          display: inline-flex;
          align-items: center;
          gap: 12px;
          font-size: 12px;
          color: var(--muted);
          flex-wrap: wrap;
        }

        .kv-slider-wrap-wide {
          flex: 1 1 20rem;
          justify-content: space-between;
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

        .kv-slider-wide {
          width: min(100%, 420px);
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

        .kv-status {
          font-size: 12px;
          color: var(--muted);
          margin-bottom: 16px;
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

        .kv-replay-toolbar {
          display: flex;
          flex-wrap: wrap;
          align-items: center;
          gap: 18px;
          margin-bottom: 12px;
        }

        .kv-live-note {
          font-size: 12px;
          color: var(--muted);
          line-height: 1.7;
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

        .kv-pane-meta,
        .kv-cache-label,
        .kv-cache-kind {
          color: var(--muted);
          font-size: 11px;
        }

        .kv-matrix {
          display: grid;
          gap: 2px;
          aspect-ratio: 1;
        }

        .kv-matrix-cell {
          border-radius: 2px;
          background: rgba(26, 37, 53, 0.6);
        }

        .kv-matrix-cell.is-empty {
          background: transparent;
          border: 0.5px solid rgba(26, 37, 53, 0.35);
        }

        .kv-matrix-cell.is-active {
          background: rgba(56, 189, 248, 0.28);
        }

        .kv-matrix-cell.is-history {
          background: rgba(56, 189, 248, 0.1);
        }

        .kv-matrix-cell.is-focus {
          background: rgba(56, 189, 248, 0.78);
        }

        .kv-cache-empty {
          min-height: 12rem;
          display: flex;
          align-items: center;
          color: var(--muted);
          line-height: 1.8;
        }

        .kv-cache-stack {
          display: grid;
          gap: 8px;
        }

        .kv-cache-layer {
          display: grid;
          grid-template-columns: 2rem minmax(0, 1fr);
          gap: 10px;
          align-items: start;
        }

        .kv-cache-lane-wrap {
          display: grid;
          gap: 6px;
        }

        .kv-cache-lane-row {
          display: grid;
          grid-template-columns: 1rem minmax(0, 1fr);
          gap: 8px;
          align-items: center;
        }

        .kv-cache-lane {
          display: flex;
          gap: 2px;
          flex-wrap: nowrap;
          overflow: hidden;
        }

        .kv-cache-cell {
          flex: 0 0 8px;
          height: 8px;
          border-radius: 2px;
          border: 0.5px solid rgba(26, 37, 53, 0.9);
          background: transparent;
        }

        .cache-int8 .kv-cache-cell {
          flex-basis: 4px;
        }

        .kv-cache-cell.is-filled {
          background: rgba(56, 189, 248, 0.45);
        }

        @keyframes kv-pulse {
          0%, 100% { opacity: 0.5; }
          50% { opacity: 1; }
        }

        @media (max-width: 768px) {
          .kv-root {
            padding: 24px 16px;
          }

          .kv-controls,
          .kv-replay-toolbar {
            flex-direction: column;
            align-items: stretch;
            gap: 12px;
          }

          .kv-slider,
          .kv-slider-wide {
            width: 100%;
          }

          .kv-slider-wrap-wide {
            display: grid;
            gap: 8px;
          }

          .kv-stage-grid {
            grid-template-columns: 1fr;
          }
        }
      `}</style>
    </div>
  );
}
