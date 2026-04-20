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
const TEMPERATURE = 0;
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

function speedupLabel(base: number, target: number): string {
  if (target <= 0) return '—';
  return `${(base / target).toFixed(2)}x`;
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

function TokenPane({
  tokens,
  focusIndex,
  label,
}: {
  tokens: StepTrace['tokens'];
  focusIndex: number;
  label: string;
}) {
  return (
    <section className="kv-pane">
      <div className="kv-pane-header">
        <span className="kv-pane-title">token stream</span>
        <span className="kv-pane-meta">{label}</span>
      </div>
      <div className="kv-token-output">
        {tokens.map((token, index) => (
          <span
            key={`${token.id}-${index}`}
            className={[
              'kv-token',
              token.isPrompt ? 'is-prompt' : 'is-generated',
              index === focusIndex ? 'is-focus' : '',
            ].join(' ').trim()}
            title={`token ${token.id}`}
          >
            {token.text}
          </span>
        ))}
      </div>
    </section>
  );
}

function MetricsRow({ run, step }: { run: RunTrace; step: StepTrace }) {
  const cards = [
    { label: 'prefill time', value: formatMs(run.prefillMs) },
    { label: 'decode ms/token', value: formatMs(run.decodeMsPerToken) },
    { label: 'tokens / sec', value: formatRate(run.tokensPerSec) },
    { label: 'cache used / cap', value: run.cacheMode === 'none'
      ? '0 b / 0 b'
      : `${formatBytes(step.cacheBytesUsed)} / ${formatBytes(run.cacheBytesCapacity)}` },
  ];

  return (
    <div className="kv-metrics-grid">
      {cards.map(card => (
        <div className="kv-metric-card" key={card.label}>
          <div className="kv-metric-label">{card.label}</div>
          <div className="kv-metric-value">{card.value}</div>
        </div>
      ))}
    </div>
  );
}

function WorkChart({ run, currentStep }: { run: RunTrace; currentStep: number }) {
  const maxWork = Math.max(...run.steps.map(step => step.workUnits), 1);

  return (
    <section className="kv-pane">
      <div className="kv-pane-header">
        <span className="kv-pane-title">work this step</span>
        <span className="kv-pane-meta">{modeLabel(run.mode)}</span>
      </div>
      <div className="kv-work-chart">
        {run.steps.map((step, index) => (
          <button
            type="button"
            key={`${run.mode}-${index}`}
            className={`kv-work-bar${index === currentStep ? ' is-current' : ''}`}
            style={{ height: `${Math.max((step.workUnits / maxWork) * 100, 6)}%` }}
            title={`${step.phase} step ${step.stepIndex}: ${step.workUnits.toFixed(0)} work units`}
            tabIndex={-1}
          />
        ))}
      </div>
    </section>
  );
}

function ComparePanel({ title, run, step }: { title: string; run: RunTrace; step: StepTrace }) {
  return (
    <div className="kv-compare-panel">
      <div className="kv-compare-title">{title}</div>
      <div className="kv-live-note">{step.note}</div>
      <MetricsRow run={run} step={step} />
      <div className="kv-compare-pane-grid">
        <MatrixPane run={run} step={step} />
        <CachePane run={run} step={step} />
      </div>
      <WorkChart run={run} currentStep={step.stepIndex} />
    </div>
  );
}

export default function KvCacheDemo() {
  const [phase, setPhase] = useState<Phase>('idle');
  const [progress, setProgress] = useState({ loaded: 0, total: 0, label: '' });
  const [error, setError] = useState('');
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [maxNewTokens, setMaxNewTokens] = useState(DEFAULT_TOKENS);
  const [selectedMode, setSelectedMode] = useState<BenchmarkMode>('kv-fp32');
  const [compare, setCompare] = useState(false);
  const [suite, setSuite] = useState<BenchmarkSuite | null>(null);
  const [replayStep, setReplayStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  const modelRef = useRef<ReturnType<typeof loadReplayModel> | null>(null);
  const tokenizerRef = useRef<BPETokenizer | null>(null);

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
      temperature: TEMPERATURE,
      onProgress: setProgress,
    });

    setSuite(nextSuite);
    setReplayStep(0);
    setPhase('ready');
  }, [maxNewTokens, prompt]);

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
  const baselineRun = suite?.runs.baseline ?? null;
  const baselineStep = baselineRun?.steps[replayStep] ?? null;
  const sharedTokens = suite?.outputsMatch
    ? baselineStep?.tokens ?? selectedStep?.tokens ?? []
    : selectedStep?.tokens ?? [];
  const sharedFocus = suite?.outputsMatch
    ? baselineStep?.focusIndex ?? selectedStep?.focusIndex ?? 0
    : selectedStep?.focusIndex ?? 0;

  const takeaways = suite ? [
    `kv fp32 end-to-end speedup vs baseline: ${speedupLabel(suite.runs.baseline.totalMs, suite.runs['kv-fp32'].totalMs)}`,
    `kv int8 cache reduction vs kv fp32: ${speedupLabel(suite.runs['kv-fp32'].cacheBytesUsed, suite.runs['kv-int8'].cacheBytesUsed)}`,
    suite.outputsMatch
      ? 'same greedy output across baseline, kv fp32, and kv int8 for this run'
      : 'greedy outputs diverged across modes for this run',
  ] : [];

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
            load model and run benchmark
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
        </div>
      )}

      {phase === 'error' && (
        <div className="kv-card kv-card-error">
          <div className="kv-error-text">{error}</div>
          <button type="button" onClick={() => { setError(''); setPhase('idle'); }} className="kv-btn">
            try again
          </button>
        </div>
      )}

      {(suite || phase === 'running') && (
        <>
          <div className="kv-toolbar">
            <div className="kv-mode-group">
              {MODE_ORDER.map(mode => (
                <button
                  type="button"
                  key={mode}
                  className={`kv-mode-btn${selectedMode === mode ? ' is-active' : ''}`}
                  onClick={() => {
                    setSelectedMode(mode);
                    if (mode === 'baseline') setCompare(false);
                    setReplayStep(0);
                    setIsPlaying(false);
                  }}
                >
                  {modeLabel(mode)}
                </button>
              ))}
            </div>
            <button
              type="button"
              className={`kv-toggle-btn${compare ? ' is-active' : ''}`}
              onClick={() => setCompare(value => !value)}
              disabled={selectedMode === 'baseline'}
            >
              compare
            </button>
            <button type="button" className="kv-btn" onClick={handleLoadAndRun}>
              rerun benchmark
            </button>
          </div>

          <div className="kv-controls">
            <label className="kv-input-group">
              <span className="kv-label">prompt</span>
              <input
                value={prompt}
                onChange={event => setPrompt(event.target.value)}
                className="kv-input"
                spellCheck={false}
              />
            </label>
            <div className="kv-token-group">
              <span className="kv-label">generated tokens</span>
              <div className="kv-chip-row">
                {TOKEN_OPTIONS.map(value => (
                  <button
                    type="button"
                    key={value}
                    className={`kv-chip${maxNewTokens === value ? ' is-active' : ''}`}
                    onClick={() => setMaxNewTokens(value)}
                  >
                    {value}
                  </button>
                ))}
              </div>
            </div>
            <div className="kv-temp-lock">
              <span className="kv-label">temperature</span>
              <span className="kv-lock-pill">0 (greedy)</span>
            </div>
          </div>

          {suite && (
            <div className="kv-summary-strip">
              {MODE_ORDER.map(mode => {
                const run = suite.runs[mode];
                return (
                  <button
                    type="button"
                    key={mode}
                    className={`kv-summary-card${selectedMode === mode ? ' is-selected' : ''}`}
                    onClick={() => {
                      setSelectedMode(mode);
                      if (mode === 'baseline') setCompare(false);
                    }}
                  >
                    <div className="kv-summary-mode">{modeLabel(mode)}</div>
                    <div className="kv-summary-grid">
                      <span>{formatMs(run.totalMs)}</span>
                      <span>{formatMs(run.decodeMsPerToken)}</span>
                      <span>{run.cacheMode === 'none' ? '0 b' : formatBytes(run.cacheBytesUsed)}</span>
                    </div>
                    <div className="kv-summary-labels">
                      <span>total</span>
                      <span>decode</span>
                      <span>cache</span>
                    </div>
                  </button>
                );
              })}
            </div>
          )}

          {phase === 'running' && suite && (
            <div className="kv-running-banner">{progress.label}</div>
          )}

          {suite && selectedRun && selectedStep && (
            <>
              <div className="kv-live-note">
                {compare && baselineStep
                  ? `baseline: ${baselineStep.note}  |  ${modeLabel(selectedMode)}: ${selectedStep.note}`
                  : selectedStep.note}
              </div>

              <div className="kv-replay-toolbar">
                <div className="kv-replay-buttons">
                  <button type="button" className="kv-btn" onClick={() => setReplayStep(0)}>reset</button>
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
                <div className="kv-step-rail">
                  {selectedRun.steps.map((step, index) => (
                    <button
                      type="button"
                      key={`${step.mode}-${step.stepIndex}`}
                      className={`kv-step-dot${replayStep === index ? ' is-active' : ''}`}
                      onClick={() => {
                        setIsPlaying(false);
                        setReplayStep(index);
                      }}
                    >
                      {index === 0 ? 'prefill' : index}
                    </button>
                  ))}
                </div>
              </div>

              <TokenPane
                tokens={sharedTokens}
                focusIndex={sharedFocus}
                label={suite.outputsMatch ? 'same output path' : modeLabel(selectedMode)}
              />

              {!compare && (
                <>
                  <div className="kv-stage-grid">
                    <MatrixPane run={selectedRun} step={selectedStep} />
                    <CachePane run={selectedRun} step={selectedStep} />
                  </div>
                  <MetricsRow run={selectedRun} step={selectedStep} />
                  <WorkChart run={selectedRun} currentStep={replayStep} />
                </>
              )}

              {compare && baselineRun && baselineStep && selectedMode !== 'baseline' && (
                <div className="kv-compare-grid">
                  <ComparePanel title="baseline" run={baselineRun} step={baselineStep} />
                  <ComparePanel title={modeLabel(selectedMode)} run={selectedRun} step={selectedStep} />
                </div>
              )}

              <div className="kv-takeaways">
                {takeaways.map(text => (
                  <div key={text} className="kv-takeaway">
                    {text}
                  </div>
                ))}
              </div>
            </>
          )}
        </>
      )}

      <style>{`
        .kv-root {
          max-width: 72rem;
          margin: 0 auto;
          padding: 36px 32px 60px;
        }

        .kv-header {
          margin-bottom: 28px;
          padding-bottom: 18px;
          border-bottom: 0.5px solid var(--bdr);
        }

        .kv-title {
          font-size: 22px;
          font-weight: 500;
          letter-spacing: -0.03em;
          margin: 0 0 6px;
        }

        .kv-desc {
          margin: 0;
          color: var(--muted);
          line-height: 1.75;
          max-width: 52rem;
        }

        .kv-card,
        .kv-pane,
        .kv-summary-card,
        .kv-metric-card,
        .kv-compare-panel,
        .kv-takeaway {
          border: 0.5px solid var(--bdr);
          border-radius: 6px;
          background: rgba(19, 25, 35, 0.48);
        }

        .kv-card {
          padding: 28px 24px;
          text-align: center;
        }

        .kv-card-error {
          border-color: #5c2020;
          background: rgba(92, 32, 32, 0.18);
        }

        .kv-muted,
        .kv-label,
        .kv-pane-meta,
        .kv-summary-labels,
        .kv-metric-label {
          color: var(--muted);
          font-size: 11px;
          text-transform: lowercase;
        }

        .kv-error-text {
          color: #fca5a5;
          margin-bottom: 12px;
        }

        .kv-btn,
        .kv-mode-btn,
        .kv-toggle-btn,
        .kv-chip,
        .kv-step-dot {
          font: inherit;
          color: var(--muted);
          background: transparent;
          border: 0.5px solid var(--bdr);
          border-radius: 4px;
          padding: 6px 10px;
          cursor: pointer;
          transition: color 0.18s ease, border-color 0.18s ease, background 0.18s ease;
        }

        .kv-btn:hover,
        .kv-mode-btn:hover,
        .kv-toggle-btn:hover,
        .kv-chip:hover,
        .kv-step-dot:hover {
          color: var(--txt);
          border-color: var(--acc);
        }

        .kv-mode-btn.is-active,
        .kv-toggle-btn.is-active,
        .kv-chip.is-active,
        .kv-step-dot.is-active {
          color: var(--txt);
          border-color: var(--acc);
          background: rgba(56, 189, 248, 0.08);
        }

        .kv-toggle-btn:disabled {
          opacity: 0.45;
          cursor: not-allowed;
        }

        .kv-progress-track {
          width: min(100%, 24rem);
          height: 3px;
          background: var(--bdr);
          border-radius: 2px;
          margin: 0 auto;
          overflow: hidden;
        }

        .kv-progress-bar {
          height: 100%;
          background: var(--acc);
          transition: width 0.25s ease;
        }

        .kv-progress-bar.is-pulse {
          animation: kv-pulse 1.3s ease-in-out infinite;
        }

        .kv-toolbar,
        .kv-controls,
        .kv-replay-toolbar {
          display: flex;
          gap: 12px;
          flex-wrap: wrap;
          align-items: center;
          margin-bottom: 16px;
        }

        .kv-mode-group,
        .kv-chip-row,
        .kv-replay-buttons {
          display: inline-flex;
          gap: 8px;
          flex-wrap: wrap;
        }

        .kv-controls {
          align-items: end;
        }

        .kv-input-group {
          display: grid;
          gap: 6px;
          min-width: 18rem;
          flex: 1 1 24rem;
        }

        .kv-input {
          font: inherit;
          color: var(--txt);
          background: transparent;
          border: 0.5px solid var(--bdr);
          border-radius: 4px;
          padding: 8px 10px;
        }

        .kv-temp-lock,
        .kv-token-group {
          display: grid;
          gap: 6px;
        }

        .kv-lock-pill,
        .kv-running-banner,
        .kv-live-note {
          border: 0.5px solid var(--bdr);
          border-radius: 4px;
          padding: 8px 10px;
          color: var(--txt);
          background: rgba(19, 25, 35, 0.4);
        }

        .kv-running-banner,
        .kv-live-note {
          margin-bottom: 16px;
        }

        .kv-summary-strip {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 12px;
          margin-bottom: 16px;
        }

        .kv-summary-card {
          text-align: left;
          padding: 14px;
        }

        .kv-summary-card.is-selected {
          border-color: var(--acc);
        }

        .kv-summary-mode {
          margin-bottom: 10px;
          color: var(--txt);
        }

        .kv-summary-grid,
        .kv-summary-labels {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 8px;
        }

        .kv-summary-grid {
          font-size: 12px;
          margin-bottom: 6px;
        }

        .kv-step-rail {
          display: flex;
          gap: 8px;
          overflow-x: auto;
          padding-bottom: 4px;
        }

        .kv-stage-grid {
          display: grid;
          grid-template-columns: minmax(0, 1.15fr) minmax(0, 0.85fr);
          gap: 16px;
          margin-bottom: 16px;
        }

        .kv-compare-grid,
        .kv-compare-pane-grid {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 16px;
        }

        .kv-compare-grid {
          margin-bottom: 16px;
        }

        .kv-compare-panel {
          padding: 14px;
        }

        .kv-compare-title,
        .kv-pane-title,
        .kv-metric-value {
          color: var(--txt);
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

        .kv-token-output {
          min-height: 8rem;
          line-height: 1.8;
          white-space: pre-wrap;
          word-break: break-word;
        }

        .kv-token {
          border-radius: 3px;
          padding: 1px 2px;
        }

        .kv-token.is-prompt {
          color: var(--muted);
        }

        .kv-token.is-generated {
          color: var(--txt);
        }

        .kv-token.is-focus {
          background: rgba(56, 189, 248, 0.1);
          outline: 0.5px solid rgba(56, 189, 248, 0.45);
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

        .kv-cache-label,
        .kv-cache-kind {
          color: var(--muted);
          font-size: 11px;
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

        .kv-metrics-grid {
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          gap: 12px;
          margin-bottom: 16px;
        }

        .kv-metric-card {
          padding: 12px;
        }

        .kv-metric-value {
          margin-top: 6px;
          font-size: 13px;
        }

        .kv-work-chart {
          min-height: 9rem;
          display: flex;
          align-items: end;
          gap: 6px;
        }

        .kv-work-bar {
          flex: 1 1 0;
          border: 0;
          border-radius: 3px 3px 0 0;
          background: rgba(56, 189, 248, 0.24);
          padding: 0;
          min-width: 10px;
          cursor: default;
        }

        .kv-work-bar.is-current {
          background: rgba(56, 189, 248, 0.82);
        }

        .kv-takeaways {
          display: grid;
          gap: 10px;
        }

        .kv-takeaway {
          padding: 12px;
          line-height: 1.7;
        }

        @keyframes kv-pulse {
          0%, 100% { opacity: 0.45; }
          50% { opacity: 1; }
        }

        @media (max-width: 960px) {
          .kv-root {
            padding: 28px 18px 48px;
          }

          .kv-summary-strip,
          .kv-stage-grid,
          .kv-compare-grid,
          .kv-compare-pane-grid,
          .kv-metrics-grid {
            grid-template-columns: 1fr;
          }
        }
      `}</style>
    </div>
  );
}
