import { useState, useRef, useCallback, useEffect } from 'react';
import { BPETokenizer, type TokenizerJSON } from '../../lib/inference/bpe';
import {
  MiniGPT, loadCheckpoint, getTopKTokens,
  type CheckpointData, type TopKToken,
} from '../../lib/inference/model';

const MODEL_URL =
  'https://media.githubusercontent.com/media/mni-ml/transformer/experimenting/out/model-final.json';
const TOKENIZER_URL =
  'https://raw.githubusercontent.com/mni-ml/transformer/experimenting/out/tokenizer.json';

const INITIAL_PROMPT = 'There once was';
const TOP_K = 10;
const TEMPERATURE = 0.8;

type Phase = 'idle' | 'downloading' | 'parsing' | 'ready' | 'computing' | 'error';

interface TokenSpan {
  id: number;
  text: string;
}

export default function TransformerDemo() {
  const [phase, setPhase] = useState<Phase>('idle');
  const [progress, setProgress] = useState({ loaded: 0, total: 0, label: '' });
  const [error, setError] = useState('');

  const [tokens, setTokens] = useState<TokenSpan[]>([]);
  const [topKTokens, setTopKTokens] = useState<TopKToken[]>([]);
  const [isComputing, setIsComputing] = useState(false);

  const modelRef = useRef<MiniGPT | null>(null);
  const tokenizerRef = useRef<BPETokenizer | null>(null);
  const contextRef = useRef<number[]>([]);

  const outputRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = useCallback(() => {
    if (outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight;
    }
  }, []);

  const computeTopK = useCallback(async () => {
    const model = modelRef.current;
    const tokenizer = tokenizerRef.current;
    if (!model || !tokenizer) return;

    setIsComputing(true);

    // Yield to let React render the spinner
    await new Promise(r => setTimeout(r, 20));

    try {
      const { topTokens } = getTopKTokens(model, contextRef.current, TEMPERATURE, TOP_K);
      for (const t of topTokens) {
        t.text = tokenizer.decodeToken(t.id);
      }
      setTopKTokens(topTokens);
      setPhase('ready');
    } catch (e: any) {
      setError(`Inference error: ${e.message}`);
      setPhase('error');
    } finally {
      setIsComputing(false);
    }
  }, []);

  const handleTokenClick = useCallback(async (token: TopKToken) => {
    const tokenizer = tokenizerRef.current!;
    contextRef.current.push(token.id);

    setTokens(prev => [...prev, { id: token.id, text: tokenizer.decodeToken(token.id) }]);
    setTopKTokens([]);
    scrollToBottom();

    await computeTopK();
    scrollToBottom();
  }, [computeTopK, scrollToBottom]);

  const handleReset = useCallback(async () => {
    const tokenizer = tokenizerRef.current;
    if (!tokenizer) return;

    const encoded = tokenizer.encode(INITIAL_PROMPT);
    contextRef.current = encoded;

    const spans: TokenSpan[] = encoded.map(id => ({
      id,
      text: tokenizer.decodeToken(id),
    }));
    setTokens(spans);
    setTopKTokens([]);
    await computeTopK();
  }, [computeTopK]);

  const handleStart = useCallback(async () => {
    try {
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
      const totalBytes = contentLength ? parseInt(contentLength, 10) : 0;
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
          label: `Downloading model... ${(received / 1e6).toFixed(1)} / ${totalBytes ? (totalBytes / 1e6).toFixed(0) : '?'} MB`,
        });
      }

      setPhase('parsing');
      setProgress({ loaded: 0, total: 0, label: 'Parsing model JSON (this may take a moment)...' });

      // Yield so the UI can update before the heavy parse
      await new Promise(r => setTimeout(r, 50));

      const decoder = new TextDecoder();
      let jsonStr = '';
      for (const chunk of chunks) jsonStr += decoder.decode(chunk, { stream: true });
      jsonStr += decoder.decode();

      const checkpoint: CheckpointData = JSON.parse(jsonStr);
      // Free the string to reduce memory pressure
      jsonStr = '';

      setProgress({ loaded: 0, total: 0, label: 'Loading model weights...' });
      await new Promise(r => setTimeout(r, 20));

      const model = loadCheckpoint(checkpoint);
      modelRef.current = model;

      const encoded = tokenizer.encode(INITIAL_PROMPT);
      contextRef.current = encoded;

      const spans: TokenSpan[] = encoded.map(id => ({
        id,
        text: tokenizer.decodeToken(id),
      }));
      setTokens(spans);

      setProgress({ loaded: 0, total: 0, label: 'Running first forward pass...' });
      await new Promise(r => setTimeout(r, 20));

      await computeTopK();
    } catch (e: any) {
      setError(e.message || 'Unknown error');
      setPhase('error');
    }
  }, [computeTopK]);

  const formatProb = (p: number) => {
    if (p >= 0.01) return (p * 100).toFixed(1) + '%';
    return (p * 100).toFixed(2) + '%';
  };

  const displayToken = (text: string) => {
    return text
      .replace(/ /g, '\u00B7')
      .replace(/\n/g, '\\n')
      .replace(/\t/g, '\\t');
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h1 style={styles.title}>Transformer Token Explorer</h1>
        <p style={styles.subtitle}>
          A 12M-parameter GPT trained on{' '}
          <a href="https://huggingface.co/datasets/roneneldan/TinyStories" style={styles.link}>
            TinyStories
          </a>
          {' '}using{' '}
          <a href="https://github.com/mni-ml/framework" style={styles.link}>
            @mni-ml/framework
          </a>
          . Click tokens to build a story one token at a time.
        </p>
      </div>

      {phase === 'idle' && (
        <div style={styles.startContainer}>
          <p style={styles.startText}>
            This demo downloads the model (~250 MB) and runs inference entirely in your browser.
          </p>
          <button onClick={handleStart} style={styles.startButton}>
            Load Model &amp; Start
          </button>
        </div>
      )}

      {(phase === 'downloading' || phase === 'parsing') && (
        <div style={styles.loadingContainer}>
          <div style={styles.loadingLabel}>{progress.label}</div>
          <div style={styles.progressBarOuter}>
            <div
              style={{
                ...styles.progressBarInner,
                width: progress.total > 0
                  ? `${(progress.loaded / progress.total) * 100}%`
                  : phase === 'parsing' ? '100%' : '0%',
                ...(phase === 'parsing' || !progress.total ? { animation: 'pulse 1.5s ease-in-out infinite' } : {}),
              }}
            />
          </div>
          {progress.total > 0 && (
            <div style={styles.progressPercent}>
              {((progress.loaded / progress.total) * 100).toFixed(0)}%
            </div>
          )}
        </div>
      )}

      {phase === 'error' && (
        <div style={styles.errorContainer}>
          <div style={styles.errorText}>{error}</div>
          <button onClick={() => { setPhase('idle'); setError(''); }} style={styles.resetButton}>
            Try Again
          </button>
        </div>
      )}

      {(phase === 'ready' || phase === 'computing') && (
        <>
          <div style={styles.outputSection}>
            <div style={styles.outputHeader}>
              <span style={styles.outputLabel}>Generated Text</span>
              <button onClick={handleReset} style={styles.resetButton}>Reset</button>
            </div>
            <div ref={outputRef} style={styles.outputBox}>
              {tokens.map((t, i) => (
                <span
                  key={i}
                  style={{
                    ...styles.tokenSpan,
                    ...(i < tokenizerRef.current!.encode(INITIAL_PROMPT).length
                      ? styles.promptToken
                      : styles.generatedToken),
                  }}
                  title={`token ${t.id}`}
                >
                  {t.text}
                </span>
              ))}
              {isComputing && <span style={styles.cursor}>▊</span>}
            </div>
          </div>

          <div style={styles.pickerSection}>
            <div style={styles.pickerHeader}>
              {isComputing
                ? 'Computing next token probabilities...'
                : 'Pick the next token:'}
            </div>
            {isComputing ? (
              <div style={styles.spinnerRow}>
                <div style={styles.spinner} />
              </div>
            ) : (
              <div style={styles.tokenGrid}>
                {topKTokens.map((t, i) => (
                  <button
                    key={t.id}
                    onClick={() => handleTokenClick(t)}
                    style={styles.tokenCard}
                    onMouseEnter={e => {
                      (e.currentTarget as HTMLElement).style.borderColor = 'var(--acc)';
                      (e.currentTarget as HTMLElement).style.background = 'rgba(56,189,248,0.08)';
                    }}
                    onMouseLeave={e => {
                      (e.currentTarget as HTMLElement).style.borderColor = 'var(--bdr)';
                      (e.currentTarget as HTMLElement).style.background = 'var(--surf)';
                    }}
                  >
                    <span style={styles.tokenText}>{displayToken(t.text)}</span>
                    <div style={styles.probBarOuter}>
                      <div
                        style={{
                          ...styles.probBarInner,
                          width: `${Math.max(t.prob * 100, 1)}%`,
                          opacity: 0.4 + t.prob * 0.6,
                        }}
                      />
                    </div>
                    <span style={styles.probText}>{formatProb(t.prob)}</span>
                    <span style={styles.rankBadge}>#{i + 1}</span>
                  </button>
                ))}
              </div>
            )}
          </div>
        </>
      )}

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 0.6; }
          50% { opacity: 1; }
        }
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    maxWidth: '54rem',
    margin: '0 auto',
    padding: '2rem 1.5rem',
    fontFamily: 'var(--font-mono)',
    color: 'var(--txt)',
  },
  header: {
    marginBottom: '2rem',
  },
  title: {
    fontSize: '1.4rem',
    fontWeight: 600,
    margin: '0 0 0.5rem',
    color: 'var(--txt)',
  },
  subtitle: {
    fontSize: '0.85rem',
    color: 'var(--muted)',
    margin: 0,
    lineHeight: 1.6,
  },
  link: {
    color: 'var(--acc)',
    textDecoration: 'none',
  },
  startContainer: {
    textAlign: 'center' as const,
    padding: '3rem 1rem',
    border: '1px solid var(--bdr)',
    borderRadius: '8px',
    background: 'var(--surf)',
  },
  startText: {
    color: 'var(--muted)',
    fontSize: '0.85rem',
    marginBottom: '1.5rem',
  },
  startButton: {
    background: 'var(--acc)',
    color: '#0d1117',
    border: 'none',
    padding: '0.7rem 2rem',
    borderRadius: '6px',
    fontSize: '0.9rem',
    fontWeight: 600,
    cursor: 'pointer',
    fontFamily: 'inherit',
  },
  loadingContainer: {
    padding: '3rem 1rem',
    textAlign: 'center' as const,
    border: '1px solid var(--bdr)',
    borderRadius: '8px',
    background: 'var(--surf)',
  },
  loadingLabel: {
    fontSize: '0.85rem',
    color: 'var(--muted)',
    marginBottom: '1rem',
  },
  progressBarOuter: {
    width: '100%',
    maxWidth: '400px',
    height: '6px',
    background: 'var(--bdr)',
    borderRadius: '3px',
    overflow: 'hidden',
    margin: '0 auto',
  },
  progressBarInner: {
    height: '100%',
    background: 'var(--acc)',
    borderRadius: '3px',
    transition: 'width 0.3s ease',
  },
  progressPercent: {
    fontSize: '0.8rem',
    color: 'var(--muted)',
    marginTop: '0.5rem',
  },
  errorContainer: {
    padding: '2rem',
    border: '1px solid #5c2020',
    borderRadius: '8px',
    background: '#1a0d0d',
    textAlign: 'center' as const,
  },
  errorText: {
    color: '#f87171',
    fontSize: '0.85rem',
    marginBottom: '1rem',
    wordBreak: 'break-word' as const,
  },
  outputSection: {
    marginBottom: '1.5rem',
  },
  outputHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '0.5rem',
  },
  outputLabel: {
    fontSize: '0.75rem',
    color: 'var(--muted)',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.05em',
  },
  resetButton: {
    background: 'transparent',
    border: '1px solid var(--bdr)',
    color: 'var(--muted)',
    padding: '0.3rem 0.8rem',
    borderRadius: '4px',
    fontSize: '0.75rem',
    cursor: 'pointer',
    fontFamily: 'inherit',
  },
  outputBox: {
    background: 'var(--surf)',
    border: '1px solid var(--bdr)',
    borderRadius: '8px',
    padding: '1rem 1.2rem',
    minHeight: '5rem',
    maxHeight: '12rem',
    overflowY: 'auto' as const,
    fontSize: '0.9rem',
    lineHeight: 1.8,
    whiteSpace: 'pre-wrap' as const,
    wordBreak: 'break-word' as const,
  },
  tokenSpan: {
    borderRadius: '2px',
    padding: '1px 0',
  },
  promptToken: {
    color: 'var(--muted)',
  },
  generatedToken: {
    color: 'var(--txt)',
  },
  cursor: {
    color: 'var(--acc)',
    animation: 'pulse 1s ease-in-out infinite',
  },
  pickerSection: {
    border: '1px solid var(--bdr)',
    borderRadius: '8px',
    background: 'var(--surf)',
    padding: '1rem 1.2rem',
  },
  pickerHeader: {
    fontSize: '0.75rem',
    color: 'var(--muted)',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.05em',
    marginBottom: '0.8rem',
  },
  spinnerRow: {
    display: 'flex',
    justifyContent: 'center',
    padding: '2rem 0',
  },
  spinner: {
    width: '24px',
    height: '24px',
    border: '2px solid var(--bdr)',
    borderTopColor: 'var(--acc)',
    borderRadius: '50%',
    animation: 'spin 0.8s linear infinite',
  },
  tokenGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))',
    gap: '0.5rem',
  },
  tokenCard: {
    display: 'flex',
    flexDirection: 'column' as const,
    alignItems: 'stretch',
    background: 'var(--surf)',
    border: '1px solid var(--bdr)',
    borderRadius: '6px',
    padding: '0.6rem 0.7rem',
    cursor: 'pointer',
    transition: 'border-color 0.15s, background 0.15s',
    textAlign: 'left' as const,
    fontFamily: 'inherit',
    color: 'var(--txt)',
    position: 'relative' as const,
  },
  tokenText: {
    fontSize: '0.9rem',
    fontWeight: 500,
    marginBottom: '0.3rem',
    whiteSpace: 'pre' as const,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
  },
  probBarOuter: {
    width: '100%',
    height: '3px',
    background: 'var(--bdr)',
    borderRadius: '2px',
    overflow: 'hidden',
    marginBottom: '0.2rem',
  },
  probBarInner: {
    height: '100%',
    background: 'var(--acc)',
    borderRadius: '2px',
  },
  probText: {
    fontSize: '0.7rem',
    color: 'var(--muted)',
  },
  rankBadge: {
    position: 'absolute' as const,
    top: '4px',
    right: '6px',
    fontSize: '0.6rem',
    color: 'var(--muted)',
    opacity: 0.5,
  },
};
