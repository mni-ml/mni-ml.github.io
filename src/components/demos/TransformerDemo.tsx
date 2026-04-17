import { useState, useRef, useCallback } from 'react';
import { BPETokenizer, type TokenizerJSON } from '../../lib/inference/bpe';
import {
  MiniGPT, loadCheckpoint, getTopKTokens,
  type CheckpointData, type TopKToken,
} from '../../lib/inference/model';

const MODEL_URL =
  'https://media.githubusercontent.com/media/mni-ml/transformer/main/out/model-final.json';
const TOKENIZER_URL =
  'https://raw.githubusercontent.com/mni-ml/transformer/main/out/tokenizer.json';

const INITIAL_PROMPT = 'There once was';
const TOP_K = 9;
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
    if (outputRef.current) outputRef.current.scrollTop = outputRef.current.scrollHeight;
  }, []);

  const computeTopK = useCallback(async () => {
    const model = modelRef.current;
    const tokenizer = tokenizerRef.current;
    if (!model || !tokenizer) return;
    setIsComputing(true);
    await new Promise(r => setTimeout(r, 20));
    try {
      const { topTokens } = getTopKTokens(model, contextRef.current, TEMPERATURE, TOP_K);
      for (const t of topTokens) t.text = tokenizer.decodeToken(t.id);
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
    setTokens(encoded.map(id => ({ id, text: tokenizer.decodeToken(id) })));
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
      setProgress({ loaded: 0, total: 0, label: 'Parsing model JSON...' });
      await new Promise(r => setTimeout(r, 50));

      const decoder = new TextDecoder();
      let jsonStr = '';
      for (const chunk of chunks) jsonStr += decoder.decode(chunk, { stream: true });
      jsonStr += decoder.decode();
      const checkpoint: CheckpointData = JSON.parse(jsonStr);
      jsonStr = '';

      setProgress({ loaded: 0, total: 0, label: 'Loading model weights...' });
      await new Promise(r => setTimeout(r, 20));
      const model = loadCheckpoint(checkpoint);
      modelRef.current = model;

      const encoded = tokenizer.encode(INITIAL_PROMPT);
      contextRef.current = encoded;
      setTokens(encoded.map(id => ({ id, text: tokenizer.decodeToken(id) })));

      setProgress({ loaded: 0, total: 0, label: 'Running first forward pass...' });
      await new Promise(r => setTimeout(r, 20));
      await computeTopK();
    } catch (e: any) {
      setError(e.message || 'Unknown error');
      setPhase('error');
    }
  }, [computeTopK]);

  const formatProb = (p: number) =>
    p >= 0.01 ? (p * 100).toFixed(1) + '%' : (p * 100).toFixed(2) + '%';

  const displayToken = (text: string) =>
    text.replace(/ /g, '\u00B7').replace(/\n/g, '\\n').replace(/\t/g, '\\t');

  return (
    <div className="demo-root">
      <div className="demo-header">
        <h1 className="demo-title">Transformer Token Explorer</h1>
        <p className="demo-desc">
          A 12M-parameter LLM trained and running on the web using{' '}
          <a href="https://github.com/mni-ml/framework">@mni-ml/framework</a>
          . See the next predicted tokens and their probability distributions.
        </p>
      </div>

      {phase === 'idle' && (
        <div className="demo-card demo-idle">
          <p className="demo-muted">
            This demo downloads the model (~250 MB) and runs inference entirely in your browser.
          </p>
          <button onClick={handleStart} className="demo-btn">
            Load Model &amp; Start
          </button>
        </div>
      )}

      {(phase === 'downloading' || phase === 'parsing') && (
        <div className="demo-card demo-loading">
          <div className="demo-muted">{progress.label}</div>
          <div className="demo-progress-track">
            <div
              className={`demo-progress-bar${phase === 'parsing' || !progress.total ? ' demo-progress-pulse' : ''}`}
              style={{
                width: progress.total > 0
                  ? `${(progress.loaded / progress.total) * 100}%`
                  : phase === 'parsing' ? '100%' : '0%',
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
          <button onClick={() => { setPhase('idle'); setError(''); }} className="demo-btn">
            Try Again
          </button>
        </div>
      )}

      {(phase === 'ready' || phase === 'computing') && (
        <>
          <div className="demo-section">
            <div className="demo-section-header">
              <span className="demo-label">Generated Text</span>
              <button onClick={handleReset} className="demo-btn">Reset</button>
            </div>
            <div ref={outputRef} className="demo-output">
              {tokens.map((t, i) => (
                <span
                  key={i}
                  className={
                    i < tokenizerRef.current!.encode(INITIAL_PROMPT).length
                      ? 'demo-tok-prompt'
                      : 'demo-tok-gen'
                  }
                  title={`token ${t.id}`}
                >
                  {t.text}
                </span>
              ))}
              {isComputing && <span className="demo-cursor">▊</span>}
            </div>
          </div>

          <div className="demo-section">
            <div className="demo-label" style={{ marginBottom: 10 }}>
              {isComputing ? 'Computing next token probabilities...' : 'Pick the next token:'}
            </div>
            {isComputing ? (
              <div className="demo-spinner-row"><div className="demo-spinner" /></div>
            ) : (
              <div className="demo-grid">
                {topKTokens.map((t, i) => (
                  <button key={t.id} onClick={() => handleTokenClick(t)} className="demo-token-card">
                    <span className="demo-token-text">{displayToken(t.text)}</span>
                    <div className="demo-prob-track">
                      <div
                        className="demo-prob-bar"
                        style={{
                          width: `${Math.max(t.prob * 100, 1)}%`,
                          opacity: 0.4 + t.prob * 0.6,
                        }}
                      />
                    </div>
                    <span className="demo-prob-label">{formatProb(t.prob)}</span>
                    <span className="demo-rank">#{i + 1}</span>
                  </button>
                ))}
              </div>
            )}
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

        .demo-idle {}
        .demo-loading {}

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

        .demo-section-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 8px;
        }

        .demo-label {
          font-size: 10px;
          color: var(--muted);
          text-transform: uppercase;
          letter-spacing: 0.1em;
        }

        .demo-output {
          border: 0.5px solid var(--bdr);
          border-radius: 6px;
          padding: 14px 16px;
          min-height: 4rem;
          max-height: 10rem;
          overflow-y: auto;
          font-size: 12.5px;
          line-height: 1.75;
          white-space: pre-wrap;
          word-break: break-word;
        }

        .demo-tok-prompt { color: var(--muted); }
        .demo-tok-gen { color: var(--txt); }

        .demo-cursor {
          color: var(--acc);
          animation: demo-pulse 1s ease-in-out infinite;
        }

        .demo-spinner-row {
          display: flex;
          justify-content: center;
          padding: 24px 0;
        }

        .demo-spinner {
          width: 18px;
          height: 18px;
          border: 1.5px solid var(--bdr);
          border-top-color: var(--acc);
          border-radius: 50%;
          animation: demo-spin 0.8s linear infinite;
        }

        .demo-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
          gap: 6px;
        }

        .demo-token-card {
          display: flex;
          flex-direction: column;
          align-items: stretch;
          background: none;
          border: 0.5px solid var(--bdr);
          border-radius: 6px;
          padding: 8px 10px;
          cursor: pointer;
          transition: border-color 0.2s;
          text-align: left;
          font-family: var(--font-mono);
          color: var(--txt);
          position: relative;
        }

        .demo-token-card:hover {
          border-color: var(--acc);
        }

        .demo-token-text {
          font-size: 12.5px;
          font-weight: 500;
          margin-bottom: 4px;
          white-space: pre;
          overflow: hidden;
          text-overflow: ellipsis;
        }

        .demo-prob-track {
          width: 100%;
          height: 2px;
          background: var(--bdr);
          border-radius: 1px;
          overflow: hidden;
          margin-bottom: 3px;
        }

        .demo-prob-bar {
          height: 100%;
          background: var(--acc);
          border-radius: 1px;
        }

        .demo-prob-label {
          font-size: 10px;
          color: var(--muted);
        }

        .demo-rank {
          position: absolute;
          top: 4px;
          right: 6px;
          font-size: 9px;
          color: var(--muted);
          opacity: 0.4;
        }

        @keyframes demo-pulse {
          0%, 100% { opacity: 0.5; }
          50% { opacity: 1; }
        }

        @keyframes demo-spin {
          to { transform: rotate(360deg); }
        }

        @media (max-width: 768px) {
          .demo-root {
            padding: 24px 16px;
          }
          .demo-grid {
            grid-template-columns: repeat(auto-fill, minmax(110px, 1fr));
          }
        }
      `}</style>
    </div>
  );
}
