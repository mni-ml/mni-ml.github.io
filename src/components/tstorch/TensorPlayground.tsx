import { useState, useCallback } from 'react';

interface TensorData {
  rows: number;
  cols: number;
  values: number[];
}

function randomTensor(rows: number, cols: number): TensorData {
  return {
    rows,
    cols,
    values: Array.from({ length: rows * cols }, () =>
      Math.round(Math.random() * 10)
    ),
  };
}

function multiplyTensors(a: TensorData, b: TensorData): TensorData {
  const result: number[] = [];
  for (let i = 0; i < a.rows; i++) {
    for (let j = 0; j < b.cols; j++) {
      let sum = 0;
      for (let k = 0; k < a.cols; k++) {
        sum += a.values[i * a.cols + k] * b.values[k * b.cols + j];
      }
      result.push(sum);
    }
  }
  return { rows: a.rows, cols: b.cols, values: result };
}

function TensorGrid({ tensor, label }: { tensor: TensorData; label: string }) {
  return (
    <div className="tensor-grid">
      <h4>{label}</h4>
      <table>
        <tbody>
          {Array.from({ length: tensor.rows }, (_, i) => (
            <tr key={i}>
              {Array.from({ length: tensor.cols }, (_, j) => (
                <td key={j}>{tensor.values[i * tensor.cols + j]}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function TensorPlayground() {
  const [size, setSize] = useState(3);
  const [tensorA, setTensorA] = useState(() => randomTensor(3, 3));
  const [tensorB, setTensorB] = useState(() => randomTensor(3, 3));

  const result = multiplyTensors(tensorA, tensorB);

  const regenerate = useCallback(() => {
    setTensorA(randomTensor(size, size));
    setTensorB(randomTensor(size, size));
  }, [size]);

  return (
    <div style={{ padding: '1rem', border: '1px solid var(--color-border, #e5e7eb)', borderRadius: '0.5rem' }}>
      <div style={{ marginBottom: '1rem', display: 'flex', gap: '1rem', alignItems: 'center' }}>
        <label>
          Size:{' '}
          <select
            value={size}
            onChange={(e) => {
              const s = Number(e.target.value);
              setSize(s);
              setTensorA(randomTensor(s, s));
              setTensorB(randomTensor(s, s));
            }}
          >
            <option value={2}>2x2</option>
            <option value={3}>3x3</option>
            <option value={4}>4x4</option>
          </select>
        </label>
        <button onClick={regenerate}>Randomize</button>
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', alignItems: 'center' }}>
        <TensorGrid tensor={tensorA} label="A" />
        <span style={{ fontSize: '1.5rem' }}>&times;</span>
        <TensorGrid tensor={tensorB} label="B" />
        <span style={{ fontSize: '1.5rem' }}>=</span>
        <TensorGrid tensor={result} label="A &times; B" />
      </div>

      <style>{`
        .tensor-grid h4 { margin: 0 0 0.5rem; font-size: 0.9rem; }
        .tensor-grid table { border-collapse: collapse; }
        .tensor-grid td {
          border: 1px solid var(--color-border, #e5e7eb);
          padding: 0.25rem 0.5rem;
          text-align: center;
          font-family: monospace;
          font-size: 0.85rem;
          min-width: 2.5rem;
        }
        button {
          padding: 0.375rem 0.75rem;
          border: 1px solid var(--color-border, #e5e7eb);
          border-radius: 0.25rem;
          background: var(--color-surface, #f9fafb);
          cursor: pointer;
          font-size: 0.85rem;
        }
        button:hover { background: var(--color-border, #e5e7eb); }
        select {
          padding: 0.25rem 0.5rem;
          border: 1px solid var(--color-border, #e5e7eb);
          border-radius: 0.25rem;
          font-size: 0.85rem;
        }
      `}</style>
    </div>
  );
}
