import { useState } from 'react';

interface Node {
  id: string;
  label: string;
  x: number;
  y: number;
  value: number;
  grad: number;
}

interface Edge {
  from: string;
  to: string;
}

const INITIAL_NODES: Node[] = [
  { id: 'x', label: 'x', x: 50, y: 150, value: 2, grad: 0 },
  { id: 'w', label: 'w', x: 50, y: 50, value: 3, grad: 0 },
  { id: 'mul', label: '×', x: 200, y: 100, value: 6, grad: 0 },
  { id: 'b', label: 'b', x: 200, y: 200, value: 1, grad: 0 },
  { id: 'add', label: '+', x: 350, y: 150, value: 7, grad: 0 },
];

const EDGES: Edge[] = [
  { from: 'x', to: 'mul' },
  { from: 'w', to: 'mul' },
  { from: 'mul', to: 'add' },
  { from: 'b', to: 'add' },
];

function computeGradients(nodes: Node[]): Node[] {
  const map = new Map(nodes.map((n) => [n.id, { ...n }]));

  const x = map.get('x')!;
  const w = map.get('w')!;
  const mul = map.get('mul')!;
  const b = map.get('b')!;
  const add = map.get('add')!;

  // Forward
  mul.value = x.value * w.value;
  add.value = mul.value + b.value;

  // Backward from output
  add.grad = 1;
  mul.grad = 1; // d(add)/d(mul) = 1
  b.grad = 1; // d(add)/d(b) = 1
  w.grad = x.value; // d(mul)/d(w) = x
  x.grad = w.value; // d(mul)/d(x) = w

  return [x, w, mul, b, add];
}

export default function GradientViz() {
  const [nodes, setNodes] = useState(() => computeGradients(INITIAL_NODES));
  const [showGrad, setShowGrad] = useState(false);

  const updateInput = (id: string, value: number) => {
    setNodes((prev) => {
      const updated = prev.map((n) => (n.id === id ? { ...n, value } : n));
      return computeGradients(updated);
    });
  };

  return (
    <div style={{ padding: '1rem', border: '1px solid var(--color-border, #e5e7eb)', borderRadius: '0.5rem' }}>
      <div style={{ marginBottom: '1rem', display: 'flex', gap: '1rem', alignItems: 'center', flexWrap: 'wrap' }}>
        <label>
          x ={' '}
          <input
            type="number"
            value={nodes.find((n) => n.id === 'x')!.value}
            onChange={(e) => updateInput('x', Number(e.target.value))}
            style={{ width: '4rem', padding: '0.25rem', textAlign: 'center' }}
          />
        </label>
        <label>
          w ={' '}
          <input
            type="number"
            value={nodes.find((n) => n.id === 'w')!.value}
            onChange={(e) => updateInput('w', Number(e.target.value))}
            style={{ width: '4rem', padding: '0.25rem', textAlign: 'center' }}
          />
        </label>
        <label>
          b ={' '}
          <input
            type="number"
            value={nodes.find((n) => n.id === 'b')!.value}
            onChange={(e) => updateInput('b', Number(e.target.value))}
            style={{ width: '4rem', padding: '0.25rem', textAlign: 'center' }}
          />
        </label>
        <button
          onClick={() => setShowGrad(!showGrad)}
          style={{
            padding: '0.375rem 0.75rem',
            border: '1px solid var(--color-border, #e5e7eb)',
            borderRadius: '0.25rem',
            background: showGrad ? 'var(--color-accent, #2563eb)' : 'var(--color-surface, #f9fafb)',
            color: showGrad ? '#fff' : 'inherit',
            cursor: 'pointer',
            fontSize: '0.85rem',
          }}
        >
          {showGrad ? 'Hide' : 'Show'} Gradients
        </button>
      </div>

      <svg viewBox="0 0 430 260" style={{ width: '100%', maxWidth: '430px', height: 'auto' }}>
        {EDGES.map((e) => {
          const from = nodes.find((n) => n.id === e.from)!;
          const to = nodes.find((n) => n.id === e.to)!;
          return (
            <line
              key={`${e.from}-${e.to}`}
              x1={from.x + 30}
              y1={from.y + 20}
              x2={to.x}
              y2={to.y + 20}
              stroke="var(--color-text-muted, #6b7280)"
              strokeWidth={2}
              markerEnd="url(#arrow)"
            />
          );
        })}

        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--color-text-muted, #6b7280)" />
          </marker>
        </defs>

        {nodes.map((node) => (
          <g key={node.id}>
            <rect
              x={node.x}
              y={node.y}
              width={60}
              height={40}
              rx={6}
              fill="var(--color-surface, #f9fafb)"
              stroke="var(--color-border, #e5e7eb)"
              strokeWidth={1.5}
            />
            <text
              x={node.x + 30}
              y={node.y + 16}
              textAnchor="middle"
              fontSize={12}
              fontWeight="bold"
              fill="var(--color-text, #1a1a2e)"
            >
              {node.label}
            </text>
            <text
              x={node.x + 30}
              y={node.y + 32}
              textAnchor="middle"
              fontSize={11}
              fill="var(--color-text-muted, #6b7280)"
            >
              {node.value}
            </text>
            {showGrad && (
              <text
                x={node.x + 30}
                y={node.y - 6}
                textAnchor="middle"
                fontSize={10}
                fill="var(--color-accent, #2563eb)"
                fontWeight="bold"
              >
                grad={node.grad}
              </text>
            )}
          </g>
        ))}
      </svg>

      <p style={{ fontSize: '0.8rem', color: 'var(--color-text-muted, #6b7280)', margin: '0.5rem 0 0' }}>
        Computation graph: y = w&middot;x + b. Toggle gradients to see ∂y/∂(each variable).
      </p>
    </div>
  );
}
