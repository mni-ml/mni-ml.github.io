<img width="2132" height="1200" alt="Logo" src="https://github.com/user-attachments/assets/053142ce-810c-4cf1-8335-b96efc676b14" />

# mni-ml

Understanding machine learning by building it from scratch.

[mni-ml.github.io](https://mni-ml.github.io)

## About

mni-ml is a curriculum-style blog that teaches machine learning from the ground up. Rather than treating models as black boxes, each article walks through the math, intuition, and implementation — building everything from scratch using the [mni-ml/framework](https://github.com/mni-ml/framework), a TypeScript ML framework with Rust native backends.

## What it covers

The curriculum progresses from fundamentals to full model training:

- **Intro to ML** — What machine learning is, how models learn from data, and the core loop of prediction, loss, and optimization.
- **Gradient Descent** — How gradients drive learning, computation graphs, the forward and backward pass, and automatic differentiation.
- **CUDA & GPU Computing** — Why GPUs matter for ML, how CUDA parallelism works, and how the framework leverages native GPU acceleration.
- **Transformers** — Attention mechanisms, the transformer architecture, and training a language model on TinyStories from scratch.

## Interactive demos

- [Transformer Token Explorer](https://mni-ml.github.io/demos/transformer) — A 12M-parameter GPT trained on TinyStories, running inference entirely in the browser using the framework's pure TypeScript web backend. See next-token predictions and their probability distributions in real time.

## Development

```sh
pnpm install
pnpm dev
```

Articles are authored in Notion and pulled at build time via a custom Astro content loader.

## Built with

- [Astro](https://astro.build) — Static site generator
- [@mni-ml/framework](https://github.com/mni-ml/framework) — TypeScript ML framework with Rust native backends
- [React](https://react.dev) — Interactive demo components
- [Notion](https://notion.so) — Content management
