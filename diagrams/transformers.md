# Transformers Diagrams

## 1. Attention Heatmap

SVG: `/public/diagrams/transformer-attention.svg`
URL: `https://mni-ml.github.io/diagrams/transformer-attention.svg`

Place after the paragraph about the library analogy (Query/Key/Value),
before the sentence example. Shows a 5x5 attention weight grid using
the blog's example sentence "I'm building a minimal ML".

## 2. Causal Attention Mask

SVG: `/public/diagrams/transformer-causal-mask.svg`
URL: `https://mni-ml.github.io/diagrams/transformer-causal-mask.svg`

Place in the "Attention Masking" section. Shows the lower-triangular
pattern: lit cells (visible) vs dark cells (masked). Each token can
only attend to itself and tokens before it.

## 3. Transformer Block

SVG: `/public/diagrams/transformer-block.svg`
URL: `https://mni-ml.github.io/diagrams/transformer-block.svg`

Place at the start of the Encoder section or the Decoder-Only section.
Shows the repeating block: Multi-Head Attention -> Add & Norm -> FFN ->
Add & Norm, with skip connections and x N bracket, feeding into
Linear + Softmax.
