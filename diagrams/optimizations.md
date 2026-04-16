# Optimizations Diagrams

## CUDA Execution Hierarchy

SVG: `/public/diagrams/opt-cuda-hierarchy.svg`
URL: `https://mni-ml.github.io/diagrams/opt-cuda-hierarchy.svg`

Place after the hierarchical overview list (Thread, Warp, Block, Grid)
and before the vector addition example. Shows the nested structure:
Grid contains Blocks, Blocks contain Warps, Warps contain Threads.
