# CudaTextSearch

CUDA-accelerated whole-word token counting with a CPU baseline and per-word timing stats.

## Highlights
- Whole-word token matching using ASCII letter boundaries.
- GPU kernel with block-level reduction to cut atomic contention.
- Reuse of a GPU-resident text buffer for repeated queries.
- Reports avg/min/max timings for CPU, GPU host, and GPU kernel.

## Requirements
- CMake 3.17+
- C++17 compiler
- CUDA Toolkit (nvcc) and a CUDA-capable NVIDIA GPU
- If needed, set `CMAKE_CUDA_ARCHITECTURES` to match your GPU

## Build
```bash
cmake -S . -B build
cmake --build build -j
```

## Run
```bash
./build/cw2 [iterations] [block_size]
```

- `iterations`: number of timing iterations per word (default: 10)
- `block_size`: CUDA threads per block (default: 256)

Example:
```bash
./build/cw2 20 256
```

The executable expects `dataset/` next to it; CMake copies it on build.

## Output metrics
For each word, the program prints:
- CPU count and timing (avg/min/max, ms)
- GPU count and timing for two scopes:
  - GPU host: end-to-end call time (token transfer + launch + sync)
  - GPU kernel: device time measured with CUDA events

Example format (numbers vary by hardware):
```
Iterations per word: 10 | GPU block size: 256
Word: sword | CPU count: 123 | GPU count: 123 | Diff: 0
  CPU ms    (avg/min/max): 2.34 / 2.10 / 2.89
  GPU host  (avg/min/max): 0.45 / 0.41 / 0.53
  GPU kernel(avg/min/max): 0.12 / 0.10 / 0.15
```

## How it works (short)
The CPU path scans the text linearly, checks for a token match, and verifies non-letter boundaries on both sides. The GPU path launches one thread per potential start index, performs the same checks, then reduces matches within each block and atomically adds per-block totals to the global count. The text is uploaded once and reused across tokens to reduce transfer overhead.

## Dataset
`dataset/` includes several classic texts (e.g., `shakespeare.txt`, `pride_and_prejudice.txt`). The default run uses `dataset/shakespeare.txt`. You can point to a different file by changing `filepath` in `main.cpp`.

## Notes and limitations
- Text is lowercased on load; tokens should be provided in lowercase for consistent matching.
- Word boundaries treat only `a`-`z` as letters; punctuation, whitespace, and numbers are separators.
- The entire text is loaded into memory and uploaded to the GPU.
- A small set of sanity tests runs before benchmarking and will abort on mismatch.

## License
MIT. See `LICENSE`.
