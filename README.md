# dtw_cuda

🚀 CUDA-accelerated Dynamic Time Warping (DTW) for PyTorch.

This library provides efficient computation of all-pairs DTW distances between sequences using rolling buffers and shared memory in CUDA.

---

## Features

- ✅ DTW on 1D or 2D float sequences
- ✅ Optimized memory usage with rolling buffer (O(L) space)
- ✅ All-pairs distance computation: input `[N, L] → [N, N]`
- ✅ CUDA shared memory & parallelism
- ✅ Fallback CPU implementation for debugging

---

## Installation

```bash
pip install torch
python setup.py install
