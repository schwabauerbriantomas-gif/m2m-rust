# m2m-rust

<div align="center">

**Native Rust implementation of M2M Vector Search — GPU-accelerated with custom CUDA kernels.**

[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)
[![Rust 2021](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)
[![Tests: 226](https://img.shields.io/badge/tests-226%20passing-brightgreen.svg)]()

</div>

---

## Overview

Native Rust port of [m2m-vector-search](https://github.com/schwabauerbriantomas-gif/m2m-vector-search),
the Python vector search engine. This implementation provides:

- **20K+ lines** of pure Rust + CUDA C
- **5 custom CUDA kernels** (L2, cosine, batch, top-k variants)
- **HRM2 hierarchical retrieval** — two-level KMeans clustering
- **HNSW** incremental indexing for 640x speedup over linear scan
- **Quantization** — TurboQuant (4/8-bit) and PolarQuant (3-bit)
- **Knowledge graph** with entity extraction and relation mapping
- **Distributed sharding** — cluster protocol with energy-aware routing
- **226 tests** passing with 0 warnings

## Architecture

```
src/
├── splats.rs              — SplatStore: high-level vector operations
├── hrm2_engine.rs         — HRM2 hierarchical retrieval engine
├── engine.rs              — CPU-optimized distance computation (rayon)
├── clustering.rs          — KMeans++ with Lloyd's algorithm
├── energy.rs              — Energy-based model functions
├── geometry.rs            — Manifold geometry operations
├── splat_types.rs         — GaussianSplat, Hrm2Node, MemoryPartition
├── encoding.rs            — Embedding builders and encoders
├── hnsw_index.rs          — HNSW incremental indexing
├── graph_splat.rs         — Knowledge graph traversal
├── gpu/                   — CUDA kernel wrappers (cudarc)
├── quant/                 — TurboQuant, PolarQuant, QJL
├── cluster/               — Distributed sharding + routing
├── storage/               — SQLite, JSON, WAL persistence
└── api_server.rs          — HTTP API (axum) with API key auth
```

## Performance

All benchmarks measured on **2026-03-29**. Hardware: AMD Ryzen 5 3400G (4c/8t), 32GB DDR4, NVIDIA RTX 3090 (24GB VRAM, sm_86, CUDA 12.4).

### GPU Search (persistent VRAM)

| Vectors | Dim | CPU QPS | GPU QPS | Speedup |
|---------|-----|---------|---------|---------|
| 10K | 640 | 269 | 1,667 | **6.2x** |
| 100K | 640 | 214 | 1,667 | **7.8x** |

GPU query time is constant regardless of dataset size (VRAM-resident, compute-bound).

### HNSW vs Linear Scan

| Method | 10K | 100K | Recall@10 |
|--------|-----|------|-----------|
| Linear | 269 QPS | 214 QPS | 100% |
| HNSW | 314K QPS | 137K QPS | >99.5% |
| **Speedup** | **1,170x** | **640x** | — |

### DatasetTransformer Pipeline

100K × 640D vectors → 100 Gaussian splats (1,000:1 compression):
- One-time ingest: 2.1 min (KMeans dominates)
- Search: 20,000 QPS over 100 splats (50µs/query)
- 12x faster queries than GPU raw search

See [BENCHMARKS.md](BENCHMARKS.md) for full details.

## Build

```bash
# CPU-only (no GPU required)
cargo build --release

# With CUDA support (requires CUDA Toolkit 12.x)
cargo build --release --features cuda
```

## Test

```bash
cargo test --lib          # 226 tests, CPU-only
cargo test --lib --features cuda   # same tests, GPU modules available
```

## Relationship to Python version

| | m2m-vector-search (Python) | m2m-rust (this repo) |
|---|---|---|
| Language | Python + NumPy | Rust + ndarray |
| GPU | Vulkan | CUDA (custom kernels) |
| Backend | CPU/Vulkan/CUDA | CPU/CUDA |
| Speed | Baseline | ~10-50x faster (compiled) |
| Status | Production | Native port |

## License

AGPL-3.0
