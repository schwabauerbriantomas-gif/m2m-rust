# m2m-rust

<div align="center">

**Native Rust port of M2M Vector Search — HRM2 hierarchical retrieval, HNSW, and quantization.**

[![CI](https://github.com/schwabauerbriantomas-gif/m2m-rust/actions/workflows/ci.yml/badge.svg)](https://github.com/schwabauerbriantomas-gif/m2m-rust/actions/workflows/ci.yml)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)
[![Rust 2021](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)

</div>

---

## Overview

Native Rust port of [m2m-vector-search](https://github.com/schwabauerbriantomas-gif/m2m-vector-search),
the Python vector search engine. Written in ~23K lines of Rust with ndarray + rayon.

**Features:**

- **HRM2 hierarchical retrieval** — two-level KMeans clustering with IVF-style candidate pruning
- **HNSW** incremental indexing for approximate nearest neighbor search
- **Quantization** — TurboQuant (4/8-bit) and PolarQuant (3-bit)
- **Knowledge graph** with entity extraction and relation mapping
- **Storage** — SQLite, JSON, WAL persistence
- **HTTP API** (axum) with API key auth
- **259 unit tests** passing

> **Note on GPU:** CUDA support is stubbed via `cudarc` bindings but no native CUDA kernels
> are implemented yet. GPU code paths fall back to CPU. The `--features cuda` flag compiles
> but does not activate GPU compute.

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
├── gpu/                   — CUDA wrappers (cudarc) — stub, CPU fallback
├── quant/                 — TurboQuant, PolarQuant, QJL
├── cluster/               — Distributed sharding + routing
├── storage/               — SQLite, JSON, WAL persistence
└── api_server.rs          — HTTP API (axum) with API key auth
```

## Build

```bash
# CPU-only (no GPU required)
cargo build --release

# With CUDA feature flag (stubs, falls back to CPU)
cargo build --release --features cuda
```

## Test

```bash
cargo test --lib    # 259 tests, CPU-only
```

## Relationship to Python version

| | m2m-vector-search (Python) | m2m-rust (this repo) |
|---|---|---|
| Language | Python + NumPy/Numba | Rust + ndarray/rayon |
| GPU | Vulkan + CUDA (PyTorch) | Stub (cudarc, not implemented) |
| Backend | CPU/Vulkan/CUDA | CPU only |
| Status | v2.3.0 — production | v2.1.0 — native port |
| Successor | — | **[splatdb](https://github.com/schwabauerbriantomas-gif/splatdb)** (active development) |

> **This repo is superseded by [splatdb](https://github.com/schwabauerbriantomas-gif/splatdb)**
> for production Rust vector search with real CUDA kernels, HNSW, and quantization.

## License

AGPL-3.0
