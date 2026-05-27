# DGX Spark (GB10) CUDA Inference Optimization Plan

## Context

> **Tip:** Use the **Context7** skill to look up CUDA Python / CUDA Toolkit docs on demand.
> Examples: `context7_get_library_docs(/nvidia/cuda-python, "WGMMA INT8 matmul")` or
> `context7_get_library_docs(/websites/nvidia_cuda, "TMA async copy")`.
> Available libraries: `/nvidia/cuda-python`, `/websites/nvidia_cuda`, `/websites/nvidia_github_io_numba-cuda`.
>
> **Tip:** You can also **search the web** and **fetch URLs** to find external optimization ideas,
> blog posts, kernel implementations, and community benchmarks. Examples:
> `web_search("Blackwell GB10 WGMMA INT8 matmul optimization")` or
> `web_fetch("https://developer.nvidia.com/blog/new-features-cuda-blackwell")`.

**DwarfStar (ds4)** is a specialized inference engine for DeepSeek V4 Flash/PRO.
The CUDA backend (`ds4_cuda.cu`, ~10,700 lines) handles MoE experts, attention
with compressed KV, hyper-connections, RoPE, and various quantized matmuls.

### Current DGX Spark Baseline (from README)

| Metric | Value |
|--------|-------|
| Model | q2 |
| Context | 7047 tokens |
| Prefill | 343.81 t/s |
| Generation | 13.75 t/s |

Generation is notably slower than Metal:
- MacBook M5 Max: 34.27 t/s (q2, short)
- MacBook M3 Max: 26.68 t/s (q2, short)
- MacBook M3 Ultra: 36.86 t/s (q2, short)

This indicates significant CUDA optimization headroom.

---

## Experiment Organization

Each experiment follows this pattern:

```
1. Profile: identify the target kernel via nsys/nsight
2. Baseline: ds4-bench across relevant context frontiers
3. Implement: targeted kernel optimization
4. Verify: make test (correctness) + ds4-bench (speed)
5. Record: before/after CSV, commit diff
```

### Git Workflow

- **Branch per experiment**: `git checkout -b cuda/phase1-fuse-hc-split`
- **Commit 1** — baseline results (CSV into `speed-bench/`)
- **Commit 2** — implementation + before/after CSVs + diff notes
- **Message template**: include t/s before→after and which gates passed

```
git add speed-bench/dgx_spark_fuse_hc_baseline.csv
git commit -m "cuda: baseline for HC split fuse experiment (13.75 t/s gen)"

git add ds4_cuda.cu speed-bench/dgx_spark_fuse_hc_after.csv
git commit -m "cuda: fuse hc_split + weighted_sum + norm into single kernel

Before: 13.75 t/s gen (3 kernel launches)
After:  18.20 t/s gen (1 kernel launch)
Verified: make test, ds4-eval correctness gate passed"
```

- **Before pushing**: run `make test`, `./ds4-eval` (correctness gate), and `./ds4-bench` (speed regression check) per `CONTRIBUTING.md`

### Correctness Gate

After every inference change, run the deterministic gate (from CONTRIBUTING.md):

```sh
./ds4-eval \
  -m ds4flash.gguf \
  --plain \
  --questions 4 \
  --tokens 2048 \
  --temp 0 \
  --seed 1
```

Generated-token counts must match:

| Question | Expected state | Expected tokens | Expected answer |
|---:|---|---:|---|
| 1 | PASSED | 2048 | B |
| 2 | PASSED | 438 | C |
| 3 | PASSED | 666 | 70 |
| 4 | FAILED | 2048 | A (given) / C (correct) |

### Benchmark Template

```sh
./ds4-bench \
  -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 65536 \
  --step-incr 2048 \
  --gen-tokens 128 \
  --csv /tmp/ds4-speed.csv
```

Generate a chart:

```sh
python3 speed-bench/plot_speed.py /tmp/ds4-speed.csv --title "DGX Spark GB10 t/s"
```

Store results as `speed-bench/dgx_spark_<experiment_name>.csv`.

### Profiling Template

```sh
nsys profile --trace=cuda,nvtx -o /tmp/ds4_profile \
  ./ds4 -m ds4flash.gguf --nothink --ctx 8192 -n 128 \
  -p "Explain Redis streams in one paragraph."
```

Focus on the decode loop (single-token generation) for generation speed,
and the prefill region for prefill speed.

---

## Phase 0: Baseline & Instrumentation

- [ ] Build with `make cuda-spark`
- [ ] Run full benchmark sweep and save `speed-bench/dgx_spark_baseline.csv`
- [ ] Profile generation decode path with nsys — identify top 5 kernels by time
- [ ] Profile prefill path — identify top 5 kernels by time
- [ ] Document kernel breakdown (percentage of total time per kernel group)

---

## Phase 1: Generation Decode Path (highest impact)

The single-token decode loop iterates every layer for each generated token.
Current generation: **13.75 t/s** on DGX Spark.

### Kernels in the Decode Loop (per layer)

1. **Embedding / HC split**
   - `hc_split_weighted_sum_norm_tensor` — HC mixer split + weighted sum + RMS norm
2. **Q projection**
   - `matmul_q8_0_tensor` or `matmul_f16_tensor` (Q weights)
3. **KV projection**
   - `matmul_q8_0_tensor` or `matmul_f16_tensor` (KV weights)
4. **Q/KV RoPE + FP8 quantize**
   - `rope_tail_tensor` + `dsv4_fp8_kv_quantize_tensor`
5. **KV store (raw cache)**
   - `kv_fp8_store_raw_tensor`
6. **Attention decode**
   - `attention_decode_heads_tensor` (raw + compressed)
   - `attention_decode_mixed_batch_heads_tensor` (with indexer mask)
7. **Attention output projection**
   - `attention_output_q8_batch_tensor` (low-rank grouped matmul)
8. **Router**
   - `router_select_tensor` (per-token expert selection)
9. **MoE expert dispatch**
   - `routed_moe_one_tensor` — gate (IQ2_XXS) + up (IQ2_XXS) + down (Q2_K)
10. **Shared expert SwiGLU**
    - `shared_gate_up_swiglu_q8_0_tensor` + `swiglu_tensor`
11. **HC expand**
    - `hc_expand_tensor` / `hc_expand_split_tensor`
12. **Final norm + output projection**

### Optimization Opportunities

- [ ] **Fuse HC split + weighted sum + norm** into single kernel (reduce 2→1 launch + memory round-trip)
- [ ] **Fuse Q projection + KV projection** (shared input, two weight lookups)
- [ ] **Fuse RoPE + FP8 KV quantize + raw store** into one kernel (currently 2-3 dispatches)
- [ ] **Fuse router + MoE gate+up+down** — avoid staging selected experts in separate buffers
- [ ] **Fuse SwiGLU + HC expand** (element-wise ops on same data layout)
- [ ] **Batch attention decode** over multiple layers where Q/raw_KV layout permits

### Target

Reach 25+ t/s (comparable to M3 Max Metal) on generation.

---

## Phase 2: Prefill Path

Prefill is already fast (343 t/s) but may benefit at larger context windows.

### Kernels

- `embed_tokens_hc_tensor` — batch embedding lookup
- `compressor_prefill_tensor` — KV compression during prefill
- `compressor_prefill_ratio4_replay_tensor` — ratio-4 layer compression
- `indexer_scores_prefill_tensor` + `indexer_topk_tensor` — indexer selection
- `attention_prefill_static_mixed_heads_tensor` — mixed raw+compressed attention
- `attention_prefill_masked_mixed_heads_tensor` — with indexer mask

### Optimization Opportunities

- [ ] **Larger tile sizes** for batched projections (GB10 has larger shared mem)
- [ ] **Parallelize compressor** across heads
- [ ] **Batch indexer scoring** across all prefill tokens in one kernel
- [ ] **Async prefill** with multiple CUDA streams for chunked prefill

---

## Phase 3: GB10 (SM 120) Specific Features

GB10 / Blackwell introduces new instructions and hardware features.

- [ ] **TMA (Tensor Memory Accelerator)** for weight transfers in matmul kernels
- [ ] **WGMMA for INT8/FP8** — use WGMA instructions for Q8_0 × F16, IQ2_XXS dequantize
- [ ] **FP8 accumulators** — if MoE path is memory-bound, accumulate in FP8
- [ ] **Cooperative groups** — for MoE expert-level parallelism
- [ ] **L2 cache hints** — `__l2cache_hint()` for weight streaming vs KV cache residency
- [ ] **Multi-block cluster** — for attention across blocks (if applicable)

### Build note

`make cuda-spark` currently omits `-arch` flag (fastest default on GB10).
For explicit SM 120 features:

```sh
make cuda CUDA_ARCH=sm_120
```

---

## Phase 4: Weight Loading & Caching

The weight caching layer already supports async staging with 4 streams
(`g_model_stage[4]`), but may not be fully utilized.

- [ ] **Q8→F16 dequant cache**: `dequant_q8_0_to_f16_kernel` runs on every use
  unless cached — verify cache hit rates on DGX Spark (128GB device memory)
- [ ] **Async prefetch timing**: ensure staging stream overlaps with compute
- [ ] **cudaMemcpyAsync vs TMA**: evaluate whether TMA can replace async copies
- [ ] **Managed memory** path (`cudaHostRegister + cudaHostGetDevicePointer`) —
  verify zero-copy performance on GB10

---

## CUDA Kernel Inventory

Reference: `ds4_cuda.cu` (~10,700 lines, `ds4_gpu.h` for public API).

### Quantization Formats

| Format | Used in | Notes |
|--------|---------|-------|
| Q8_0 | shared experts, attention output | 1 scale per 128 bytes, bsums for fast dot |
| IQ2_XXS | MoE gate + up (routed experts) | 2-bit asymmetrical, lookup tables |
| Q2_K | MoE down (routed experts) | 2-bit with per-group scales |
| FP8 (E4M3) | KV cache (compressed) | per-token or per-head scaling |
| FP16 | embeddings, norms, projections | standard BF16/FP16 matmul |

### Kernel Categories

| Category | Key Kernels | Phase |
|----------|------------|-------|
| HC (hyper-connection) | `hc_split_sinkhorn`, `hc_weighted_sum`, `hc_expand` | 1 |
| Matmul | `matmul_q8_0`, `matmul_f16`, `matmul_f16_pair`, `matmul_f32` | 1, 3 |
| Norm/RoPE | `rms_norm_*`, `rope_tail`, `head_rms_norm` | 1 |
| KV quantize/store | `dsv4_fp8_kv_quantize`, `kv_fp8_store_raw`, `store_raw_kv` | 1 |
| Attention | `attention_decode_*`, `attention_prefill_*` | 1, 2 |
| Compressor | `compressor_update`, `compressor_store_batch`, `compressor_prefill_*` | 1, 2 |
| Indexer | `indexer_score_*`, `indexer_topk`, `dsv4_topk_mask` | 1, 2 |
| Router | `router_select`, `router_select_batch` | 1 |
| MoE | `routed_moe_one`, `routed_moe_batch` | 1 |
| SwiGLU | `shared_gate_up_swiglu_q8_0`, `swiglu` | 1 |
| Embedding | `embed_token_hc`, `embed_tokens_hc` | 0 |
| Steering | `directional_steering_project` | — |

---

## Results Tracking

| Experiment | Date | Prefill t/s | Gen t/s | Notes |
|-----------|------|------------|---------|-------|
| Baseline | — | 343.81 | 13.75 | README published numbers |
