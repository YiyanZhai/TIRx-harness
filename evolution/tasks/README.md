# Workload declarations

`grouped_gemm_fp8` uses the existing Grouped GEMM
preparation and DeepGEMM baseline from `tirx-kernels` on B200.
All four configurations must pass; report per-workload and geometric-mean speedup.
The `benchmark` dependency group installs the matching DeepGEMM wheel
(Python 3.12/3.13, Linux x86_64, glibc >= 2.38). The GPU worker needs
CUDA Toolkit 13.2 for runtime JIT; set `CUDA_HOME` if it is not on PATH.
Use `uv run --package tirx-evolution evolve init --task grouped_gemm_fp8`
with `--remote URL` for KCoral; the generated prompt includes evaluation commands.

`evolution/tasks/` contains one immutable benchmark contract per task.
`evolution/toolsets/kda_flow.yaml` is the authority for the skills and shared
leak bans installed by KDA setup. The loaders live in
`evolution/preparation/declare.py`.

## Registering a task

Every workload runs through the single packaged entry point:

```text
python evolution/benchmark/adapter.py <workload_dir> vN
```

Add the task's inputs, correctness gate, timing baseline, definition, and
workload rows to `evolution/benchmark/flashinfer_bench_evolve/tasks/`, then
map its workload directory to that package task in the adapter's `PACKAGED`
table. Workload leaves contain no committed benchmark source; candidate
`v<N>/` directories are created only inside generated run worktrees.

The adapter's `PACKAGED` registry fixes each task's shape mode for both local
and remote scoring. `BENCH_OFFICIAL_SHAPE_MODE` cannot override it. Tasks
without a specific shape suffix use `all`, including any synthetic stress
rows supplied by the packaged task. Register single-shape tasks separately
with a shape suffix, as with KDA forward and backward.

Use `pinned` for selection by a reviewed UUID in
`adapter.PINNED_WORKLOAD_UUIDS`. Use `max` for the packaged benchmark's
own largest-shape selection, as the existing KDA forward/backward tasks do.
The adapter resolves `pinned` rows before passing explicit workloads to the
packaged benchmark; it delegates `all` and `max` to the package unchanged.

The six CAKE single-shape tasks use explicit dimension suffixes, following
the existing `kda_forward_b1_t8192_h96` convention:

| Workload | Single-shape task |
| --- | --- |
| Alpha-MoE | `alphamoe_m128_e512_topk10_k2048_i128_fp8` |
| KDA decode | `kda_decode_b128_t1_h16_hv32_d128_bf16` |
| MLA DSV4 prefill | `mla_dsv4_prefill_b2_qsum386_qmax257_h128_d512_swa16384_c16384_topk1152_bf16_hnd_varlen` |
| MSA prefill | `msa_prefill_b1_q4096_kv4096_hq64_hkv4_d128_topk16_bf16_flat` |
| MSA decode | `msa_decode_b128_q16_kv4096_hq64_hkv4_d128_topk16_bf16_flat` |
| VSA | `vsa_s80000_h8_d128_blk128_topk156_bf16` |

Each task's YAML filename matches its `name`. Its `workload_dir` uses the
same suffix under the existing workload family (`kda/decode_...`,
`msa/prefill_...`, and `msa/decode_...`). `hq` and `hkv` distinguish MSA
query and KV heads; KDA's `h` and `hv` distinguish Q/K and value heads.
`d`, `blk`, and `topk` denote head dimension, block size, and selection count.
For Alpha-MoE, `m`, `e`, `k`, and `i` denote tokens, experts, hidden size,
and intermediate size; `fp8` refers to weights and quantized computation,
with bf16 activation inputs.

Alpha-MoE, KDA decode, MSA prefill/decode and VSA use fixed sequence lengths.
For fixed attention tasks, both Q and KV lengths are uniform across
requests; a uniform `cu_seqlens` array is compatible with this rule.

MLA DSV4 explicitly selects the large varlen prefill row. Its `qsum386` and
`qmax257` suffixes mean 386 total queries and at most 257 per request; the
actual Q lengths are [129,257]. The SWA and compressed KV pools have base
lengths 16384, with actual lengths [16384,16640] and [16384,16448]. All KV
lengths fill complete pages. `topk1152` is the sparse width including 128
SWA slots; active sparse lengths vary from 832 to 1152. Query and pools
use bf16/HND. The row has 56,918,016 nominal query/key pairs across heads
and 49,545,216 valid pairs. Its 4K-pool counterpart has the same pair counts;
the selected 16K row uses larger KV pools. This task retains the original
varlen contract and published seed.

`nvfp4_attention` already contains one row and has no duplicate. Each
single-shape declaration states the selected official row and its full
contract. The six choices favor substantial work within the intended
contract and baseline arm, with consistent storage. Power-of-two block
counts are not required.
All six tasks use the `pinned` mode. `adapter.PINNED_WORKLOAD_UUIDS`
selects their reviewed official rows for both local and remote scoring.
Pinned selection excludes synthetic stress rows and requires exactly one
matching official UUID. It preserves original row metadata, including seeds,
and uses the full official list regardless of `BENCH_INCLUDE_OFFICIAL` or
`BENCH_MAX_OFFICIAL`. Other tasks keep their existing shape selection.

Declare the optimization contract in `evolution/tasks/<name>.yaml`:

```yaml
name: my_kernel
workload_dir: candidates/my_kernel
kernel_authoring: TIRx-lite
sota_baseline:
  name: baseline_name
bench_timeout_s: 120
banned_paths:
  - .claude/skills/tirx-wiki/references/repos/baseline-source/**
spec: |
  Complete immutable math, tensor, interface, correctness, and scoring
  contract for the workload.
```

`name`, `workload_dir`, `kernel_authoring`, `sota_baseline.name`, and a
non-empty `spec` are required. `kernel_authoring` is `TIRx-lite` for the shared TIRx-lite
contract or `task` when the task spec defines another implementation language.
`bench_timeout_s`, when present, must be positive.

## Leak bans

- Ban the workload's own SOTA source and any prior solution corpus.
- Setup also bans the source checkout and other run directories, while
  allowing the current run to access its own files.
- Source carrying a reference implementation is physically removed from the
  generated worktree; the hook is an additional access boundary, not a
  substitute for removal.
- Setup warns if a `tirx-kernels/` ban matches nothing in the installed
  kernel package (`tirx_kernels/` paths only) or in the reference checkout of
  upstream main. When upstream moves banned files, update the ban.
- Shared prior-solution corpora belong in `evolution/toolsets/kda_flow.yaml`;
  workload-specific bans belong in the task YAML.

## Model weights

`HF_MODEL_PATH` is the harness's model root on the machine executing the benchmark;
it defaults to `/raid/catalyst/models`. Set it in the local benchmark shell
or, for remote runs, the GPU worker's startup environment. Remote clients
submit the task and candidate without needing the worker's filesystem paths.

Each workload uses its declared subdirectory under this root. Existing files
are read directly; missing files are downloaded from Hugging Face at the
workload's pinned revision into the same subdirectory. This also completes
partial downloads on a later invocation. Reading a complete model needs no
network access or filesystem writes. `HF_HOME` and `HF_HUB_CACHE` do not select
the benchmark's model directory.

Each model's section below declares its subdirectory, source, revision and
required files. Provisioned files must match that revision; existing files
are reused without version or integrity checks. Install the `benchmark`
dependency group on the GPU worker.

Downloads require network access and a writable model directory. To retain
weights across requests, keep the model root on persistent storage visible
to each worker. A read-only, offline worker must have the complete model
mounted before running. Downloads count toward the request timeout, outside
the GPU score. For KCoral's server configuration example, see
[Model weights on the worker](../../docs/components/kcoral.md#model-weights-on-the-worker).

## Qwen3.8 full-model workloads

These six TIRx-lite tasks score a prepared Qwen3.8-27B forward, including
all 64 layers, full-vocabulary logits and cache/state writes:

| Workload | Multi-shape (13 cases) | Single-shape | Selected case (batch, previous, new) |
| --- | --- | --- | --- |
| Decode | `qwen38_decode` | `qwen38_decode_b128_p4096_n1` | D4: (128, 4096, 1) |
| Prefill | `qwen38_prefill` | `qwen38_prefill_b1_p0_n32768` | P3: (1, 0, 32768) |
| Expand | `qwen38_expand` | `qwen38_expand_b64_p8192_n256` | S6: (64, 8192, 256) |

`p` counts existing prefix tokens and `n` counts new tokens per request.
The selected rows cover ordinary decode, long fresh prefill and cached
expansion. Multi-shape decode also includes four-position verification;
multi-shape expand includes the three heterogeneous request batches.
The single-shape tasks use the adapter's existing `pinned` selection.

The shared `qwen38/shapes.py` and independent `qwen38/model.py` are imported
from [qwen38-inference at 05e8ef0](https://github.com/mlc-ai/qwen38-inference/tree/05e8ef0db302a2c005994b79d9a5e78b96e2d88b).
The model imports Gemma RMSNorm directly from FlashInfer, the same BF16
implementation used by the upstream SGLang wrappers, avoiding their compiled
Torch ABI dependency. Its input/cache helpers are adapted in `qwen38/baseline.py`; the three task
modules share `qwen38/benchmark.py`. Upstream licenses accompany the model.
Optimized upstream solutions are not part of the benchmark package.

These tasks follow the [model weights convention](#model-weights):

| Model setting | Value |
| --- | --- |
| Subdirectory | `Qwen3.8-27B` |
| Hugging Face repository | `Qwen/Qwen3.8-27B` |
| Pinned revision | `MODEL_REVISION` in [qwen38/model.py](../benchmark/flashinfer_bench_evolve/tasks/qwen38/model.py) |

The model directory must contain `config.json`, `generation_config.json`,
`model.safetensors.index.json` and the indexed shards. A B200, CUDA 13 C++
toolchain and sufficient GPU/host memory for the 27B weights, KV cache and
CPU state snapshots are required.

Each task invocation loads the model once and reuses its weights across all
selected shapes. Each shape prepares its own real prefix and cache. Separate
task invocations load separate model instances. Set `QWEN38_SEED` to reproduce
the same inputs across invocations; by default each invocation draws a new seed.
CPU snapshots are copied directly into pinned memory; initial all-zero states
are restored by zeroing instead of saving/transferring a snapshot.

```bash
uv run --package tirx-evolution evolve init --task qwen38_decode_b128_p4096_n1
# In the prepared run, use the generated prompt's benchmark command.
python evolution/benchmark/adapter.py candidates/qwen38/decode_b128_p4096_n1 baseline
```

Candidates export `setup(data) -> callable`. The YAML states the task contract
and points to `benchmark.tirx_prepare`, `model.Qwen38._prepare_step` and
`model.Qwen38._forward` for model data, input/cache layouts and computation.
Each call computes current logits and state with TIRx-lite. Prefix preparation,
compilation, capture and state restoration are untimed. Decode uses outer CUDA
graph replay; prefill/expand
time the prepared callable directly. Scoring uses GPU-event samples
and the upstream per-element `atol=rtol=0.001` gate on logits and caches.
The harness checks full logits and all cache tensors once before timing, then
releases those reference snapshots. Each implementation starts its timing
block from the same saved state and runs 3 warmup calls followed by 9 retained
samples without intermediate restoration. Prepared tokens, positions and
lengths stay fixed while cache state advances; this measures repeated prepared
calls, not continuous generation. Final block logits must remain finite, but
are not compared to the single-step reference. `BENCH_TRIALS` defaults to 1;
additional trials reverse implementation order and restore before every block.
Baseline self-checks use the reference in both blocks. Prefix preparation,
snapshots, transfers and correctness checks remain outside the GPU score.

## GR00T N1.7 action-head boundary on Jetson Thor

`groot_ah_boundary_thor` (workload `groot/ah_boundary_thor`) scores one
denoising-step boundary of the GR00T N1.7 action head on Jetson AGX Thor
(sm_110a). The boundary is the action decoder, the Euler update, the action
encoder for the next step, the position embedding and the `sa_embs`
concatenation. The task requires one persistent kernel per call: the
harness counts the kernel nodes of the captured candidate graph. The three
rows are the three boundaries of one real LIBERO frame, and the score is
the sum of the TensorRT medians divided by the sum of the candidate medians.

The baseline is the deployed sequence: the pipeline's TensorRT 10.16.1.11
action-decoder and action-encoder engines plus the GR00T runtime's PyTorch
glue, captured in one CUDA graph and timed in the same run. The oracle is an
independent float64 implementation with the exported graph's fp16 storage
points. The baseline is 1 fp16 ulp from it.

These tasks follow the [model weights convention](#model-weights):

| Model setting | Value |
| --- | --- |
| Subdirectory | `groot-n17-libero-c235` |
| Source | no download; built by `kda/tirx/m1_prepare.py` in groot-perf-diagnosis (a private repository on the Thor; the worker already holds the files) |
| Files | `action_decoder.engine`, `action_encoder.engine`, `weights.safetensors`, `golden.safetensors`, `manifest.json` (SHA-256 of the others) |

The GPU worker needs:
- TensorRT 10.16 Python bindings. `GROOT_TRT_SITE` names a side-by-side
  site directory, default `/opt/trt-10.16.1/site`.
- `safetensors` and `cuda.bindings`.
- Optionally, a shared GPU timing lock. `GROOT_GPU_LOCK` defaults to
  `/sources/.thor-gpu-timing.lock`; the suite skips the lock if the file does
  not exist. Otherwise it holds the lock for the whole run. It waits for the
  lock at most `GROOT_GPU_LOCK_WAIT_S` (default 2700 s, inside the task's
  3600 s budget), then fails with a clear error. Do not hold the same lock
  around a remote request on the client side, or the worker waits on its own
  client.

Each timing sample follows a 128 MB read, so the weights start in DRAM as
they do after a DiT step. The memory clock is warmed to its maximum before
timing. Baseline and candidate graph replays alternate sample by sample.

The GPU worker is the kcoral server on the Thor, which listens on the
Thor's 127.0.0.1:8901 only. The uv workspace resolves on Linux x86_64 only,
so the agent machine is an x86_64 host that reaches the server through an
SSH tunnel; all GPU work then runs on the Thor. The step-by-step runbook,
with a goal prompt for each task, is [groot_thor_runbook.md](groot_thor_runbook.md):

```bash
ssh -N -L 8901:127.0.0.1:8901 yiyanz@catalyst-thor &
uv run --package tirx-evolution evolve init --task groot_ah_boundary_thor --remote http://127.0.0.1:8901
python evolution/remote/kcoral_remote.py groot/ah_boundary_thor baseline --remote http://127.0.0.1:8901
```

TIRx-lite code written for B200 runs on sm_110a after retargeting
(`TIRX_PREPARE_CUDA_ARCH=sm_110a`): tirx-kernels' tcgen05 fp16 GEMM and
NVFP4 block-scaled GEMM, including 2-SM clusters and TMA multicast, pass on
the Thor unchanged except for the target and the persistent grid size (20
SMs). Tile, grid and pipeline tuning do not carry over. The kcoral worker on
the Thor exports `TIRX_PREPARE_CUDA_ARCH=sm_110a`; without it, tcgen05 kernels
traced with the B200 default `arch="sm_100a"` fail with
`CUDA_ERROR_NO_BINARY_FOR_GPU`. IKET on the Thor needs `run-iket` from
nvidia-cutlass-dsl 4.8 or later (4.7.0 has no sm110 trace support); the
worker provides it, as the runbook describes.

## GR00T N1.7 DiT step on Jetson Thor

`groot_dit_step_thor` (workload `groot/dit_step_thor`) scores one denoising
step of the GR00T N1.7 DiT on Jetson AGX Thor (sm_110a). The step is 32
NVFP4 W4A4 transformer blocks on 41 tokens, alternating cross- and
self-attention, plus the output head. The task requires one persistent
kernel per call: the harness counts the kernel nodes of the captured
candidate graph. The four rows are the four steps of one real LIBERO frame,
and the score is the sum of the TensorRT medians divided by the sum of the
candidate medians. TensorRT runs 516 kernels and takes 3.86-3.89 ms per step
here. The DRAM floor is 1.90 ms.

The model files (`dit_nvfp4.engine`, `dit_weights.safetensors`,
`dit_golden.safetensors`, `dit_manifest.json`) sit in the same
`groot-n17-libero-c235` directory as the action-head task. They are built by
`kda/tirx/m2_prepare.py` in groot-perf-diagnosis. The worker requirements,
timing protocol and lock are the action-head task's: the task imports that
task's helpers, so both packages travel together.

The correctness gate sits at the NVFP4 noise floor. Dynamic activation
quantization flips borderline roundings, so TensorRT and an independent
Triton implementation both differ from the PyTorch oracle by 0.026-0.049
relative L2 per step. Each step allows 1.4x the larger of the two in
relative L2 and 2x in max abs error. Controls:

| control | per-step relative L2 vs the oracle | verdict |
| --- | --- | --- |
| TensorRT (baseline) | 0.027 / 0.033 / 0.038 / 0.048 | PASS |
| cross-attention masks swapped | 0.36-0.89 | FAIL |
| block 31 skipped | 0.11-0.21 | FAIL |
| activations not quantized (W4A16) | 0.062-0.076 | FAIL |
| output cached in setup | caught by the in-place input change | FAIL |
| correct PyTorch implementation | passes, then 6415 kernels | FAIL (one-kernel check) |

Defects below the floor, such as a missing bias in one block or a wrong
epsilon, are not detectable from the step output. Before deployment a
candidate must also pass the pipeline's task-level gate.

```bash
python evolution/remote/kcoral_remote.py groot/dit_step_thor baseline --remote http://127.0.0.1:8901
```

## GR00T N1.7 LLM prefill on Jetson Thor

`groot_llm_prefill_thor` (workload `groot/llm_prefill_thor`) scores the
prefill of the GR00T N1.7 backbone LLM on Jetson AGX Thor (sm_110a): 16
Qwen3 layers on the 156 vision-language tokens of one frame. q / k / v and
gate_up are NVFP4 W4A4, o and down are FP8; the layers also do per-head
q/k RMSNorm, interleaved multimodal rope and causal GQA (16 / 8 heads x
128), and add the deepstack features after layers 0-2. The task requires one
persistent kernel per call. The four rows are four real frames of one
LIBERO episode. TensorRT runs 364 kernels and takes 5.36-5.51 ms per call
here. The DRAM floor is 2.24 ms (571 MB of weights at 255 GB/s).

The model files (`llm_nvfp4.engine`, `llm_weights.safetensors`,
`llm_golden.safetensors`, `llm_manifest.json`) sit in the same
`groot-n17-libero-c235` directory as the other GR00T tasks. They are built
by `kda/tirx/m3_prepare.py` in groot-perf-diagnosis. The task imports the
action-head task's timing, lock and graph helpers.

The gate measures error per token. Token 0 is an attention sink (values up
to ~15000, rms ~400 against ~4 for the other tokens), so a whole-tensor
relative L2 only measures that token. The primary check is the mean over
tokens of each token's relative L2. TensorRT and an independent Triton
implementation sit 0.12-0.13 from the oracle on it, because NVFP4
quantization flips are amplified over 16 layers. Each frame allows 1.4x the
larger of the two (and 1.4x their whole-tensor error, 0.9 for any single
token). Two probes change the inputs in place: new embeddings and deepstack
values (limit 0.35), and a new token layout with positions permuted and the
visual mask moved (limit 0.20). Controls (`kda/tirx/m3_controls/`):

| control | per-token mean vs the oracle | verdict |
| --- | --- | --- |
| TensorRT (baseline) | 0.127-0.133 | PASS |
| layer 15 skipped | 0.31-0.32 | FAIL |
| deepstack features not added | 0.29-0.30 | FAIL |
| rope sections contiguous instead of interleaved | 0.19-0.20 | FAIL (narrowly) |
| output cached in setup | 0.47-0.48 after the value probe | FAIL |
| rope tables and deepstack rows from setup | 0.42 after the layout probe | FAIL |
| rope tables from setup | 0.26-0.28 after the layout probe | FAIL |
| activations not quantized (W4A16) | below the floor | passes, then the one-kernel check |
| correct PyTorch implementation | passes, then 3652 kernels | FAIL (one-kernel check) |

```bash
python evolution/remote/kcoral_remote.py groot/llm_prefill_thor baseline --remote http://127.0.0.1:8901
```

## GR00T N1.7 vl_self_attention + cross-KV on Jetson Thor

`groot_vlsa_xkv_thor` (workload `groot/vlsa_xkv_thor`) scores the action
head's per-frame vision-language stage on Jetson AGX Thor (sm_110a):
vl_self_attention (4 transformer blocks on the 156 tokens: fp16 attention
projections, FP8 GELU FFN) and the keys and values of the DiT's 16
cross-attention blocks (32 FP8 linears). The task requires one persistent
kernel per call. The four rows are four real frames (the LLM task's golden
outputs through the action head's `vlln` LayerNorm). The baseline is the
deployed pair of engines back to back: 29 + 4 kernels, 2.12-2.13 ms per call
here. The DRAM floor is 1.45 ms (369 MB of weights).

The model files (`vl_self_attention.engine`, `dit_cross_kv.engine`,
`vlsa_weights.safetensors`, `vlsa_golden.safetensors`,
`vlsa_manifest.json`) are built by `kda/tirx/m4_prepare.py`. The live
engines reproduce a dumped pipeline call bit for bit.

The gate sits at the noise floor of the 4 blocks: TensorRT and an
independent Triton implementation differ from the oracle by 0.019-0.025
relative L2 on the keys and values. Each frame allows 1.4x the larger, per
output. The max abs error is heavy-tailed (a change in the last bits of one
scale reaches 0.59), so its limit is a flat 1.0. Controls
(`kda/tirx/m4_controls/`):

| control | relative L2 vs the oracle | verdict |
| --- | --- | --- |
| TensorRT (baseline) | 0.021-0.025 | PASS |
| block 3 skipped | 0.57-0.58 | FAIL |
| causal attention mask | 0.60-0.70 | FAIL |
| keys and values swapped | 1.31-1.32 | FAIL |
| outputs cached in setup | 0.27-0.29 after the input change | FAIL |
| erf GELU instead of tanh | below the floor | passes, then the one-kernel check |
| correct PyTorch implementation | passes, then 678 kernels | FAIL (one-kernel check) |

```bash
python evolution/remote/kcoral_remote.py groot/vlsa_xkv_thor baseline --remote http://127.0.0.1:8901
```
