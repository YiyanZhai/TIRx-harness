# Running the GR00T Thor tasks from an x86_64 agent machine

The agent runs on an x86_64 machine (the RTX 5090 box). Every compile, score and profile request goes to the kcoral
server on the Jetson AGX Thor. The local GPU is never used for scoring: it is sm_120, and every task stops unless the
device is compute capability 11.0.

Everything the agent machine needs is in this fork (branch `groot-ah-boundary-thor`) or on the Thor's worker. The
model files, the TensorRT engines and the timing lock live only on the worker. The tasks were built in
groot-perf-diagnosis, a private repository on the Thor that has no remote. The agent machine does not need it: its
controls, calibration runs and the 997-frame task gate stay there.

## 1. One-time setup

Prerequisites: `uv`, Claude Code, and SSH access to the Thor (`yiyanz@catalyst-thor`, 192.168.10.117 on the lab
LAN). Add the agent machine's public key to the Thor's `~/.ssh/authorized_keys`.

```bash
git clone -b groot-ah-boundary-thor git@github.com:YiyanZhai/TIRx-harness.git
cd TIRx-harness
git submodule update --init thirdparty/tvm-rust-ext
```

The Thor's kcoral listens on the Thor's 127.0.0.1:8901 only. Keep a tunnel open for the whole run (autossh, tmux):

```bash
ssh -N -L 8901:127.0.0.1:8901 yiyanz@catalyst-thor
curl -fsS http://127.0.0.1:8901/health    # expect "target":{"arch":"sm_110a"}
```

## 2. Prepare one run per task

| order | task (`--task`) | workload (kcoral_remote) | TensorRT baseline | reference points |
|---|---|---|---|---|
| 1 | `groot_ah_boundary_thor` (M1) | `groot/ah_boundary_thor` | 130-135 us per boundary, 13 kernels | cold-read floor ~47 us; best per-op chain 50-60 us |
| 2 | `groot_dit_step_thor` (M2) | `groot/dit_step_thor` | 3.86-3.89 ms per step, 516 kernels | best per-op chain ~1.99 ms |
| 3 | `groot_vlsa_xkv_thor` (M4) | `groot/vlsa_xkv_thor` | 2.12-2.13 ms, 29 + 4 kernels | DRAM floor 1.45 ms; best chain ~1.73 ms |
| 4 | `groot_llm_prefill_thor` (M3) | `groot/llm_prefill_thor` | 5.36-5.51 ms, 364 kernels | DRAM floor 2.24 ms; best chain 2.42 ms |

```bash
uv run --package tirx-evolution evolve init --task groot_ah_boundary_thor --name m1-ah-boundary \
  --remote http://127.0.0.1:8901
```

Copy the three shell lines that setup prints (run_dir, cd, activate). Then run the baseline self-check once from the
worktree; TensorRT runs in both arms, so the aggregate should be 0.98-1.02x:

```bash
python evolution/remote/kcoral_remote.py groot/ah_boundary_thor baseline --remote http://127.0.0.1:8901
```

Measured self-checks on the Thor: M1 0.999x, M2 0.996x, M3 0.990-0.993x, M4 1.019x.

Run at most two tasks at a time. There is one Thor GPU: the server has one worker, and every suite takes the shared
GPU timing lock, which other users of the Thor can hold for 20+ minutes. A request waits up to its 3600 s budget, then
fails with a clear error.

## 3. Launch the agent

```bash
cd "$run_dir/worktree" && source .venv/bin/activate && claude
```

Then enter the task's goal (one line in Claude Code; wrapped here for reading):

M1:

```text
/goal Read ../PROMPT.md and reach at least 2.0x aggregate speedup over the deployed TensorRT sequence on the Jetson
Thor (stretch: 2.5x) with ONE TIRx-lite persistent kernel per call that passes every correctness check on all three
boundaries. Score only through evolution/remote/kcoral_remote.py --remote http://127.0.0.1:8901; never run the
benchmark on the local GPU (sm_120). Send one scoring request at a time and do not retry while one is queued: the
Thor GPU is shared and requests can wait for its timing lock. Never change clocks or power: pass
--clock-control none to every kcoral_ncu.py run (ncu's default locks the Thor's clocks). Start with the baseline
self-check, then get a correct one-kernel candidate before optimizing. Remeasure a promising candidate before keeping
it: per-row noise is about +-4 %, aggregate about +-2 %.
```

M2:

```text
/goal Read ../PROMPT.md and reach at least 1.5x aggregate speedup over the deployed TensorRT DiT engine on the Jetson
Thor (stretch: 1.9x) with ONE TIRx-lite persistent kernel per DiT step that passes every correctness check on all
four steps. Score only through evolution/remote/kcoral_remote.py --remote http://127.0.0.1:8901; never run the
benchmark on the local GPU (sm_120). Send one scoring request at a time and do not retry while one is queued: the
Thor GPU is shared and requests can wait for its timing lock. Never change clocks or power: pass
--clock-control none to every kcoral_ncu.py run (ncu's default locks the Thor's clocks). Start with the baseline
self-check, then get a correct one-kernel candidate before optimizing. Remeasure a promising candidate before keeping
it.
```

M4:

```text
/goal Read ../PROMPT.md and reach at least 1.15x aggregate speedup over the deployed TensorRT vl_self_attention and
cross-KV engines on the Jetson Thor (stretch: 1.25x) with ONE TIRx-lite persistent kernel per call that passes every
correctness check on all four frames. Score only through evolution/remote/kcoral_remote.py --remote
http://127.0.0.1:8901; never run the benchmark on the local GPU (sm_120). Send one scoring request at a time and do
not retry while one is queued: the Thor GPU is shared and requests can wait for its timing lock. Never change
clocks or power: pass --clock-control none to every kcoral_ncu.py run (ncu's default locks the Thor's clocks).
Start with the baseline self-check, then get a correct one-kernel candidate before optimizing. Remeasure a promising
candidate before keeping it.
```

M3:

```text
/goal Read ../PROMPT.md and reach at least 1.5x aggregate speedup over the deployed TensorRT LLM engine on the Jetson
Thor (stretch: 2.0x) with ONE TIRx-lite persistent kernel per call that passes every correctness check on all four
frames. Score only through evolution/remote/kcoral_remote.py --remote http://127.0.0.1:8901; never run the
benchmark on the local GPU (sm_120). Send one scoring request at a time and do not retry while one is queued: the
Thor GPU is shared and requests can wait for its timing lock. Never change clocks or power: pass
--clock-control none to every kcoral_ncu.py run (ncu's default locks the Thor's clocks). Start with the baseline
self-check, then get a correct one-kernel candidate before optimizing. Remeasure a promising candidate before keeping
it.
```

Why these targets: M1's TensorRT is latency-bound small GEMMs (2x = ~65 us, near the best per-op chain); M2's best
chain is 1.95x and a megakernel pays 82-373 us per step in cross-CTA publishes; M4's DRAM floor caps it at 1.46x;
M3's best chain is 2.2x and its floor 2.4x.

## 4. What the worker provides

- kcoral on the Thor: target sm_110a, torch 2.10 (NV 25.12), CUDA 13.1, TVM 0.27.0.post1, apache-tvm-ffi
  0.1.14.post0, tirx-kernels 0.1.2.post1 (the versions `uv.lock` pins).
- `TIRX_PREPARE_CUDA_ARCH=sm_110a` is exported, so kernels written with tirx-kernels' B200 default
  `arch="sm_100a"` are retargeted. Without it, tcgen05 kernels fail with `CUDA_ERROR_NO_BINARY_FOR_GPU`.
- TensorRT 10.16.1.11 (the deployed engines' release) and the model files under `$HF_MODEL_PATH/groot-n17-libero-c235`.
  They are on the worker only; inspect them remotely, for example:

  ```bash
  python evolution/remote/kcoral_python.py --remote http://127.0.0.1:8901 -- -c '
  import os, json
  d = os.path.join(os.environ["HF_MODEL_PATH"], "groot-n17-libero-c235")
  print(sorted(os.listdir(d)))
  print(json.load(open(os.path.join(d, "dit_manifest.json"))))'
  ```

- A TIRx-lite kernel is captured in the harness's CUDA graph as one kernel node; tvm-ffi follows torch's current
  stream. A one-kernel TIRx-lite control passed correctness through the real remote path.
- IKET works through `kcoral_iket.py`. The worker's `run-iket` comes from nvidia-cutlass-dsl 4.8.1
  (`/opt/iket-cutlass-4.8.1`); 4.7.0, the version `uv.lock` pins, has no sm110 trace support. A shim maps run-iket
  4.8's `IKET_INJECTION_CONFIG` to the `SMODEL_INJECTION_CONFIG` that TVM 0.27 reads. Checked 2026-10-31 with an
  annotated TIRx-lite kernel through the remote path and with tirx-kernels' FA4 (tcgen05, TMA, 2336 ranges on 20
  SMs); clusters larger than 1 are untested.
- ncu on the Thor must run with `--clock-control none`, and even then it flushes L2 and the clocks drop
  between replays, so kernel times read 2-5x their in-chain values. Use the benchmark's timings.
- Compute Sanitizer, nsys, nvdisasm and cuobjdump (CUDA 13.1) are on the worker; reach the last three with
  `kcoral_remote.py --cmd`.

## 5. Results and rechecks

Candidates live in the run worktree under `candidates/groot/<task>/{scratch,frontier}/<candidate>/solution.py`.
Recheck one from the worktree:

```bash
python evolution/remote/kcoral_remote.py groot/ah_boundary_thor frontier/<candidate> \
  --remote http://127.0.0.1:8901
```

The version argument is resolved against the workload directory (`candidates/groot/ah_boundary_thor/`);
an absolute path also works.

A passing candidate is not yet deployable: it must also pass the pipeline's 997-frame task gate, which runs on the
Thor in groot-perf-diagnosis. Push the run's branch (or send the candidate directory) so it can be run there.
