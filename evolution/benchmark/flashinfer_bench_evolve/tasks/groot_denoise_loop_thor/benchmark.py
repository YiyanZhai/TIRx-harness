#!/usr/bin/env python3
"""GR00T N1.7 action-head denoising loop on Jetson Thor, against the deployed TensorRT sequence.

Each workload row is one frame's whole loop (4 steps of action encoder, DiT, action decoder and
Euler update) on the backbone outputs of one real LIBERO frame. noise0-3 are the golden instruction
(156 VL tokens) from four initial noises: noise0 is the frame's own draw, noise1-3 are seeded draws.
len<L> are the same frame and noise0 under another instruction, so with L VL tokens: LIBERO-10's
tasks have 145-156, and only one of the ten has 156. The deployed runtime keeps one action-head graph
per token count, so setup runs once per count in one process, and a count set up earlier must keep
working after later ones; the suite checks that. Timing, graph capture, the kernel-node count and the
shared GPU lock reuse the action-head boundary task's validated helpers: alternating graph replays,
each sample after a 128 MB L2-evicting read, memory clock warmed first.

The gate checks both outputs against the oracle: the final actions, and every step's DiT output
(model_outputs). NVFP4 activation quantization in the DiT flips borderline roundings, so correct
implementations do not match the oracle exactly; the limits come from TensorRT and an independent
implementation (our CUDA DiT with a PyTorch encoder / decoder). The final actions barely depend on
the state token or on small changes of the frame's K/V, so the DiT outputs carry most of the
input-dependence check, and the probe swaps in another scene's K/V.
"""

from __future__ import annotations

import statistics
import time

import torch

from ...benchmark_common import choose_device, config_from_env, summarize
from ..groot_ah_boundary_thor import benchmark as common
from . import baseline

TASK_NAME = "groot_denoise_loop_thor"
HARNESS_FILES = tuple(
    f"tasks/{path}"
    for path in (
        "groot_denoise_loop_thor/baseline.py",
        "groot_denoise_loop_thor/definition.json",
        "groot_dit_step_thor/__init__.py",
        "groot_dit_step_thor/baseline.py",
        "groot_ah_boundary_thor/__init__.py",
        "groot_ah_boundary_thor/benchmark.py",
        "groot_ah_boundary_thor/baseline.py",
    )
)

NOISE_ROWS = ("noise0", "noise1", "noise2", "noise3")
LEN_ROWS = ("len140", "len145", "len149", "len150", "len151", "len152", "len155", "len172")
ROWS = (*NOISE_ROWS, *LEN_ROWS)
# A setup after the first row's may compile for its token count, but must not take this long.
SETUP_MAX_S = 60.0
# Relative L2 limits: about 1.4x the largest error of TensorRT and the independent implementation
# over the four rows and the probe (calibrated with groot-perf-diagnosis kda/tirx/m5_calib.py).
# They reach 0.0035 on the actions, 0.032, 0.039, 0.056, 0.070 on the four steps' DiT outputs, and
# 0.035 on the state token's row of any step; the mildest defects reach 0.021 on the actions.
# Max abs errors are heavy-tailed (TensorRT reached 0.113 on the actions after the probe, against
# 0.02-0.05 otherwise), so their limits are flat and only catch garbage elements.
ACTIONS_MAX_REL, ACTIONS_MAX_ABS = 0.006, 0.5
STEP_MAX_REL = (0.046, 0.055, 0.079, 0.098)
STEP_MAX_ABS = (1.0, 1.0, 1.0, 1.0)
# The state reaches the actions only through attention (zeroing it moves them by 0.005), so its
# use is checked on its own row of the DiT outputs: zeroed, that row is 0.09-0.13 off.
ROW0_MAX_REL = (0.05, 0.05, 0.05, 0.05)
# One step may exceed that up to ROW0_ONE_STEP_MAX_REL: on len150, TensorRT and the independent
# implementation both reach 0.063-0.064 on step 2 and stay near 0.02 elsewhere. A state read once in
# setup is over 0.05 on at least two steps on 11 of the 12 rows, zeroed state on all rows.
ROW0_ONE_STEP_MAX_REL = 0.09
# The probe adds noise to initial_actions (0.25 of their std) and state_features (2.0 of its std),
# and swaps in another scene's cross K/V. Outputs computed before the change sit at 0.27 (actions);
# ignoring the new noise 0.24 (actions), the new K/V 0.12-0.13 (actions) and 0.36-0.59 (steps),
# the new state 0.053-0.106 (state rows; at least three steps over the limit on every row).
PROBE_SCALE = {"initial_actions": 0.25, "state_features": 2.0}


def default_config():
    return config_from_env(warmup=5, iters=40, trials=3)


def make_workloads(config=None):
    return [
        {
            "suite": "official",
            "id": row,
            "axes": {
                "row": row,
                "vl_tokens": baseline.vl_tokens(row),
                "steps": baseline.STEPS,
                "tokens": 41,
            },
            "raw": {"uuid": row},
        }
        for row in ROWS
    ]


def make_data(weights, golden, row, device, block_types):
    """The data dict for one row: weights, fresh input and constant copies, the output buffers."""

    data = dict(weights)
    data["block_types"] = list(block_types)
    for name in (*baseline.INPUTS, *baseline.CONSTANTS):
        data[name] = baseline.row_tensor(golden, row, name).to(device).clone()
    data["actions_out"] = torch.empty_like(data["initial_actions"])
    data["model_outputs"] = torch.empty(
        baseline.STEPS, 1, 41, 1024, dtype=torch.float16, device=device
    )
    data["dt"] = baseline.DT
    data["embodiment"] = baseline.EMBODIMENT
    return data


def tirx_prepare(solution_module, data):
    launch = solution_module.setup(data)
    if not callable(launch):
        raise TypeError("solution.setup(data) must return a callable")
    return launch


def tirx_run(launch):
    return launch()


def _err(out, ref):
    d = out.double() - ref.double()
    return (d.norm() / ref.double().norm()).item(), d.abs().max().item()


def _check(data, expected, label):
    """(actions relative L2, actions max abs, worst step relative L2 / its limit)."""

    for name, ref in zip(baseline.OUTPUTS, expected):
        out = data[name]
        if tuple(out.shape) != tuple(ref.shape) or out.dtype != ref.dtype:
            raise AssertionError(f"{label}: {name} is {tuple(out.shape)} {out.dtype}")
        if not bool(torch.isfinite(out).all()):
            raise AssertionError(f"{label}: {name} has non-finite values")
    bad = []
    rel, worst = _err(data["actions_out"], expected[0])
    if rel > ACTIONS_MAX_REL or worst > ACTIONS_MAX_ABS:
        bad.append(
            f"actions_out relative L2 {rel:.4f} (limit {ACTIONS_MAX_REL}), max abs {worst:.4f} "
            f"(limit {ACTIONS_MAX_ABS})"
        )
    step_ratio = 0.0
    row0 = [
        _err(data["model_outputs"][k][:, :1], expected[1][k][:, :1])[0]
        for k in range(baseline.STEPS)
    ]
    row0_over = sum(r0 > limit for r0, limit in zip(row0, ROW0_MAX_REL))
    for k in range(baseline.STEPS):
        out, ref = data["model_outputs"][k], expected[1][k]
        r, w = _err(out, ref)
        r0 = row0[k]
        step_ratio = max(step_ratio, r / STEP_MAX_REL[k])
        row0_bad = r0 > ROW0_ONE_STEP_MAX_REL or (r0 > ROW0_MAX_REL[k] and row0_over > 1)
        if r > STEP_MAX_REL[k] or w > STEP_MAX_ABS[k] or row0_bad:
            bad.append(
                f"model_outputs[{k}] relative L2 {r:.4f} (limit {STEP_MAX_REL[k]}), max abs "
                f"{w:.4f} (limit {STEP_MAX_ABS[k]}), state row {r0:.4f} (limit {ROW0_MAX_REL[k]})"
            )
    if bad:
        raise AssertionError(f"{label}: " + "; ".join(bad))
    return rel, worst, step_ratio


def _poison(data):
    for name in baseline.OUTPUTS:
        data[name].fill_(float("nan"))


def _perturb(data, golden, row, seed):
    """New noise and state values, and another scene's cross K/V (frame 30 of the episode)."""

    for i, (name, scale) in enumerate(PROBE_SCALE.items()):
        t = data[name]
        gen = torch.Generator(device=t.device).manual_seed(seed + i)
        noise = torch.randn(t.shape, generator=gen, device=t.device, dtype=torch.float32)
        t.copy_((t.float() + scale * t.float().std() * noise).to(t.dtype))
    for name in ("cross_keys", "cross_values"):
        data[name].copy_(baseline.row_tensor(golden, row, f"probe.{name}"))


def _run_row(
    k, weights, golden, block_types, device, cfg, candidate_fn, candidate_prepare_fn, flush, first
):
    row = ROWS[k]
    tokens = baseline.vl_tokens(row)
    stream = torch.cuda.Stream(device)
    base_data = make_data(weights, golden, row, device, block_types)
    base = baseline.TrtLoop(base_data)
    expected = baseline.reference(base_data, block_types)
    _poison(base_data)
    common._call(base, stream)
    base_rel, base_abs, base_steps = _check(base_data, expected, "TensorRT baseline")

    data = make_data(weights, golden, row, device, block_types)
    setup_s = 0.0
    if candidate_fn is None:  # self-check: the baseline in both arms
        candidate = baseline.TrtLoop(data)
    else:
        started = time.perf_counter()
        try:
            launch = candidate_prepare_fn(data)
        except Exception as exc:
            raise AssertionError(
                f"setup failed at {tokens} VL tokens: {type(exc).__name__}: {exc}"
            ) from exc
        setup_s = time.perf_counter() - started
        if first and setup_s > SETUP_MAX_S:
            raise AssertionError(
                f"setup at {tokens} VL tokens took {setup_s:.0f} s (limit {SETUP_MAX_S:.0f} s after "
                "the first row's)"
            )

        def candidate():
            return candidate_fn(launch)

    names = (*baseline.INPUTS, *baseline.CONSTANTS, *baseline.weight_names(block_types))
    snapshot = {name: data[name].clone() for name in names}

    def unchanged(label):
        for name, value in snapshot.items():
            if not torch.equal(data[name], value):
                raise AssertionError(f"{label}: the candidate modified {name}")

    outputs, rel, worst, steps = [], 0.0, 0.0, 0.0
    for run in range(cfg.correctness_runs):
        _poison(data)
        common._call(candidate, stream)
        r, w, sr = _check(data, expected, f"correctness run {run}")
        rel, worst, steps = max(rel, r), max(worst, w), max(steps, sr)
        outputs.append([data[name].clone() for name in baseline.OUTPUTS])
    if cfg.require_repeatable_outputs and any(
        not all(torch.equal(a, b) for a, b in zip(o, outputs[0])) for o in outputs[1:]
    ):
        raise AssertionError("outputs differ between identical calls")
    unchanged("correctness runs")
    checks = cfg.correctness_runs

    if cfg.check_input_dependence:
        _perturb(data, golden, row, 5000 + k)
        perturbed = baseline.reference(data, block_types)
        _poison(data)
        common._call(candidate, stream)
        _check(
            data,
            perturbed,
            "after an in-place change of initial_actions, state_features and cross K/V",
        )
        for name in baseline.INPUTS:
            data[name].copy_(snapshot[name])
        checks += 1

    data["embodiment_id"].fill_(baseline.EMBODIMENT + 1)
    data["actions_out"].zero_()
    common._call(candidate, stream)
    data["embodiment_id"].fill_(baseline.EMBODIMENT)
    if not bool(torch.isnan(data["actions_out"]).all()):
        raise AssertionError(
            "guard probe: embodiment guard (NaN actions for a foreign embodiment_id) not honored"
        )
    checks += 1

    if first and first["tokens"] != tokens:  # the setup of the first count still works after this one
        _poison(first["data"])
        common._call(first["candidate"], first["stream"])
        _check(
            first["data"],
            first["expected"],
            f"the {first['tokens']}-token setup called again after the {tokens}-token one",
        )
        checks += 1
    if not first:
        first.update(
            tokens=tokens, data=data, expected=expected, candidate=candidate, stream=stream
        )

    graphs = {}  # destroyed before the engines and modules whose kernels they captured
    try:
        try:
            graphs["baseline"], base_kernels = common._graph(base, stream)
            graphs["candidate"], cand_kernels = common._graph(candidate, stream)
        except RuntimeError as exc:
            raise AssertionError(f"CUDA graph capture failed: {exc}") from exc
        if candidate_fn is not None and cand_kernels != 1:
            raise AssertionError(
                f"one call launched {cand_kernels} kernels; the contract is one persistent kernel"
            )
        samples = common._time(graphs, cfg, flush)
    finally:
        graphs.clear()
        torch.cuda.synchronize()
    if cfg.check_after_timing:
        _check(data, expected, "outputs left by timing")
        checks += 1
    unchanged("timing")
    baseline_ms = statistics.median(samples["baseline"])
    kernel_ms = statistics.median(samples["candidate"])
    return dict(
        passed=True,
        verdict="PASS",
        note="",
        correctness_checks=checks,
        rel_l2=rel,
        max_abs_err=worst,
        step_rel_vs_limit=steps,
        baseline_rel_l2=base_rel,
        baseline_max_abs_err=base_abs,
        baseline_step_rel_vs_limit=base_steps,
        matched=1.0,
        baseline_ms=baseline_ms,
        kernel_ms=kernel_ms,
        speedup=baseline_ms / kernel_ms,
        vl_tokens=tokens,
        setup_s=setup_s,
        baseline_mean_ms=statistics.mean(samples["baseline"]),
        kernel_mean_ms=statistics.mean(samples["candidate"]),
        baseline_p95_ms=common._p95(samples["baseline"]),
        kernel_p95_ms=common._p95(samples["candidate"]),
        samples=len(samples["candidate"]),
        baseline_kernels=base_kernels,
        kernel_launches=cand_kernels,
        emc_mhz=common._mhz(f"{common.DEVFREQ}/bwmgr/cur_freq"),
        gpc_mhz=common._mhz(f"{common.DEVFREQ}/gpu-gpc-0/cur_freq"),
        timed=True,
        timer="cuda_event",
        timing_mode="graph_replay_cold_l2",
    )


def _print_aggregate(rows):
    if not all(row["passed"] for row in rows):
        print("\n# Aggregate result\n| candidate | FAIL | no aggregate score |", flush=True)
        return
    total_baseline = sum(row["baseline_ms"] for row in rows)
    total_kernel = sum(row["kernel_ms"] for row in rows)
    print()
    print("# Aggregate result (one frame: four initial noises, eight other instructions)")
    print(
        f"| candidate ({len(rows)}/{len(rows)} PASS) | {total_kernel * 1000:.1f}us | "
        f"TensorRT {total_baseline * 1000:.1f}us | {total_baseline / total_kernel:.3f}x |",
        flush=True,
    )
    print("| VL tokens | candidate median / mean / p95 ms | TensorRT median ms | speedup | setup s |")
    for row in rows:
        print(
            f"| {row['vl_tokens']} | {row['kernel_ms']:.3f} / {row['kernel_mean_ms']:.3f} / "
            f"{row['kernel_p95_ms']:.3f} | {row['baseline_ms']:.3f} | {row['speedup']:.3f}x | "
            f"{row['setup_s']:.1f} |",
            flush=True,
        )


@torch.no_grad()
def run_suite(config=None, candidate_fn=None, workloads=None, candidate_prepare_fn=None):
    cfg = config or default_config()
    entries = make_workloads(cfg) if workloads is None else workloads
    if cfg.warmup < 0 or cfg.iters < 1 or cfg.trials < 1 or cfg.correctness_runs < 1:
        raise ValueError("warmup must be nonnegative; repeat, trials and correctness runs positive")
    # Locked from the first CUDA call: an unlocked model load overlapped others' timed runs.
    with common._gpu_lock():
        # One GPU on Thor; nvidia-smi reports no memory use there, so "auto" cannot rank devices.
        device = choose_device("cuda:0" if cfg.device == "auto" else cfg.device)
        major, minor = torch.cuda.get_device_capability(device)
        if (major, minor) != (11, 0):
            raise RuntimeError(
                f"this task targets Jetson Thor (compute capability 11.0), not {major}.{minor}"
            )
        weights, golden, block_types, rows = baseline.load_model(device)
        if tuple(rows) != ROWS:
            raise RuntimeError(f"the model directory holds rows {rows}, the task expects {ROWS}")
        flush_src = torch.ones(common.FLUSH_BYTES // 4, dtype=torch.float32, device=device)
        flush_dst = torch.empty((), dtype=torch.float32, device=device)

        def flush():
            torch.sum(flush_src, dim=0, out=flush_dst)

        print(
            f"OP: {TASK_NAME} ({len(entries)} rows, "
            f"warmup/iters/trials {cfg.warmup}/{cfg.iters}/{cfg.trials})"
        )
        results = []
        first = {}  # the first row's setup, kept to check it after the setups of other token counts
        for entry in entries:
            k = ROWS.index(entry["axes"]["row"])
            result = {"suite": entry["suite"], "id": entry["id"], "axes": entry["axes"]}
            try:
                result.update(
                    _run_row(
                        k,
                        weights,
                        golden,
                        block_types,
                        device,
                        cfg,
                        candidate_fn,
                        candidate_prepare_fn,
                        flush,
                        first,
                    )
                )
            except AssertionError as exc:
                result.update(
                    passed=False,
                    verdict="FAIL",
                    note=str(exc),
                    speedup=None,
                    baseline_ms=None,
                    kernel_ms=None,
                    timed=True,
                    timer="cuda_event",
                )
            results.append(result)
            print(result, flush=True)
            torch.cuda.empty_cache()
        first.clear()
    _print_aggregate(results)
    if all(r["passed"] for r in results):
        summarize(results, "row")
    return results


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    run_suite()


if __name__ == "__main__":
    main()
