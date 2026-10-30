#!/usr/bin/env python3
"""GR00T N1.7 DiT denoising step on Jetson Thor, against the deployed TensorRT engine.

Each workload row is one of the four denoising steps of a real frame. Timing, graph capture,
the kernel-node count and the shared GPU lock reuse the action-head boundary task's validated
helpers: alternating graph replays, each sample after a 128 MB L2-evicting read, memory clock
warmed first.

The correctness gate sits at the NVFP4 noise floor. TensorRT and an independent Triton
implementation differ from the oracle by a relative L2 error of 0.026-0.049 per step (dynamic
NVFP4 activation quantization flips borderline roundings), so each row allows 1.4x the larger of
the two. Errors below that floor (a missing bias in one block, a wrong LayerNorm epsilon) are not
detectable here.
"""

from __future__ import annotations

import statistics

import torch

from ...benchmark_common import choose_device, config_from_env, summarize
from ..groot_ah_boundary_thor import benchmark as common
from . import baseline

TASK_NAME = "groot_dit_step_thor"
HARNESS_FILES = tuple(
    f"tasks/{path}"
    for path in (
        "groot_dit_step_thor/baseline.py",
        "groot_dit_step_thor/definition.json",
        "groot_ah_boundary_thor/__init__.py",
        "groot_ah_boundary_thor/benchmark.py",
        "groot_ah_boundary_thor/baseline.py",
    )
)

# Per step: 1.4x the larger relative L2 error, and 2x the larger max abs error, of TensorRT and the
# independent Triton implementation against the oracle (calibrated on the golden frame).
MAX_REL = (0.039, 0.047, 0.054, 0.069)
MAX_ABS = (0.16, 0.17, 0.16, 0.23)
PROBE_MAX_REL, PROBE_MAX_ABS = 0.10, 0.35  # after the in-place input change (detects stale outputs)
MUTABLE = ("sa_embs", "cross_keys", "cross_values")  # new values every call in deployment


def default_config():
    return config_from_env(warmup=10, iters=60, trials=3)


def make_workloads(config=None):
    return [
        {
            "suite": "official",
            "id": f"step{k}",
            "axes": {"step": k, "tokens": 41, "blocks": 32},
            "raw": {"uuid": f"step{k}"},
        }
        for k in range(4)
    ]


def make_data(weights, golden, block_types, k, device):
    """The data dict for step k: weights, fresh input copies, the preallocated output."""

    data = dict(weights)
    for name in baseline.INPUTS:
        key = name if name in golden else f"step{k}.{name}"
        data[name] = golden[key].to(device).clone()
    data["output"] = torch.empty(1, 41, 1024, dtype=torch.float16, device=device)
    data["block_types"] = list(block_types)
    return data


def tirx_prepare(solution_module, data):
    launch = solution_module.setup(data)
    if not callable(launch):
        raise TypeError("solution.setup(data) must return a callable")
    return launch


def tirx_run(launch):
    return launch()


def _check(out, expected, max_rel, max_abs, label):
    if tuple(out.shape) != tuple(expected.shape) or out.dtype != expected.dtype:
        raise AssertionError(f"{label}: output is {tuple(out.shape)} {out.dtype}")
    a, e = out.double(), expected.double()
    if not bool(torch.isfinite(a).all()):
        raise AssertionError(f"{label}: output has non-finite values")
    diff = (a - e).abs()
    rel = (diff.norm() / e.norm()).item()
    worst = diff.max().item()
    if rel > max_rel or worst > max_abs:
        raise AssertionError(
            f"{label}: relative L2 error {rel:.4f} (limit {max_rel}), max abs {worst:.4f} "
            f"(limit {max_abs})"
        )
    return rel, worst


def _perturb(data, seed):
    for i, name in enumerate(MUTABLE):
        t = data[name]
        gen = torch.Generator(device=t.device).manual_seed(seed + i)
        noise = torch.randn(t.shape, generator=gen, device=t.device, dtype=torch.float32)
        t.copy_((t.float() + 0.25 * t.float().std() * noise).to(t.dtype))
    data["image_mask"][..., :8] = ~data["image_mask"][..., :8]  # both masks keep valid keys


def _run_row(
    k, weights, golden, block_types, device, cfg, candidate_fn, candidate_prepare_fn, flush
):
    stream = torch.cuda.Stream(device)
    limits = (MAX_REL[k], MAX_ABS[k])
    base_data = make_data(weights, golden, block_types, k, device)
    base = baseline.TrtDitStep(base_data)
    expected = baseline.reference(base_data, block_types)
    base_data["output"].fill_(float("nan"))
    common._call(base, stream)
    base_rel, base_abs = _check(base_data["output"], expected, *limits, "TensorRT baseline")

    data = make_data(weights, golden, block_types, k, device)
    if candidate_fn is None:  # self-check: the baseline in both arms
        candidate = baseline.TrtDitStep(data)
    else:
        launch = candidate_prepare_fn(data)

        def candidate():
            return candidate_fn(launch)

    names = (*baseline.INPUTS, *baseline.weight_names(block_types))
    snapshot = {name: data[name].clone() for name in names}

    def unchanged(label):
        for name, value in snapshot.items():
            if not torch.equal(data[name], value):
                raise AssertionError(f"{label}: the candidate modified {name}")

    outputs, rel, worst = [], 0.0, 0.0
    for run in range(cfg.correctness_runs):
        data["output"].fill_(float("nan"))
        common._call(candidate, stream)
        r, w = _check(data["output"], expected, *limits, f"correctness run {run}")
        rel, worst = max(rel, r), max(worst, w)
        outputs.append(data["output"].clone())
    if cfg.require_repeatable_outputs and any(not torch.equal(o, outputs[0]) for o in outputs[1:]):
        raise AssertionError("outputs differ between identical calls")
    unchanged("correctness runs")
    checks = cfg.correctness_runs

    if cfg.check_input_dependence:
        _perturb(data, 2000 + k)
        perturbed = baseline.reference(data, block_types)
        data["output"].fill_(float("nan"))
        common._call(candidate, stream)
        _check(
            data["output"],
            perturbed,
            PROBE_MAX_REL,
            PROBE_MAX_ABS,
            "after an in-place change of sa_embs, cross_keys, cross_values, image_mask",
        )
        for name in (*MUTABLE, "image_mask"):
            data[name].copy_(snapshot[name])
        checks += 1

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
        _check(data["output"], expected, *limits, "output left by timing")
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
        max_abs=worst,
        baseline_rel_l2=base_rel,
        baseline_max_abs=base_abs,
        matched=1.0,
        baseline_ms=baseline_ms,
        kernel_ms=kernel_ms,
        speedup=baseline_ms / kernel_ms,
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
    print("# Aggregate result (the four denoising steps of one frame)")
    print(
        f"| candidate ({len(rows)}/{len(rows)} PASS) | {total_kernel * 1000:.1f}us | "
        f"TensorRT {total_baseline * 1000:.1f}us | {total_baseline / total_kernel:.3f}x |",
        flush=True,
    )


@torch.no_grad()
def run_suite(config=None, candidate_fn=None, workloads=None, candidate_prepare_fn=None):
    cfg = config or default_config()
    entries = make_workloads(cfg) if workloads is None else workloads
    if cfg.warmup < 0 or cfg.iters < 1 or cfg.trials < 1 or cfg.correctness_runs < 1:
        raise ValueError("warmup must be nonnegative; repeat, trials and correctness runs positive")
    # One GPU on Thor; nvidia-smi reports no memory use there, so "auto" cannot rank devices.
    device = choose_device("cuda:0" if cfg.device == "auto" else cfg.device)
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) != (11, 0):
        raise RuntimeError(
            f"this task targets Jetson Thor (compute capability 11.0), not {major}.{minor}"
        )
    weights, golden, block_types = baseline.load_model(device)
    flush_src = torch.ones(common.FLUSH_BYTES // 4, dtype=torch.float32, device=device)
    flush_dst = torch.empty((), dtype=torch.float32, device=device)

    def flush():
        torch.sum(flush_src, dim=0, out=flush_dst)

    print(
        f"OP: {TASK_NAME} ({len(entries)} steps, "
        f"warmup/iters/trials {cfg.warmup}/{cfg.iters}/{cfg.trials})"
    )
    rows = []
    with common._gpu_lock():
        for entry in entries:
            k = int(entry["axes"]["step"])
            row = {"suite": entry["suite"], "id": entry["id"], "axes": entry["axes"]}
            try:
                row.update(
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
                    )
                )
            except AssertionError as exc:
                row.update(
                    passed=False,
                    verdict="FAIL",
                    note=str(exc),
                    speedup=None,
                    baseline_ms=None,
                    kernel_ms=None,
                    timed=True,
                    timer="cuda_event",
                )
            rows.append(row)
            print(row, flush=True)
            torch.cuda.empty_cache()
    _print_aggregate(rows)
    if all(row["passed"] for row in rows):
        summarize(rows, "step")
    return rows


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    run_suite()


if __name__ == "__main__":
    main()
