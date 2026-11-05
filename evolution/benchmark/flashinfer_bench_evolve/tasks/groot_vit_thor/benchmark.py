#!/usr/bin/env python3
"""GR00T N1.7 backbone ViT on Jetson Thor, against the deployed TensorRT engine.

Each workload row is one real frame of a LIBERO episode (two 256-patch images). Timing, graph
capture, the kernel-node count and the shared GPU lock reuse the action-head boundary task's
validated helpers: alternating graph replays, each sample after a 128 MB L2-evicting read, memory
clock warmed first.

The gate is the relative L2 error of each output against the oracle: image_embeds and the three
deepstack levels. The ViT is chaotic: FP8 activation quantization flips borderline roundings and 24
blocks amplify the flips, so TensorRT and our CUDA ViT sit 0.14-0.24 from the oracle on
image_embeds, 0.033-0.038 on deepstack 0, and 0.09-0.135 on deepstack 1 and 2 (golden and probe
frames). The limits are 1.4x the largest of those, the same on every frame. Errors at or below that
floor (activations left unquantized, erf GELU in the blocks) are not detectable here.
"""

from __future__ import annotations

import statistics

import torch

from ...benchmark_common import choose_device, config_from_env, summarize
from ..groot_ah_boundary_thor import benchmark as common
from . import baseline

TASK_NAME = "groot_vit_thor"
HARNESS_FILES = tuple(
    f"tasks/{path}"
    for path in (
        "groot_vit_thor/baseline.py",
        "groot_vit_thor/definition.json",
        "groot_ah_boundary_thor/__init__.py",
        "groot_ah_boundary_thor/benchmark.py",
        "groot_ah_boundary_thor/baseline.py",
    )
)

FRAMES = (0, 1, 30, 31)
# Relative L2 limits for image_embeds and deepstack levels 0, 1, 2: 1.4x the largest error of
# TensorRT and an independent CUDA implementation against the oracle, over the golden frames and
# perturbed copies of them (calibrated with groot-perf-diagnosis kda/tirx/m6_calib.py).
MAX_REL = (0.34, 0.055, 0.19, 0.185)
# The probe perturbs pixel_values in place (Gaussian noise, 0.25 of their std) and reuses the
# limits: TensorRT stays below them, an output computed before the change sits at 0.43-0.56 /
# 0.22-0.31 / 0.37-0.50 / 0.38-0.49.
PROBE_NOISE = 0.25
LABELS = ("image_embeds", "deepstack 0", "deepstack 1", "deepstack 2")


def default_config():
    return config_from_env(warmup=10, iters=60, trials=3)


def make_workloads(config=None):
    return [
        {
            "suite": "official",
            "id": f"frame{c}",
            "axes": {"frame": c, "patches": baseline.PATCHES, "blocks": baseline.BLOCKS},
            "raw": {"uuid": f"frame{c}"},
        }
        for c in FRAMES
    ]


def make_data(weights, golden, call, device):
    """The data dict for one frame: weights, a fresh input copy, the preallocated outputs."""

    data = dict(weights)
    data["pixel_values"] = golden[f"call{call}.pixel_values"].to(device).clone()
    data["image_embeds"] = torch.empty(
        baseline.MERGED, baseline.OUT_HIDDEN, dtype=torch.float16, device=device
    )
    data["deepstack_features"] = torch.empty(
        len(baseline.DEEPSTACK_AFTER),
        baseline.MERGED,
        baseline.OUT_HIDDEN,
        dtype=torch.float16,
        device=device,
    )
    return data


def tirx_prepare(solution_module, data):
    launch = solution_module.setup(data)
    if not callable(launch):
        raise TypeError("solution.setup(data) must return a callable")
    return launch


def tirx_run(launch):
    return launch()


def _poison(data):
    for name in baseline.OUTPUTS:
        data[name].fill_(float("nan"))


def _parts(data):
    ds = data["deepstack_features"]
    return (data["image_embeds"], *ds.unbind(0))


def _rel(out, ref):
    d = out.double() - ref.double()
    return (d.norm() / ref.double().norm()).item()


def _check(data, expected, label):
    for name, ref in zip(baseline.OUTPUTS, expected):
        out = data[name]
        if tuple(out.shape) != tuple(ref.shape) or out.dtype != ref.dtype:
            raise AssertionError(f"{label}: {name} is {tuple(out.shape)} {out.dtype}")
        if not bool(torch.isfinite(out).all()):
            raise AssertionError(f"{label}: {name} has non-finite values")
    errors = [_rel(o, r) for o, r in zip(_parts(data), (expected[0], *expected[1].unbind(0)))]
    bad = [
        f"{what} relative L2 {e:.4f} (limit {lim})"
        for what, e, lim in zip(LABELS, errors, MAX_REL)
        if e > lim
    ]
    if bad:
        raise AssertionError(f"{label}: " + "; ".join(bad))
    return errors


def _perturb(data, seed):
    t = data["pixel_values"]
    gen = torch.Generator(device=t.device).manual_seed(seed)
    noise = torch.randn(t.shape, generator=gen, device=t.device, dtype=torch.float32)
    t.copy_((t.float() + PROBE_NOISE * t.float().std() * noise).to(t.dtype))


def _run_row(k, weights, golden, device, cfg, candidate_fn, candidate_prepare_fn, flush):
    call = FRAMES[k]
    stream = torch.cuda.Stream(device)
    base_data = make_data(weights, golden, call, device)
    base = baseline.TrtVit(base_data)
    expected = baseline.reference(base_data)
    _poison(base_data)
    common._call(base, stream)
    base_errors = _check(base_data, expected, "TensorRT baseline")

    data = make_data(weights, golden, call, device)
    if candidate_fn is None:  # self-check: the baseline in both arms
        candidate = baseline.TrtVit(data)
    else:
        launch = candidate_prepare_fn(data)

        def candidate():
            return candidate_fn(launch)

    snapshot = {name: data[name].clone() for name in (*baseline.INPUTS, *baseline.weight_names())}

    def unchanged(label):
        for name, value in snapshot.items():
            if not torch.equal(data[name], value):
                raise AssertionError(f"{label}: the candidate modified {name}")

    outputs, errors = [], [0.0] * len(MAX_REL)
    for run in range(cfg.correctness_runs):
        _poison(data)
        common._call(candidate, stream)
        run_errors = _check(data, expected, f"correctness run {run}")
        errors = [max(a, b) for a, b in zip(errors, run_errors)]
        outputs.append([data[name].clone() for name in baseline.OUTPUTS])
    if cfg.require_repeatable_outputs and any(
        not all(torch.equal(a, b) for a, b in zip(o, outputs[0])) for o in outputs[1:]
    ):
        raise AssertionError("outputs differ between identical calls")
    unchanged("correctness runs")
    checks = cfg.correctness_runs

    if cfg.check_input_dependence:
        _perturb(data, 3000 + call)
        perturbed = baseline.reference(data)
        _poison(data)
        common._call(candidate, stream)
        _check(data, perturbed, "after an in-place change of pixel_values")
        data["pixel_values"].copy_(snapshot["pixel_values"])
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
        rel_l2=errors[0],
        deepstack_rel_l2=errors[1:],
        baseline_rel_l2=base_errors[0],
        baseline_deepstack_rel_l2=base_errors[1:],
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
    print("# Aggregate result (four frames of one episode)")
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
    # Locked from the first CUDA call: an unlocked model load overlapped others' timed runs.
    with common._gpu_lock():
        # One GPU on Thor; nvidia-smi reports no memory use there, so "auto" cannot rank devices.
        device = choose_device("cuda:0" if cfg.device == "auto" else cfg.device)
        major, minor = torch.cuda.get_device_capability(device)
        if (major, minor) != (11, 0):
            raise RuntimeError(
                f"this task targets Jetson Thor (compute capability 11.0), not {major}.{minor}"
            )
        weights, golden, calls = baseline.load_model(device)
        if tuple(calls) != FRAMES:
            raise RuntimeError(
                f"the model directory holds frames {calls}, the task expects {FRAMES}"
            )
        flush_src = torch.ones(common.FLUSH_BYTES // 4, dtype=torch.float32, device=device)
        flush_dst = torch.empty((), dtype=torch.float32, device=device)

        def flush():
            torch.sum(flush_src, dim=0, out=flush_dst)

        print(
            f"OP: {TASK_NAME} ({len(entries)} frames, "
            f"warmup/iters/trials {cfg.warmup}/{cfg.iters}/{cfg.trials})"
        )
        rows = []
        for entry in entries:
            k = FRAMES.index(int(entry["axes"]["frame"]))
            row = {"suite": entry["suite"], "id": entry["id"], "axes": entry["axes"]}
            try:
                row.update(
                    _run_row(
                        k, weights, golden, device, cfg, candidate_fn, candidate_prepare_fn, flush
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
        summarize(rows, "frame")
    return rows


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    run_suite()


if __name__ == "__main__":
    main()
