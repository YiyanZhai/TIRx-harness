#!/usr/bin/env python3
"""GR00T N1.7 vl_self_attention + DiT cross-KV on Jetson Thor, against the deployed TRT engines.

Each workload row is one real frame (156 vision-language tokens). Timing, graph capture, the
kernel-node count and the shared GPU lock reuse the action-head boundary task's validated helpers:
alternating graph replays, each sample after a 128 MB L2-evicting read, memory clock warmed first.

The correctness gate sits at the noise floor of the 4 vl_self_attention blocks. TensorRT and an
independent Triton implementation differ from the oracle by a relative L2 error of 0.019-0.025 on
the keys and values (fp16 accumulation order and FP8 re-quantization flips, amplified over the
blocks), so each row allows 1.4x the larger of the two. Max abs errors are heavy-tailed (a change of
one dequantization scale in its last bits reaches 0.57), so the max abs limit is a flat 1.0. Errors
below that floor (an erf GELU, unquantized FFN activations) are not detectable here.
"""

from __future__ import annotations

import statistics

import torch

from ...benchmark_common import choose_device, config_from_env, summarize
from ..groot_ah_boundary_thor import benchmark as common
from . import baseline

TASK_NAME = "groot_vlsa_xkv_thor"
HARNESS_FILES = tuple(
    f"tasks/{path}"
    for path in (
        "groot_vlsa_xkv_thor/baseline.py",
        "groot_vlsa_xkv_thor/definition.json",
        "groot_ah_boundary_thor/__init__.py",
        "groot_ah_boundary_thor/benchmark.py",
        "groot_ah_boundary_thor/baseline.py",
    )
)

FRAMES = (0, 1, 30, 31)
OUTPUTS = baseline.OUTPUTS
# Per frame and output (keys, values): 1.4x the larger relative L2 error of TensorRT and the
# independent Triton implementation against the oracle (calibrated on the four frames).
MAX_REL = {
    "cross_keys": (0.034, 0.034, 0.030, 0.030),
    "cross_values": (0.036, 0.036, 0.032, 0.032),
}
MAX_ABS = 1.0  # legitimate implementations reach 0.23-0.34 (0.59 for near-floor variants); bugs 6+
PROBE_MAX_REL, PROBE_MAX_ABS = (
    0.10,
    2.0,
)  # after the in-place input change (TensorRT 0.022-0.026, stale 0.26+)


def default_config():
    return config_from_env(warmup=10, iters=60, trials=3)


def make_workloads(config=None):
    return [
        {
            "suite": "official",
            "id": f"frame{c}",
            "axes": {"frame": c, "tokens": 156, "blocks": baseline.BLOCKS},
            "raw": {"uuid": f"frame{c}"},
        }
        for c in FRAMES
    ]


def make_data(weights, golden, call, device):
    """The data dict for one frame: weights, a fresh input copy, the preallocated outputs."""

    data = dict(weights)
    data["hidden_states"] = golden[f"call{call}.hidden_states"].to(device).clone()
    shape = (baseline.CROSS_BLOCKS, 1, baseline.KV_HEADS, 156, baseline.KV_DIM)
    for name in OUTPUTS:
        data[name] = torch.empty(shape, dtype=torch.float16, device=device)
    return data


def tirx_prepare(solution_module, data):
    launch = solution_module.setup(data)
    if not callable(launch):
        raise TypeError("solution.setup(data) must return a callable")
    return launch


def tirx_run(launch):
    return launch()


def _poison(data):
    for name in OUTPUTS:
        data[name].fill_(float("nan"))


def _check(data, expected, max_rel, max_abs, label):
    """max_rel: per-output limits. Returns the worst (relative L2, max abs) over the outputs."""

    rel, worst = 0.0, 0.0
    for name, ref in zip(OUTPUTS, expected):
        out = data[name]
        if tuple(out.shape) != tuple(ref.shape) or out.dtype != ref.dtype:
            raise AssertionError(f"{label}: {name} is {tuple(out.shape)} {out.dtype}")
        a, e = out.double(), ref.double()
        if not bool(torch.isfinite(a).all()):
            raise AssertionError(f"{label}: {name} has non-finite values")
        diff = (a - e).abs()
        r, w = (diff.norm() / e.norm()).item(), diff.max().item()
        if r > max_rel[name] or w > max_abs:
            raise AssertionError(
                f"{label}: {name} relative L2 error {r:.4f} (limit {max_rel[name]}), "
                f"max abs {w:.4f} (limit {max_abs})"
            )
        rel, worst = max(rel, r), max(worst, w)
    return rel, worst


def _perturb(data, seed):
    t = data["hidden_states"]
    gen = torch.Generator(device=t.device).manual_seed(seed)
    noise = torch.randn(t.shape, generator=gen, device=t.device, dtype=torch.float32)
    t.copy_((t.float() + 0.25 * t.float().std() * noise).to(t.dtype))


def _run_row(k, weights, golden, device, cfg, candidate_fn, candidate_prepare_fn, flush):
    call = FRAMES[k]
    stream = torch.cuda.Stream(device)
    limits = ({name: MAX_REL[name][k] for name in OUTPUTS}, MAX_ABS)
    base_data = make_data(weights, golden, call, device)
    base = baseline.TrtVlsaXkv(base_data)
    expected = baseline.reference(base_data)
    _poison(base_data)
    common._call(base, stream)
    base_rel, base_abs = _check(base_data, expected, *limits, "TensorRT baseline")

    data = make_data(weights, golden, call, device)
    if candidate_fn is None:  # self-check: the baseline in both arms
        candidate = baseline.TrtVlsaXkv(data)
    else:
        launch = candidate_prepare_fn(data)

        def candidate():
            return candidate_fn(launch)

    snapshot = {name: data[name].clone() for name in (*baseline.INPUTS, *baseline.weight_names())}

    def unchanged(label):
        for name, value in snapshot.items():
            if not torch.equal(data[name], value):
                raise AssertionError(f"{label}: the candidate modified {name}")

    outputs, rel, worst = [], 0.0, 0.0
    for run in range(cfg.correctness_runs):
        _poison(data)
        common._call(candidate, stream)
        r, w = _check(data, expected, *limits, f"correctness run {run}")
        rel, worst = max(rel, r), max(worst, w)
        outputs.append([data[name].clone() for name in OUTPUTS])
    if cfg.require_repeatable_outputs and any(
        not all(torch.equal(a, b) for a, b in zip(o, outputs[0])) for o in outputs[1:]
    ):
        raise AssertionError("outputs differ between identical calls")
    unchanged("correctness runs")
    checks = cfg.correctness_runs

    if cfg.check_input_dependence:
        _perturb(data, 2000 + call)
        perturbed = baseline.reference(data)
        _poison(data)
        common._call(candidate, stream)
        _check(
            data,
            perturbed,
            {name: PROBE_MAX_REL for name in OUTPUTS},
            PROBE_MAX_ABS,
            "after an in-place change of hidden_states",
        )
        data["hidden_states"].copy_(snapshot["hidden_states"])
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
        _check(data, expected, *limits, "outputs left by timing")
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
    # One GPU on Thor; nvidia-smi reports no memory use there, so "auto" cannot rank devices.
    device = choose_device("cuda:0" if cfg.device == "auto" else cfg.device)
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) != (11, 0):
        raise RuntimeError(
            f"this task targets Jetson Thor (compute capability 11.0), not {major}.{minor}"
        )
    weights, golden = baseline.load_model(device)
    flush_src = torch.ones(common.FLUSH_BYTES // 4, dtype=torch.float32, device=device)
    flush_dst = torch.empty((), dtype=torch.float32, device=device)

    def flush():
        torch.sum(flush_src, dim=0, out=flush_dst)

    print(
        f"OP: {TASK_NAME} ({len(entries)} frames, "
        f"warmup/iters/trials {cfg.warmup}/{cfg.iters}/{cfg.trials})"
    )
    rows = []
    with common._gpu_lock():
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
