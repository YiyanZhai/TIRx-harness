#!/usr/bin/env python3
"""GR00T N1.7 backbone LLM prefill on Jetson Thor, against the deployed TensorRT engine.

Each workload row is one real frame of a LIBERO episode (156 tokens). Timing, graph capture, the
kernel-node count and the shared GPU lock reuse the action-head boundary task's validated helpers:
alternating graph replays, each sample after a 128 MB L2-evicting read, memory clock warmed first.

The gate measures error per token. Token 0 (the attention sink) carries values up to ~15000 (rms
~400, against ~4 for the other tokens), so a whole-tensor relative L2 measures that one token's
error. The per-token relative L2, averaged over tokens, is the primary check. TensorRT and an
independent Triton implementation sit 0.12-0.13 from the oracle on it: dynamic NVFP4 activation
quantization flips borderline roundings, and 16 layers amplify the flips. Each row allows 1.4x the
larger of the two, plus 1.4x their whole-tensor error (the sink token). Errors at or below that
floor (activations left unquantized) are not detectable here.
"""

from __future__ import annotations

import statistics

import torch

from ...benchmark_common import choose_device, config_from_env, summarize
from ..groot_ah_boundary_thor import benchmark as common
from . import baseline

TASK_NAME = "groot_llm_prefill_thor"
HARNESS_FILES = tuple(
    f"tasks/{path}"
    for path in (
        "groot_llm_prefill_thor/baseline.py",
        "groot_llm_prefill_thor/definition.json",
        "groot_ah_boundary_thor/__init__.py",
        "groot_ah_boundary_thor/benchmark.py",
        "groot_ah_boundary_thor/baseline.py",
    )
)

FRAMES = (0, 1, 30, 31)
# Per frame: 1.4x the larger per-token mean relative L2, and 1.4x the larger whole-tensor relative
# L2, of TensorRT and the independent Triton implementation against the oracle (calibrated on the
# golden frames).
MAX_TOK_MEAN = (0.180, 0.183, 0.178, 0.187)
MAX_REL = (0.024, 0.024, 0.023, 0.025)
MAX_TOK = 0.9  # any single token (legitimate implementations reach 0.30-0.50)
# Probes, per-token mean limits. New inputs_embeds / deepstack values: TensorRT 0.16-0.22, a stale
# output 0.47-0.55. New position_ids / visual_pos_masks: TensorRT 0.11, stale rope tables 0.29-0.31,
# stale deepstack rows 0.37, both stale 0.42.
VALUE_PROBE_TOK_MEAN = 0.35
LAYOUT_PROBE_TOK_MEAN = 0.20
VALUES = ("inputs_embeds", "deepstack_0", "deepstack_1", "deepstack_2")  # new values every call


def default_config():
    return config_from_env(warmup=10, iters=60, trials=3)


def make_workloads(config=None):
    return [
        {
            "suite": "official",
            "id": f"frame{c}",
            "axes": {"frame": c, "tokens": 156, "layers": baseline.LAYERS},
            "raw": {"uuid": f"frame{c}"},
        }
        for c in FRAMES
    ]


def make_data(weights, golden, call, device):
    """The data dict for one frame: weights, fresh input copies, the preallocated output."""

    data = dict(weights)
    for name in baseline.INPUTS:
        data[name] = golden[f"call{call}.{name}"].to(device).clone()
    data["embeddings"] = torch.empty(1, 156, baseline.HIDDEN, dtype=torch.float16, device=device)
    return data


def tirx_prepare(solution_module, data):
    launch = solution_module.setup(data)
    if not callable(launch):
        raise TypeError("solution.setup(data) must return a callable")
    return launch


def tirx_run(launch):
    return launch()


def _errors(out, expected):
    a = out.double().reshape(-1, out.shape[-1])
    e = expected.double().reshape(-1, expected.shape[-1])
    d = a - e
    tok = d.norm(dim=-1) / e.norm(dim=-1)
    return tok.mean().item(), tok.max().item(), int(tok.argmax()), (d.norm() / e.norm()).item()


def _check(out, expected, max_tok_mean, max_rel, max_tok, label):
    if tuple(out.shape) != tuple(expected.shape) or out.dtype != expected.dtype:
        raise AssertionError(f"{label}: output is {tuple(out.shape)} {out.dtype}")
    if not bool(torch.isfinite(out).all()):
        raise AssertionError(f"{label}: output has non-finite values")
    tok_mean, tok_max, worst, rel = _errors(out, expected)
    if tok_mean > max_tok_mean or rel > max_rel or tok_max > max_tok:
        raise AssertionError(
            f"{label}: per-token relative L2 mean {tok_mean:.4f} (limit {max_tok_mean}), max "
            f"{tok_max:.4f} at token {worst} (limit {max_tok}); whole-tensor relative L2 {rel:.4f} "
            f"(limit {max_rel})"
        )
    return tok_mean, tok_max, rel


def _perturb_values(data, seed):
    for i, name in enumerate(VALUES):
        t = data[name]
        gen = torch.Generator(device=t.device).manual_seed(seed + i)
        noise = torch.randn(t.shape, generator=gen, device=t.device, dtype=torch.float32)
        t.copy_((t.float() + 0.25 * t.float().std() * noise).to(t.dtype))


def _perturb_layout(data):
    # Positions permuted across tokens (a uniform shift would leave rope's relative distances
    # unchanged), and the deepstack rows moved two tokens later.
    pos = data["position_ids"]
    perm = torch.randperm(pos.shape[-1], generator=torch.Generator().manual_seed(7))
    pos.copy_(pos[:, :, perm.to(pos.device)].clone())
    data["visual_pos_masks"].copy_(torch.roll(data["visual_pos_masks"], 2, dims=1))


def _run_row(k, weights, golden, device, cfg, candidate_fn, candidate_prepare_fn, flush):
    call = FRAMES[k]
    stream = torch.cuda.Stream(device)
    limits = (MAX_TOK_MEAN[k], MAX_REL[k], MAX_TOK)
    base_data = make_data(weights, golden, call, device)
    base = baseline.TrtLlmPrefill(base_data)
    expected = baseline.reference(base_data)
    base_data["embeddings"].fill_(float("nan"))
    common._call(base, stream)
    base_tok, _, base_rel = _check(base_data["embeddings"], expected, *limits, "TensorRT baseline")

    data = make_data(weights, golden, call, device)
    if candidate_fn is None:  # self-check: the baseline in both arms
        candidate = baseline.TrtLlmPrefill(data)
    else:
        launch = candidate_prepare_fn(data)

        def candidate():
            return candidate_fn(launch)

    snapshot = {name: data[name].clone() for name in (*baseline.INPUTS, *baseline.weight_names())}

    def unchanged(label):
        for name, value in snapshot.items():
            if not torch.equal(data[name], value):
                raise AssertionError(f"{label}: the candidate modified {name}")

    def restore():
        for name in baseline.INPUTS:
            data[name].copy_(snapshot[name])

    outputs, tok_mean, tok_max, rel = [], 0.0, 0.0, 0.0
    for run in range(cfg.correctness_runs):
        data["embeddings"].fill_(float("nan"))
        common._call(candidate, stream)
        m, w, r = _check(data["embeddings"], expected, *limits, f"correctness run {run}")
        tok_mean, tok_max, rel = max(tok_mean, m), max(tok_max, w), max(rel, r)
        outputs.append(data["embeddings"].clone())
    if cfg.require_repeatable_outputs and any(not torch.equal(o, outputs[0]) for o in outputs[1:]):
        raise AssertionError("outputs differ between identical calls")
    unchanged("correctness runs")
    checks = cfg.correctness_runs

    if cfg.check_input_dependence:
        for perturb, limit, what in (
            (
                lambda d: _perturb_values(d, 2000 + call),
                VALUE_PROBE_TOK_MEAN,
                "inputs_embeds and deepstack",
            ),
            (_perturb_layout, LAYOUT_PROBE_TOK_MEAN, "position_ids and visual_pos_masks"),
        ):
            perturb(data)
            perturbed = baseline.reference(data)
            data["embeddings"].fill_(float("nan"))
            common._call(candidate, stream)
            _check(
                data["embeddings"],
                perturbed,
                limit,
                1.0,
                2.0,
                f"after an in-place change of {what}",
            )
            restore()
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
        _check(data["embeddings"], expected, *limits, "output left by timing")
        checks += 1
    unchanged("timing")
    baseline_ms = statistics.median(samples["baseline"])
    kernel_ms = statistics.median(samples["candidate"])
    return dict(
        passed=True,
        verdict="PASS",
        note="",
        correctness_checks=checks,
        tok_rel_mean=tok_mean,
        tok_rel_max=tok_max,
        rel_l2=rel,
        baseline_tok_rel_mean=base_tok,
        baseline_rel_l2=base_rel,
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
    weights, golden, calls = baseline.load_model(device)
    if tuple(calls) != FRAMES:
        raise RuntimeError(f"the model directory holds frames {calls}, the task expects {FRAMES}")
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
