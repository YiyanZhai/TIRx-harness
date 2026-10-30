#!/usr/bin/env python3
"""GR00T N1.7 action-head step boundary on Jetson Thor, against the deployed TensorRT engines.

Each workload row is one of the three boundaries between the four DiT denoising steps of a real
frame. The baseline and the candidate are each captured in one CUDA graph and timed by graph
replay, alternating sample by sample, each sample after a 128 MB read that evicts L2 (in the
pipeline a DiT step that streams ~470 MB precedes every boundary). The suite runs under the
machine's shared GPU timing lock when it exists.
"""

from __future__ import annotations

import fcntl
import os
import statistics
import threading
import time
from contextlib import contextmanager

import torch

from ...benchmark_common import choose_device, config_from_env, summarize
from . import baseline

TASK_NAME = "groot_ah_boundary_thor"

ATOL = 0.02
RTOL = 0.01
MAX_REL_RMS = 2e-3
FLUSH_BYTES = 128 << 20  # > 4x the 32 MB L2
INPUTS = (
    "x",
    "actions_in",
    "timestep_embedding",
    "state_features",
    "position_embedding",
    "embodiment_id",
)
MUTABLE = ("x", "actions_in", "state_features")  # new values every call in deployment
OUTPUTS = ("actions_out", "sa_embs")
DEVFREQ = "/sys/class/devfreq"


def default_config():
    return config_from_env(warmup=20, iters=100, trials=3)


def make_workloads(config=None):
    return [
        {
            "suite": "official",
            "id": f"boundary{k}",
            "axes": {"boundary": k, "tokens": 41, "horizon": 40},
            "raw": {"uuid": f"boundary{k}"},
        }
        for k in range(3)
    ]


def make_data(weights, golden, k, device, alias=False):
    """The data dict for boundary k: weights, fresh input copies, preallocated outputs."""

    data = dict(weights)
    for name in INPUTS:
        key = f"b{k}.{name}" if f"b{k}.{name}" in golden else name
        data[name] = golden[key].to(device).clone()
    data["actions_out"] = data["actions_in"] if alias else torch.empty_like(data["actions_in"])
    data["sa_embs"] = torch.empty(1, 41, 1536, dtype=torch.float16, device=device)
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


def _poison(data):
    for name in OUTPUTS:
        data[name].fill_(float("nan"))


def _call(fn, stream):
    stream.wait_stream(
        torch.cuda.current_stream()
    )  # poisoning and input changes run on the current stream
    with torch.cuda.stream(stream):
        fn()
    stream.synchronize()


def _check(data, expected, label):
    worst = 0.0
    for name, ref in zip(OUTPUTS, expected):
        out = data[name]
        if tuple(out.shape) != tuple(ref.shape) or out.dtype != ref.dtype:
            raise AssertionError(f"{label}: {name} is {tuple(out.shape)} {out.dtype}")
        a, e = out.double(), ref.double()
        if not bool(torch.isfinite(a).all()):
            raise AssertionError(f"{label}: {name} has non-finite values")
        diff = (a - e).abs()
        bad = diff > ATOL + RTOL * e.abs()
        if bool(bad.any()):
            raise AssertionError(
                f"{label}: {name} {int(bad.sum())}/{bad.numel()} elements outside "
                f"atol={ATOL} rtol={RTOL} (max abs {diff.max().item():.4g})"
            )
        rel = (diff.norm() / e.norm()).item()
        if rel > MAX_REL_RMS:
            raise AssertionError(f"{label}: {name} relative RMS error {rel:.3g} > {MAX_REL_RMS}")
        worst = max(worst, diff.max().item())
    return worst


def _check_guard(fn, data, stream, label):
    """A foreign embodiment_id: NaN actions and action rows, state row kept."""

    data["embodiment_id"].fill_(baseline.EMBODIMENT + 1)
    _poison(data)
    data["sa_embs"][:, :1].zero_()
    _call(fn, stream)
    data["embodiment_id"].fill_(baseline.EMBODIMENT)
    ok = (
        bool(torch.isnan(data["actions_out"]).all())
        and bool(torch.isnan(data["sa_embs"][:, 1:]).all())
        and torch.equal(data["sa_embs"][:, :1], data["state_features"])
    )
    if not ok:
        raise AssertionError(
            f"{label}: embodiment guard (NaN outputs for a foreign embodiment_id) not honored"
        )


def _perturb(data, seed):
    for i, name in enumerate(MUTABLE):
        t = data[name]
        gen = torch.Generator(device=t.device).manual_seed(seed + i)
        noise = torch.randn(t.shape, generator=gen, device=t.device, dtype=torch.float32)
        t.copy_((t.float() + 0.25 * t.float().std() * noise).to(t.dtype))


def _kernel_nodes(raw):
    """Kernel nodes in a captured CUDA graph, child graphs included."""

    from cuda.bindings import runtime as rt

    graph = raw if isinstance(raw, rt.cudaGraph_t) else rt.cudaGraph_t(init_value=raw)
    _, _, n = rt.cudaGraphGetNodes(graph, 0)
    _, nodes, n = rt.cudaGraphGetNodes(graph, n)
    count = 0
    for node in nodes[:n]:
        _, kind = rt.cudaGraphNodeGetType(node)
        if kind == rt.cudaGraphNodeType.cudaGraphNodeTypeKernel:
            count += 1
        elif kind == rt.cudaGraphNodeType.cudaGraphNodeTypeGraph:
            _, child = rt.cudaGraphChildGraphNodeGetGraph(node)
            count += _kernel_nodes(child)
    return count


def _graph(fn, stream):
    """(graph of one call, its kernel-node count)."""

    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()  # TensorRT needs one enqueue on the capture stream before capture
        stream.synchronize()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph, stream=stream):
            fn()
    kernels = _kernel_nodes(graph.raw_cuda_graph())
    graph.instantiate()
    torch.cuda.synchronize()
    return graph, kernels


def _mhz(path):
    try:
        with open(path) as f:
            return int(f.read()) // 1_000_000
    except (OSError, ValueError):
        return 0


def _warm(graphs, flush, min_s=0.5, max_s=6.0, hold_s=0.3):
    """Replay until the memory clock has read its maximum for hold_s (Thor DVFS ramps on load)."""

    emc_max = _mhz(f"{DEVFREQ}/bwmgr/max_freq")
    t0, at_max = time.time(), None
    while True:
        for _ in range(5):
            for graph in graphs.values():
                flush()
                graph.replay()
        torch.cuda.synchronize()
        now = time.time()
        if emc_max and _mhz(f"{DEVFREQ}/bwmgr/cur_freq") >= emc_max:
            at_max = at_max or now
        else:
            at_max = None
        if now - t0 >= min_s and (not emc_max or (at_max and now - at_max >= hold_s)):
            return
        if now - t0 >= max_s:
            print("  memory clock did not settle at its maximum", flush=True)
            return


def _time(graphs, cfg, flush):
    _warm(graphs, flush)
    names = list(graphs)
    samples = {name: [] for name in names}
    for _ in range(cfg.trials):
        events = []
        for rep in range(cfg.warmup + cfg.iters):
            for name in names if rep % 2 == 0 else names[::-1]:
                flush()
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                graphs[name].replay()
                end.record()
                if rep >= cfg.warmup:
                    events.append((name, start, end))
        torch.cuda.synchronize()
        for name, start, end in events:
            samples[name].append(start.elapsed_time(end))
    return samples


def _p95(values):
    return statistics.quantiles(values, n=20)[18] if len(values) > 1 else values[0]


@contextmanager
def _gpu_lock():
    """The machine's shared GPU timing lock, if it exists; waits at most GROOT_GPU_LOCK_WAIT_S.

    A blocking flock in a helper thread queues like the other jobs' `flock` waiters (polling with
    LOCK_NB would lose every hand-over to them). If the wait times out, the thread releases the
    lock as soon as it gets it.
    """

    path = os.environ.get("GROOT_GPU_LOCK", "/sources/.thor-gpu-timing.lock")
    if not os.path.exists(path):
        print(f"GPU lock {path} not found; running without it", flush=True)
        yield
        return
    limit = float(os.environ.get("GROOT_GPU_LOCK_WAIT_S", "2700"))
    f = open(path)
    acquired, guard, state = threading.Event(), threading.Lock(), {"abandoned": False}

    def take():
        fcntl.flock(f, fcntl.LOCK_EX)
        with guard:
            if state["abandoned"]:
                fcntl.flock(f, fcntl.LOCK_UN)
                f.close()
                return
            acquired.set()

    t0 = time.monotonic()
    threading.Thread(target=take, daemon=True).start()
    if not acquired.wait(limit):
        with guard:
            if not acquired.is_set():
                state["abandoned"] = True
                raise RuntimeError(
                    f"GPU timing lock {path} busy for {limit:.0f} s (another job is timing); "
                    "retry later"
                )
    print(f"GPU lock {path} held (waited {time.monotonic() - t0:.1f} s)", flush=True)
    try:
        yield
    finally:
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()


def _run_row(k, weights, golden, device, cfg, candidate_fn, candidate_prepare_fn, flush):
    stream = torch.cuda.Stream(device)
    base_data = make_data(weights, golden, k, device)
    base = baseline.TrtBoundary(base_data)
    expected = baseline.reference(base_data)
    _poison(base_data)
    _call(base, stream)
    baseline_err = _check(base_data, expected, "TensorRT baseline")

    def build(alias=False):
        data = make_data(weights, golden, k, device, alias)
        if candidate_fn is None:  # self-check: the baseline in both arms
            return data, baseline.TrtBoundary(data)
        launch = candidate_prepare_fn(data)
        return data, lambda: candidate_fn(launch)

    data, candidate = build()
    snapshot = {name: data[name].clone() for name in (*INPUTS, *baseline.WEIGHT_NAMES)}

    def unchanged(label):
        for name, value in snapshot.items():
            if not torch.equal(data[name], value):
                raise AssertionError(f"{label}: the candidate modified {name}")

    outputs = []
    max_err = 0.0
    for run in range(cfg.correctness_runs):
        _poison(data)
        _call(candidate, stream)
        max_err = max(max_err, _check(data, expected, f"correctness run {run}"))
        outputs.append([data[name].clone() for name in OUTPUTS])
    if cfg.require_repeatable_outputs and any(
        not all(torch.equal(a, b) for a, b in zip(o, outputs[0])) for o in outputs[1:]
    ):
        raise AssertionError("outputs differ between identical calls")
    unchanged("correctness runs")
    checks = cfg.correctness_runs

    if cfg.check_input_dependence:
        _perturb(data, 1000 + k)
        perturbed = baseline.reference(data)
        _poison(data)
        _call(candidate, stream)
        _check(data, perturbed, "after an in-place change of x, actions_in, state_features")
        for name in MUTABLE:
            data[name].copy_(snapshot[name])
        checks += 1

    _check_guard(candidate, data, stream, "guard probe")
    alias_data, alias_candidate = build(alias=True)
    alias_expected = baseline.reference(alias_data)
    _poison(alias_data)
    alias_data["actions_out"].copy_(snapshot["actions_in"])  # actions_out is actions_in
    _call(alias_candidate, stream)
    _check(alias_data, alias_expected, "in place (actions_out is actions_in)")
    checks += 2

    graphs = {}  # destroyed before the engines and modules whose kernels they captured
    try:
        try:
            graphs["baseline"], base_kernels = _graph(base, stream)
            graphs["candidate"], cand_kernels = _graph(candidate, stream)
        except RuntimeError as exc:
            raise AssertionError(f"CUDA graph capture failed: {exc}") from exc
        if candidate_fn is not None and cand_kernels != 1:
            raise AssertionError(
                f"one call launched {cand_kernels} kernels; the contract is one persistent kernel"
            )
        samples = _time(graphs, cfg, flush)
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
        max_abs=max_err,
        baseline_max_abs=baseline_err,
        matched=1.0,
        baseline_ms=baseline_ms,
        kernel_ms=kernel_ms,
        speedup=baseline_ms / kernel_ms,
        baseline_p95_ms=_p95(samples["baseline"]),
        kernel_p95_ms=_p95(samples["candidate"]),
        samples=len(samples["candidate"]),
        baseline_kernels=base_kernels,
        kernel_launches=cand_kernels,
        emc_mhz=_mhz(f"{DEVFREQ}/bwmgr/cur_freq"),
        gpc_mhz=_mhz(f"{DEVFREQ}/gpu-gpc-0/cur_freq"),
        timed=True,
        timer="cuda_event",
        timing_mode="graph_replay_cold_l2",
    )


def _print_aggregate(rows):
    valid = [row for row in rows if row["passed"]]
    if len(valid) != len(rows):
        print("\n# Aggregate result\n| candidate | FAIL | no aggregate score |", flush=True)
        return
    total_baseline = sum(row["baseline_ms"] for row in rows)
    total_kernel = sum(row["kernel_ms"] for row in rows)
    print()
    print("# Aggregate result (the three boundaries of one frame)")
    print(
        f"| candidate ({len(valid)}/{len(rows)} PASS) | {total_kernel * 1000:.1f}us | "
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
    flush_src = torch.ones(FLUSH_BYTES // 4, dtype=torch.float32, device=device)
    flush_dst = torch.empty((), dtype=torch.float32, device=device)

    def flush():
        torch.sum(flush_src, dim=0, out=flush_dst)

    print(
        f"OP: {TASK_NAME} ({len(entries)} boundaries, "
        f"warmup/iters/trials {cfg.warmup}/{cfg.iters}/{cfg.trials})"
    )
    rows = []
    with _gpu_lock():
        for entry in entries:
            k = int(entry["axes"]["boundary"])
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
    _print_aggregate(rows)
    if all(row["passed"] for row in rows):
        summarize(rows, "boundary")
    return rows


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    run_suite()


if __name__ == "__main__":
    main()
