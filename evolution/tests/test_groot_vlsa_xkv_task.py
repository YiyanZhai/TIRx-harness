"""Check the GR00T vl_self_attention + cross-KV task's packaging and gate on CPU."""

from dataclasses import replace

import pytest
import torch

from evolution.benchmark import adapter
from evolution.preparation.declare import load_task

NAME = "groot_vlsa_xkv_thor"
KEY = "groot/vlsa_xkv_thor"
bench = adapter.import_benchmark(NAME)


def test_task_plan_and_bundle():
    task = load_task(adapter.REPO_ROOT / f"evolution/tasks/{NAME}.yaml")
    assert adapter.workload_key(task.workload_dir) == KEY
    name, warmup, repeat, mode = adapter.PACKAGED[KEY]
    assert name == NAME
    overrides, rows, candidate = adapter.plan(
        name,
        adapter.CANDIDATES_ROOT / KEY,
        "baseline",
        warmup=warmup,
        repeat=repeat,
        shape_mode=mode,
    )
    assert rows == bench.make_workloads(replace(bench.default_config(), **overrides))
    assert [row["axes"]["frame"] for row in rows] == [0, 1, 30, 31] and candidate is None
    sources = adapter.harness_sources(name)
    for file in (
        f"tasks/{NAME}/benchmark.py",
        f"tasks/{NAME}/baseline.py",
        f"tasks/{NAME}/definition.json",
        "tasks/groot_ah_boundary_thor/benchmark.py",
        "tasks/groot_ah_boundary_thor/baseline.py",
    ):
        assert f"flashinfer_bench_evolve/{file}" in sources


def test_weight_names():
    names = bench.baseline.weight_names()
    assert "vlsa.0.attn1.to_out.0.w" in names and "vlsa.3.ff.net.2.sxd" in names
    assert names[-1] == "xkv.15.to_v.b"
    assert len(names) == 4 * (4 + 8 + 10) + 16 * 10


def test_gelu_rounds_like_the_graph():
    x = torch.linspace(-6, 6, 4097).half()
    y = bench.baseline._gelu_tanh(x).float()
    exact = (
        0.5 * x.float() * (1 + torch.tanh(0.7978845608 * (x.float() + 0.044715 * x.float() ** 3)))
    )
    assert (y - exact).abs().max() < 0.02 and bench.baseline.QK_SCALE == 0.353515625


def test_gate_limits():
    gen = torch.Generator().manual_seed(0)
    shape = (16, 1, 32, 156, 48)
    expected = tuple(torch.randn(shape, generator=gen).half() for _ in range(2))
    data = {name: (e.float() * 1.02).half() for name, e in zip(bench.OUTPUTS, expected)}
    limits = {name: bench.MAX_REL[name][0] for name in bench.OUTPUTS}
    bench._check(data, expected, limits, bench.MAX_ABS, "near")
    data["cross_keys"], data["cross_values"] = data["cross_values"], data["cross_keys"]
    with pytest.raises(AssertionError, match="cross_keys relative L2 error"):
        bench._check(data, expected, limits, bench.MAX_ABS, "swapped")
    data = {name: e.clone() for name, e in zip(bench.OUTPUTS, expected)}
    data["cross_values"][3, 0, 5, 7, 11] += 1.5
    with pytest.raises(AssertionError, match="cross_values .*max abs"):
        bench._check(data, expected, limits, bench.MAX_ABS, "spike")
