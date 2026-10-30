"""Check the GR00T LLM prefill task's packaging and gate on CPU."""

from dataclasses import replace

import pytest
import torch

from evolution.benchmark import adapter
from evolution.preparation.declare import load_task

NAME = "groot_llm_prefill_thor"
KEY = "groot/llm_prefill_thor"
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
    # The shared timing and lock helpers travel with the bundle.
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
    assert "layers.0.self_attn.q_proj.gsx" in names and "layers.15.mlp.down_proj.sxd" in names
    assert names[-2:] == ["rope.inv_freq", "attn_scale"]
    assert len(names) == 16 * (4 + 4 * 5 + 2 * 5) + 2


def test_rope_tables_interleave_axes():
    pos = torch.stack([torch.arange(6), torch.arange(6) * 10, torch.arange(6) * 100])[:, None, :]
    inv_freq = torch.ones(64)
    cos, _ = bench.baseline._rope_tables(pos, inv_freq)
    expected_axis = [
        1 if j % 3 == 1 and j < 60 else 2 if j % 3 == 2 and j < 60 else 0 for j in range(64)
    ]
    for j, axis in enumerate(expected_axis):
        assert torch.equal(cos[:, j], torch.cos(pos[axis, 0].float()).half())
    assert torch.equal(cos[:, 64:], cos[:, :64])


def test_gate_is_per_token():
    gen = torch.Generator().manual_seed(0)
    expected = torch.randn(1, 156, 2048, generator=gen).half()
    expected[0, 0, :8] = 15000.0  # the attention sink dominates a whole-tensor norm
    near = (expected.float() * 1.01).half()
    bench._check(near, expected, bench.MAX_TOK_MEAN[0], bench.MAX_REL[0], bench.MAX_TOK, "near")
    # Every token but the sink is wrong: the whole-tensor error stays within its limit, the per-token mean does not.
    bad = expected.clone()
    bad[0, 1:] = torch.randn(155, 2048, generator=gen).half()
    tok_mean, _, _, rel = bench._errors(bad, expected)
    assert rel < bench.MAX_REL[0] and tok_mean > bench.MAX_TOK_MEAN[0]
    with pytest.raises(AssertionError, match="per-token relative L2 mean"):
        bench._check(bad, expected, bench.MAX_TOK_MEAN[0], bench.MAX_REL[0], bench.MAX_TOK, "bad")
    # One garbage token: the mean barely moves, the per-token maximum catches it.
    one = expected.clone()
    one[0, 77] = -one[0, 77]
    tok_mean, tok_max, worst, _ = bench._errors(one, expected)
    assert tok_mean < bench.MAX_TOK_MEAN[0] and tok_max > bench.MAX_TOK and worst == 77
    with pytest.raises(AssertionError, match="at token 77"):
        bench._check(one, expected, bench.MAX_TOK_MEAN[0], bench.MAX_REL[0], bench.MAX_TOK, "one")
    nan = expected.clone()
    nan[0, 3, 3] = float("nan")
    with pytest.raises(AssertionError, match="non-finite"):
        bench._check(nan, expected, 1.0, 1.0, 9.0, "nan")
