"""Check the GR00T DiT step task's packaging and gate on CPU."""

from dataclasses import replace

import pytest
import torch

from evolution.benchmark import adapter
from evolution.preparation.declare import load_task

NAME = "groot_dit_step_thor"
KEY = "groot/dit_step_thor"
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
    assert [row["axes"]["row"] for row in rows] == list(bench.ROWS) and candidate is None
    assert [row["axes"]["step"] for row in rows] == [0, 1, 2, 3] * 3
    assert [row["axes"]["vl_tokens"] for row in rows] == [156] * 4 + [140, 145, 149, 150, 151, 152, 155, 172]
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


def test_weight_names_cover_both_block_kinds():
    names = bench.baseline.weight_names(["cross", "self"])
    assert "blocks.0.attn1.to_q.w" in names and "blocks.0.attn1.to_k.w" not in names
    assert "blocks.1.attn1.to_v.gsx" in names and names[-1] == "proj_out_2.gsx"
    assert len(names) == (4 + 6) * 5 + 5


def test_gate_limits():
    gen = torch.Generator().manual_seed(0)
    expected = torch.randn(1, 41, 1024, generator=gen).half()
    near = (expected.float() * 1.02).half()  # 2 % relative error passes every step
    for k in range(4):
        bench._check(near, expected, bench.MAX_REL[k], bench.MAX_ABS[k], "near")
    far = (expected.float() * 1.1).half()
    with pytest.raises(AssertionError, match="relative L2 error"):
        bench._check(far, expected, bench.MAX_REL[3], bench.MAX_ABS[3], "far")
    bad = expected.clone()
    bad[0, 0, 0] = float("nan")
    with pytest.raises(AssertionError, match="non-finite"):
        bench._check(bad, expected, 1.0, 1.0, "nan")


def test_len_rows_read_their_own_tensors():
    golden = {
        "cross_keys": torch.zeros(16, 1, 32, 156, 48).half(),
        "step1.sa_embs": torch.zeros(1, 41, 1536).half(),
        "step1.adaln_modulations": torch.zeros(32, 1, 3072).half(),
        "len145_s1.cross_keys": torch.zeros(16, 1, 32, 145, 48).half(),
        "len145_s1.sa_embs": torch.ones(1, 41, 1536).half(),
    }
    tensor = bench.baseline.row_tensor
    assert tensor(golden, "step1", 1, "cross_keys") is golden["cross_keys"]
    assert tensor(golden, "step1", 1, "sa_embs") is golden["step1.sa_embs"]
    assert tensor(golden, "len145_s1", 1, "cross_keys").shape[3] == 145
    assert tensor(golden, "len145_s1", 1, "sa_embs") is golden["len145_s1.sa_embs"]
    assert tensor(golden, "len145_s1", 1, "adaln_modulations") is golden["step1.adaln_modulations"]
    assert bench.baseline.vl_tokens("step2") == 156 and bench.baseline.vl_tokens("len172_s3") == 172

