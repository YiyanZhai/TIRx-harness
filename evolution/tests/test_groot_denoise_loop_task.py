"""Check the GR00T denoising-loop task's packaging and gate on CPU."""

from dataclasses import replace

import pytest
import torch

from evolution.benchmark import adapter
from evolution.preparation.declare import load_task

NAME = "groot_denoise_loop_thor"
KEY = "groot/denoise_loop_thor"
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
    assert [row["axes"]["vl_tokens"] for row in rows] == [156] * 4 + [140, 145, 149, 150, 151, 152, 155, 172]
    sources = adapter.harness_sources(name)
    # The loop reuses both the boundary task's and the DiT task's baselines and oracles.
    for file in (
        f"tasks/{NAME}/benchmark.py",
        f"tasks/{NAME}/baseline.py",
        f"tasks/{NAME}/definition.json",
        "tasks/groot_dit_step_thor/__init__.py",
        "tasks/groot_dit_step_thor/baseline.py",
        "tasks/groot_ah_boundary_thor/benchmark.py",
        "tasks/groot_ah_boundary_thor/baseline.py",
    ):
        assert f"flashinfer_bench_evolve/{file}" in sources


def test_weight_names():
    block_types = ["cross", "self"] * 16
    names = bench.baseline.weight_names(block_types)
    assert names[:10] == list(bench.baseline.ah.WEIGHT_NAMES)
    assert names[-1] == "proj_out_2.gsx" and len(names) == 10 + (16 * 4 + 16 * 6 + 1) * 5


def _data(gen):
    expected = (
        torch.randn(1, 40, 132, generator=gen).half(),
        torch.randn(4, 1, 41, 1024, generator=gen).half(),
    )
    return {"actions_out": expected[0].clone(), "model_outputs": expected[1].clone()}, expected


def test_gate_checks_actions_steps_and_state_row():
    data, expected = _data(torch.Generator().manual_seed(0))
    assert bench._check(data, expected, "exact")[:2] == (0.0, 0.0)
    # 1 % on the actions exceeds their limit.
    data["actions_out"] = (expected[0].float() * 1.01).half()
    with pytest.raises(AssertionError, match="actions_out relative L2"):
        bench._check(data, expected, "actions")
    data["actions_out"] = expected[0].clone()
    # Only the state token's row of step 3 is wrong: the whole-step error stays small.
    data["model_outputs"][3, :, :1] = (expected[1][3, :, :1].float() * 1.1).half()
    with pytest.raises(AssertionError, match=r"model_outputs\[3\].*state row"):
        bench._check(data, expected, "state row")
    data["model_outputs"] = expected[1].clone()
    data["model_outputs"][1, 0, 5, 5] = float("inf")
    with pytest.raises(AssertionError, match="non-finite"):
        bench._check(data, expected, "inf")


def test_probe_uses_another_scene():
    gen = torch.Generator().manual_seed(1)
    golden = {
        "probe.cross_keys": torch.randn(16, 1, 32, 156, 48, generator=gen).half(),
        "probe.cross_values": torch.randn(16, 1, 32, 156, 48, generator=gen).half(),
    }
    data = {
        "initial_actions": torch.randn(1, 40, 132, generator=gen).half(),
        "state_features": torch.randn(1, 1, 1536, generator=gen).half(),
        "cross_keys": torch.zeros(16, 1, 32, 156, 48).half(),
        "cross_values": torch.zeros(16, 1, 32, 156, 48).half(),
    }
    before = {k: v.clone() for k, v in data.items()}
    bench._perturb(data, golden, "noise0", 0)
    assert torch.equal(data["cross_keys"], golden["probe.cross_keys"])
    assert torch.equal(data["cross_values"], golden["probe.cross_values"])
    for name in ("initial_actions", "state_features"):
        assert not torch.equal(data[name], before[name])


def test_len_rows_read_their_own_tensors():
    gen = torch.Generator().manual_seed(2)
    golden = {
        "noise0": torch.randn(1, 40, 132, generator=gen).half(),
        "cross_keys": torch.randn(16, 1, 32, 156, 48, generator=gen).half(),
        "probe.cross_keys": torch.randn(16, 1, 32, 156, 48, generator=gen).half(),
        "len145.initial_actions": torch.randn(1, 40, 132, generator=gen).half(),
        "len145.cross_keys": torch.randn(16, 1, 32, 145, 48, generator=gen).half(),
        "len145.probe.cross_keys": torch.randn(16, 1, 32, 145, 48, generator=gen).half(),
    }
    tensor = bench.baseline.row_tensor
    assert tensor(golden, "noise0", "initial_actions") is golden["noise0"]
    assert tensor(golden, "noise0", "probe.cross_keys") is golden["probe.cross_keys"]
    assert tensor(golden, "len145", "initial_actions") is golden["len145.initial_actions"]
    assert tensor(golden, "len145", "cross_keys").shape[3] == 145
    assert tensor(golden, "len145", "probe.cross_keys").shape[3] == 145
    assert bench.baseline.vl_tokens("noise2") == 156 and bench.baseline.vl_tokens("len172") == 172
