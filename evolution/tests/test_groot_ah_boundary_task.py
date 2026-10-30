"""Check the GR00T action-head boundary task's packaging, oracle semantics and gate on CPU."""

from dataclasses import replace

import pytest
import torch

from evolution.benchmark import adapter
from evolution.preparation.declare import load_task

NAME = "groot_ah_boundary_thor"
KEY = "groot/ah_boundary_thor"
bench = adapter.import_benchmark(NAME)
baseline = bench.baseline


def _data(seed=0, embodiment=2):
    gen = torch.Generator().manual_seed(seed)

    def rand(*shape, scale=1.0):
        return (torch.randn(*shape, generator=gen) * scale).half()

    data = {
        "L1": rand(1024, 1024, scale=0.03),
        "c1": rand(1024, scale=0.1),
        "L2": rand(132, 1024, scale=0.03),
        "c2": rand(132, scale=0.1),
        "W1": rand(1536, 132, scale=0.08),
        "b1": rand(1536, scale=0.1),
        "W2": rand(1536, 3072, scale=0.02),
        "b2": rand(1536, scale=0.1),
        "W3": rand(1536, 1536, scale=0.025),
        "b3": rand(1536, scale=0.1),
        "x": rand(1, 41, 1024),
        "actions_in": rand(1, 40, 132),
        "timestep_embedding": rand(1, 1536).expand(40, 1536).reshape(1, 40, 1536).clone(),
        "state_features": rand(1, 1, 1536),
        "position_embedding": rand(1, 40, 1536, scale=0.02),
        "embodiment_id": torch.tensor([embodiment]),
        "dt": baseline.DT,
        "embodiment": baseline.EMBODIMENT,
    }
    data["actions_out"] = torch.empty_like(data["actions_in"])
    data["sa_embs"] = torch.empty(1, 41, 1536, dtype=torch.float16)
    return data


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
    assert [row["axes"]["boundary"] for row in rows] == [0, 1, 2] and candidate is None
    sources = adapter.harness_sources(name)
    for file in ("baseline.py", "benchmark.py", "definition.json"):
        assert f"flashinfer_bench_evolve/tasks/{name}/{file}" in sources


def test_oracle_shapes_and_guard():
    data = _data()
    actions, sa = baseline.reference(data)
    assert actions.shape == (1, 40, 132) and sa.shape == (1, 41, 1536)
    assert actions.dtype == sa.dtype == torch.float16
    assert bool(torch.isfinite(actions).all()) and bool(torch.isfinite(sa).all())
    assert torch.equal(sa[:, :1], data["state_features"])

    foreign = _data(embodiment=3)
    actions, sa = baseline.reference(foreign)
    assert bool(torch.isnan(actions).all()) and bool(torch.isnan(sa[:, 1:]).all())
    assert torch.equal(sa[:, :1], foreign["state_features"])


def test_gate_accepts_oracle_and_rejects_one_bad_element():
    data = _data()
    expected = baseline.reference(data)
    data["actions_out"].copy_(expected[0])
    data["sa_embs"].copy_(expected[1])
    assert bench._check(data, expected, "oracle") == 0.0

    data["sa_embs"][0, 7, 100] += 1.0
    with pytest.raises(AssertionError, match="sa_embs 1/"):
        bench._check(data, expected, "corrupted")

    data["sa_embs"].copy_(expected[1])
    data["actions_out"][0, 3, 5] = float("nan")
    with pytest.raises(AssertionError, match="non-finite"):
        bench._check(data, expected, "nan")
