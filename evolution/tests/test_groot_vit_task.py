"""Check the GR00T ViT task's packaging and gate on CPU."""

from dataclasses import replace

import pytest
import torch

from evolution.benchmark import adapter
from evolution.preparation.declare import load_task

NAME = "groot_vit_thor"
KEY = "groot/vit_thor"
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
    head = ["patch_embed.w", "patch_embed.b", "pos_embed", "rope.cos", "rope.sin", "attn_scale"]
    assert names[:6] == head and names[-1] == "merger.fc2.b"
    assert "blocks.23.fc2.sxd" in names and "deepstack.2.norm.b" in names
    assert len(names) == 6 + 24 * (4 + 4 * 5) + 4 * (2 + 2 * 5)


def _outputs(gen):
    return {
        "image_embeds": torch.randn(128, 2048, generator=gen).half(),
        "deepstack_features": torch.randn(3, 128, 2048, generator=gen).half(),
    }


def test_gate_checks_every_output():
    gen = torch.Generator().manual_seed(0)
    data = _outputs(gen)
    expected = (data["image_embeds"].clone(), data["deepstack_features"].clone())
    errors = bench._check(data, expected, "exact")
    assert errors == [0.0, 0.0, 0.0, 0.0]
    # deepstack 0 is the tightest limit: an error image_embeds would tolerate fails there.
    data["deepstack_features"][0] = (expected[1][0].float() * 1.1).half()
    with pytest.raises(AssertionError, match="deepstack 0 relative L2"):
        bench._check(data, expected, "deepstack 0 off by 10 %")
    data["deepstack_features"][0] = expected[1][0]
    data["image_embeds"] = (expected[0].float() * 1.3).half()
    bench._check(data, expected, "image_embeds off by 30 %")
    data["image_embeds"] = (expected[0].float() * 1.4).half()
    with pytest.raises(AssertionError, match="image_embeds relative L2"):
        bench._check(data, expected, "image_embeds off by 40 %")
    data["image_embeds"] = expected[0].clone()
    data["image_embeds"][3, 3] = float("nan")
    with pytest.raises(AssertionError, match="non-finite"):
        bench._check(data, expected, "nan")


def test_gelu_steps_round_in_fp16():
    x = torch.linspace(-6, 6, 4097).half()
    tanh = bench.baseline._gelu_tanh(x).float()
    erf = bench.baseline._gelu_erf(x).float()
    xf = x.float()
    exact_tanh = 0.5 * xf * (1 + torch.tanh(0.7978845608 * (xf + 0.044715 * xf**3)))
    exact_erf = 0.5 * xf * (1 + torch.erf(xf / 2**0.5))
    assert tanh.dtype == erf.dtype == torch.float32
    assert (tanh - exact_tanh).abs().max() < 0.02 and (erf - exact_erf).abs().max() < 0.02
    assert not torch.equal(tanh, erf)
