"""Reference snapshots preserve values and the surrounding allocator settings."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from evolution.benchmark import adapter

adapter._bench_repo_on_path()
from flashinfer_bench_evolve.tasks.qwen38 import baseline  # noqa: E402


@pytest.fixture
def allocator(monkeypatch):
    monkeypatch.setattr(baseline.torch, "__version__", "2.13.0")
    for name in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
        monkeypatch.delenv(name, raising=False)
    settings = {
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
        "pinned_use_cuda_host_register": False,
        "pinned_num_register_threads": 1,
        "max_split_size_mb": 512,
    }
    original = deepcopy(settings)
    calls = []

    def configure(value):
        calls.append(value)
        settings["PYTORCH_CUDA_ALLOC_CONF"] = value
        for pair in filter(None, value.split(",")):
            name, raw = pair.split(":")
            settings[name] = raw == "True" if name == "pinned_use_cuda_host_register" else int(raw)

    memory = SimpleNamespace(
        _snapshot=lambda: {"allocator_settings": deepcopy(settings)},
        _set_allocator_settings=configure,
    )
    monkeypatch.setattr(baseline.torch.cuda, "memory", memory)
    return settings, original, calls


@pytest.mark.parametrize("failed", [False, True])
def test_allocator_restored_before_candidate_execution(allocator, failed):
    settings, original, calls = allocator
    try:
        with baseline._snapshot_allocator(True):
            assert settings["pinned_use_cuda_host_register"]
            assert 1 <= settings["pinned_num_register_threads"] <= 8
            assert settings["max_split_size_mb"] == 512
            if failed:
                raise RuntimeError("allocation failed")
    except RuntimeError:
        assert failed
    assert settings == original
    assert calls


@pytest.mark.parametrize("name", ["PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"])
def test_explicit_allocator_configuration_is_respected(allocator, monkeypatch, name):
    settings, original, calls = allocator
    monkeypatch.setenv(name, "pinned_use_cuda_host_register:False")
    with baseline._snapshot_allocator(True):
        assert settings == original
    assert not calls


def test_cpu_snapshot_is_independent_and_does_not_configure_cuda(allocator):
    settings, original, calls = allocator
    source = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    saved = baseline.snapshot(source)
    assert torch.equal(source, saved)
    assert saved.dtype == source.dtype
    source.zero_()
    assert saved.sum() > 0
    assert not calls and settings == original


def test_missing_allocator_introspection_uses_existing_path(allocator, monkeypatch):
    settings, original, calls = allocator
    monkeypatch.setattr(baseline.torch.cuda.memory, "_snapshot", lambda: {})
    with baseline._snapshot_allocator(True):
        assert settings == original
    assert not calls


def test_older_allocator_uses_existing_path(allocator, monkeypatch):
    settings, original, calls = allocator
    monkeypatch.setattr(baseline.torch, "__version__", "2.12.0")
    with baseline._snapshot_allocator(True):
        assert settings == original
    assert not calls


def test_remote_bundle_contains_snapshot_implementation():
    sources = adapter.harness_sources("qwen38_decode")
    name = "flashinfer_bench_evolve/tasks/qwen38/baseline.py"
    assert sources[name] == (adapter.BENCH_REPO_ROOT / name).read_text()
    assert "def _snapshot_allocator" in sources[name]
