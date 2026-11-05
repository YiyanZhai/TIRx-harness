#!/usr/bin/env python3
"""TensorRT timing baseline and oracle for the GR00T N1.7 backbone ViT.

The ViT is the Qwen3-VL vision tower on the 512 patches of one frame's two 256-patch images:
a patch embedding (fp16) plus a learned position embedding, 24 pre-LayerNorm blocks (hidden 1024,
16 heads x 64, 2D rope, attention within each image, GELU-tanh FFN 4096), three deepstack mergers
after blocks 5, 11 and 17, and the final merger. A merger joins 4 consecutive patches (a 2x2
patch group) into one 4096-wide row and maps it to 2048 through an FP8 GELU-erf MLP. Every linear
except the patch embedding is FP8: a static per-tensor activation scale and a per-tensor weight
scale.

The baseline is the deployed engine (TensorRT 10.16, one frame per call). The oracle is an
independent PyTorch emulation of the exported graph: fp32 arithmetic (TF32 off), fp16 storage where
the graph stores its tensors, and the graph's FP8 quantize / dequantize rule.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

SUBDIR = "groot-n17-libero-c235"
BLOCKS, HIDDEN, HEADS, HEAD_DIM, IMAGES = 24, 1024, 16, 64, 2
PATCHES, MERGED, OUT_HIDDEN = 512, 128, 2048
DEEPSTACK_AFTER = (5, 11, 17)
LN_EPS = 1e-6
INPUTS = ("pixel_values",)
OUTPUTS = ("image_embeds", "deepstack_features")
LINEARS = ("qkv", "proj", "fc1", "fc2")


def model_dir() -> Path:
    """The provisioned model directory (no download source exists)."""

    root = Path(os.environ.get("HF_MODEL_PATH", "/raid/catalyst/models")).expanduser()
    path = root / SUBDIR
    if not (path / "vit_manifest.json").is_file():
        raise FileNotFoundError(
            f"{path} has no ViT files; build them with groot-perf-diagnosis kda/tirx/m6_prepare.py "
            "and point HF_MODEL_PATH at the directory's parent"
        )
    return path


def load_model(device):
    """(flat weights on `device`, golden tensors on the CPU, golden call ids)."""

    from safetensors.torch import load_file

    path = model_dir()
    weights = {k: v.to(device) for k, v in load_file(str(path / "vit_weights.safetensors")).items()}
    golden = load_file(str(path / "vit_golden.safetensors"))
    calls = json.loads((path / "vit_manifest.json").read_text())["calls"]
    return weights, golden, calls


def _fp8_names(prefix):
    return [f"{prefix}.{f}" for f in ("w", "sw", "sx", "sxd", "b")]


def weight_names():
    names = ["patch_embed.w", "patch_embed.b", "pos_embed", "rope.cos", "rope.sin", "attn_scale"]
    for i in range(BLOCKS):
        names += [f"blocks.{i}.{n}.{f}" for n in ("norm1", "norm2") for f in ("w", "b")]
        for lin in LINEARS:
            names += _fp8_names(f"blocks.{i}.{lin}")
    for prefix in [f"deepstack.{j}" for j in range(len(DEEPSTACK_AFTER))] + ["merger"]:
        names += [f"{prefix}.norm.w", f"{prefix}.norm.b"]
        names += _fp8_names(f"{prefix}.fc1") + _fp8_names(f"{prefix}.fc2")
    return names


def _tensorrt():
    # The engine needs the TensorRT release that built it; it may sit beside an older image copy.
    site = os.environ.get("GROOT_TRT_SITE", "/opt/trt-10.16.1/site")
    if os.path.isdir(site) and site not in sys.path:
        sys.path.insert(0, site)
    import tensorrt as trt

    if not trt.__version__.startswith("10.16."):
        raise RuntimeError(
            f"TensorRT {trt.__version__} cannot load the 10.16 engine; set GROOT_TRT_SITE"
        )
    return trt


class TrtVit:
    """The deployed engine bound to the task's data dict; launches on the current stream."""

    def __init__(self, data):
        trt = _tensorrt()
        blob = (model_dir() / "vit_fp8.engine").read_bytes()
        self.engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(blob)
        if self.engine is None:
            raise RuntimeError("cannot deserialize vit_fp8.engine")
        self.context = self.engine.create_execution_context()
        for name in (*INPUTS, *OUTPUTS):
            self.context.set_tensor_address(name, data[name].data_ptr())
        self.data = data

    def __call__(self):
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT enqueue failed")
        return self.data["image_embeds"], self.data["deepstack_features"]


def _linear(data, prefix, x16):
    """fp16 in -> static FP8 q (fp32 scale) / dq (its fp16 rounding) -> fp32 GEMM + bias -> fp16."""

    sx, sxd = data[f"{prefix}.sx"].float(), data[f"{prefix}.sxd"].float()
    x = (x16.float() / sx).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * sxd
    weight = data[f"{prefix}.w"].view(torch.float8_e4m3fn).float() * data[f"{prefix}.sw"].float()
    return (x @ weight.t() + data[f"{prefix}.b"].float()).half()


def _layernorm(x16, data, prefix):
    w, b = data[f"{prefix}.w"].float(), data[f"{prefix}.b"].float()
    return torch.nn.functional.layer_norm(x16.float(), (x16.shape[-1],), w, b, LN_EPS).half()


def _h(x):
    return torch.tensor(x, dtype=torch.float16)


def _gelu_tanh(x16):
    """The graph's tanh GELU, every step stored in fp16."""

    c0, c1 = _h(0.044715).to(x16.device), _h(0.7978845608).to(x16.device)
    a = x16 * (x16 * x16)
    a = x16 + c0 * a
    a = torch.tanh((c1 * a).float()).half()
    a = x16 * (1.0 + a.float()).half()
    return (0.5 * a.float()).half()


def _gelu_erf(x16):
    """The graph's erf GELU, every step stored in fp16 (sqrt(2) as its fp16 rounding)."""

    a = torch.erf((x16.float() / float(_h(2.0**0.5))).half().float()).half()
    a = x16 * (a.float() + 1.0).half()
    return (a.float() * 0.5).half()


def _rope(x16, cos, sin):
    x = x16.float()
    rotated = torch.cat((-x[..., HEAD_DIM // 2 :], x[..., : HEAD_DIM // 2]), dim=-1)
    return (x * cos + rotated * sin).half()


def _attention(qkv16, data):
    """Per image: 2D rope on q and k, fp16 scores times the fp16 scale, fp32 softmax, fp16
    probabilities."""

    n = qkv16.shape[0]
    qkv = qkv16.view(n, 3, HEADS, HEAD_DIM)
    cos, sin = data["rope.cos"].float()[:, None, :], data["rope.sin"].float()[:, None, :]
    q, k, v = _rope(qkv[:, 0], cos, sin), _rope(qkv[:, 1], cos, sin), qkv[:, 2]
    scale = data["attn_scale"].half()
    per = n // IMAGES
    outs = []
    for i in range(IMAGES):
        rows = slice(i * per, (i + 1) * per)
        qi, ki, vi = (t[rows].transpose(0, 1).float() for t in (q, k, v))
        s = ((qi @ ki.transpose(-1, -2)).half() * scale).float()
        p = torch.softmax(s, dim=-1).half().float()
        outs.append((p @ vi).half().transpose(0, 1))
    return torch.cat(outs, 0).reshape(n, HIDDEN)


def _merge(h16, data, prefix, final):
    if final:  # LayerNorm over 1024, then the 2x2 merge
        x = _layernorm(h16, data, f"{prefix}.norm").reshape(-1, 4 * HIDDEN)
    else:  # deepstack: the 2x2 merge, then LayerNorm over 4096
        x = _layernorm(h16.reshape(-1, 4 * HIDDEN), data, f"{prefix}.norm")
    x = _gelu_erf(_linear(data, f"{prefix}.fc1", x))
    return _linear(data, f"{prefix}.fc2", x)


@torch.no_grad()
def reference(data):
    """Independent oracle: (image_embeds [128, 2048], deepstack_features [3, 128, 2048]) fp16."""

    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        px = data["pixel_values"].float()
        patch = px @ data["patch_embed.w"].float().t() + data["patch_embed.b"].float()
        h = patch.half() + data["pos_embed"]
        deepstack = []
        for i in range(BLOCKS):
            p = f"blocks.{i}"
            a = _attention(_linear(data, f"{p}.qkv", _layernorm(h, data, f"{p}.norm1")), data)
            h = (h.float() + _linear(data, f"{p}.proj", a).float()).half()
            f = _gelu_tanh(_linear(data, f"{p}.fc1", _layernorm(h, data, f"{p}.norm2")))
            h = (h.float() + _linear(data, f"{p}.fc2", f).float()).half()
            if i in DEEPSTACK_AFTER:
                j = DEEPSTACK_AFTER.index(i)
                deepstack.append(_merge(h, data, f"deepstack.{j}", final=False))
        return _merge(h, data, "merger", final=True), torch.stack(deepstack)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
