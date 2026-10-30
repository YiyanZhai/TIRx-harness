#!/usr/bin/env python3
"""TensorRT timing baseline and oracle for one GR00T N1.7 DiT denoising step.

The DiT runs 32 transformer blocks on 41 tokens (hidden 1536, 32 heads x 48, FFN 6144). Even
blocks cross-attend to precomputed per-frame K/V [16, 1, 32, 156, 48] (text mask when the block
index is a multiple of 4, image mask otherwise); odd blocks self-attend. Every linear is NVFP4
W4A4: the input is quantized dynamically (per-16 e2m1 with ue4m3 scales under a static global
scale gsx), the weight is e2m1 with ue4m3 per-16 scales under a global scale gsw.

The baseline is the deployed engine (TensorRT 10.16, one step per call). The oracle is an
independent PyTorch emulation of the exported graph: fp32 arithmetic (TF32 off), fp16 storage
where the graph stores its tensors, and ModelOpt's NVFP4 quantization rule.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

SUBDIR = "groot-n17-libero-c235"
HEADS, HEAD_DIM, HIDDEN = 32, 48, 1536
INPUTS = (
    "sa_embs",
    "cross_keys",
    "cross_values",
    "adaln_modulations",
    "output_modulation",
    "image_mask",
    "backbone_attention_mask",
)
LINEARS = {
    "cross": ("attn1.to_q", "attn1.to_out.0", "ff.net.0.proj", "ff.net.2"),
    "self": (
        "attn1.to_q",
        "attn1.to_k",
        "attn1.to_v",
        "attn1.to_out.0",
        "ff.net.0.proj",
        "ff.net.2",
    ),
}
GELU_C0 = float(torch.tensor(0.044715, dtype=torch.float16))  # the graph's fp16 constants
GELU_C1 = float(torch.tensor(0.7978845608, dtype=torch.float16))
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def model_dir() -> Path:
    """The provisioned model directory (no download source exists)."""

    root = Path(os.environ.get("HF_MODEL_PATH", "/raid/catalyst/models")).expanduser()
    path = root / SUBDIR
    if not (path / "dit_manifest.json").is_file():
        raise FileNotFoundError(
            f"{path} has no DiT files; build them with groot-perf-diagnosis kda/tirx/m2_prepare.py "
            "and point HF_MODEL_PATH at the directory's parent"
        )
    return path


def load_model(device):
    """(flat weights on `device`, golden tensors on the CPU, block types)."""

    from safetensors.torch import load_file

    path = model_dir()
    weights = {k: v.to(device) for k, v in load_file(str(path / "dit_weights.safetensors")).items()}
    golden = load_file(str(path / "dit_golden.safetensors"))
    block_types = json.loads((path / "dit_manifest.json").read_text())["block_types"]
    return weights, golden, block_types


def weight_names(block_types):
    names = []
    for i, kind in enumerate(block_types):
        for lin in LINEARS[kind]:
            names += [f"blocks.{i}.{lin}.{f}" for f in ("w", "sf", "b", "gsw", "gsx")]
    return names + [f"proj_out_2.{f}" for f in ("w", "sf", "b", "gsw", "gsx")]


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


class TrtDitStep:
    """The deployed engine bound to the task's data dict; launches on the current stream."""

    def __init__(self, data):
        trt = _tensorrt()
        blob = (model_dir() / "dit_nvfp4.engine").read_bytes()
        self.engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(blob)
        if self.engine is None:
            raise RuntimeError("cannot deserialize dit_nvfp4.engine")
        self.context = self.engine.create_execution_context()
        for name in INPUTS:
            self.context.set_input_shape(name, tuple(data[name].shape))
            self.context.set_tensor_address(name, data[name].data_ptr())
        self.context.set_tensor_address("output", data["output"].data_ptr())
        self.data = data

    def __call__(self):
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT enqueue failed")
        return self.data["output"]


def _e2m1_rne(y):
    """Nearest e2m1 value, ties to the even mantissa, saturating at 6 (ModelOpt's cast)."""

    grid = torch.tensor(E2M1[:8], device=y.device)
    bounds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=y.device)
    odd_ties = torch.tensor([0.75, 1.75, 3.5], device=y.device)
    a = y.abs().clamp(max=6.0)
    idx = torch.searchsorted(bounds, a) + torch.isin(a, odd_ties).long()
    return torch.sign(y) * grid[idx]


def _qdq_input(x, gsx):
    """Dynamic NVFP4 quantize-dequantize of an activation along its last axis."""

    blocks = x.reshape(*x.shape[:-1], -1, 16)
    sf = (blocks.abs().amax(-1) / (6.0 * gsx)).clamp(max=448.0).to(torch.float8_e4m3fn).float()
    sf = torch.where(sf == 0, torch.ones_like(sf), sf)
    return (_e2m1_rne(blocks / (sf[..., None] * gsx)) * sf[..., None] * gsx).reshape(x.shape)


def _weight(data, prefix):
    w = data[f"{prefix}.w"]
    sf = data[f"{prefix}.sf"].view(torch.float8_e4m3fn).float()
    codes = torch.stack([w & 0xF, w >> 4], dim=-1).reshape(w.shape[0], -1)
    lut = torch.tensor(E2M1, device=w.device)
    return lut[codes.long()] * sf.repeat_interleave(16, dim=1) * data[f"{prefix}.gsw"].float()


def _linear(data, prefix, x16):
    """fp16 in -> NVFP4 q/dq -> fp32 GEMM -> + bias -> fp16 out."""

    x = _qdq_input(x16.float(), data[f"{prefix}.gsx"].float())
    return (x @ _weight(data, prefix).t() + data[f"{prefix}.b"].float()).half()


def _layernorm(x16, eps):
    x = x16.float()
    mu = x.mean(-1, keepdim=True)
    return (x - mu) / torch.sqrt(((x - mu) ** 2).mean(-1, keepdim=True) + eps)


def _attention(q16, k16, v16, mask=None):
    q = q16.float().view(1, -1, HEADS, HEAD_DIM).transpose(1, 2)
    s = (q @ k16.float().transpose(-1, -2)) / HEAD_DIM**0.5
    if mask is not None:
        s = s.masked_fill(~mask[:, None, None, :], float("-inf"))
    o = torch.softmax(s, dim=-1) @ v16.float()
    return o.transpose(1, 2).reshape(1, -1, HIDDEN).half()


def _heads(x16):
    return x16.float().view(1, -1, HEADS, HEAD_DIM).transpose(1, 2).half()


@torch.no_grad()
def reference(data, block_types):
    """Independent oracle: the step's output [1, 41, 1024] fp16 from the inputs and weights."""

    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        h = data["sa_embs"]
        mods = data["adaln_modulations"].float()
        image, valid = data["image_mask"], data["backbone_attention_mask"]
        image_mask, text_mask = image & valid, (~image) & valid
        cross = 0
        for i, kind in enumerate(block_types):
            p = f"blocks.{i}"
            scale, shift = mods[i][:, None, :HIDDEN], mods[i][:, None, HIDDEN:]
            n = (_layernorm(h, 1e-5) * (1 + scale) + shift).half()
            q = _linear(data, f"{p}.attn1.to_q", n)
            if kind == "cross":
                mask = text_mask if i % 4 == 0 else image_mask
                a = _attention(q, data["cross_keys"][cross], data["cross_values"][cross], mask)
                cross += 1
            else:
                k = _heads(_linear(data, f"{p}.attn1.to_k", n))
                v = _heads(_linear(data, f"{p}.attn1.to_v", n))
                a = _attention(q, k, v)
            h = (_linear(data, f"{p}.attn1.to_out.0", a).float() + h.float()).half()
            u = _linear(data, f"{p}.ff.net.0.proj", _layernorm(h, 1e-5).half()).float()
            g = (0.5 * u * (1.0 + torch.tanh(GELU_C1 * (u + GELU_C0 * u * u * u)))).half()
            h = (_linear(data, f"{p}.ff.net.2", g).float() + h.float()).half()
        om = data["output_modulation"].float()
        shift, scale = om[:, None, :HIDDEN], om[:, None, HIDDEN:]
        return _linear(data, "proj_out_2", (_layernorm(h, 1e-6) * (1 + scale) + shift).half())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
