#!/usr/bin/env python3
"""TensorRT timing baseline and oracle for the GR00T N1.7 backbone LLM prefill.

The LLM is 16 Qwen3 decoder layers (hidden 2048, 16 query / 8 key-value heads x 128, SwiGLU FFN
6144) applied to the 156 vision-language tokens of one frame. q / k / v / gate_up are NVFP4 W4A4
(dynamic per-16 activation quantization after an AWQ-style pre-quant scale); o / down are FP8
(static per-tensor activation scale). q and k get a per-head RMSNorm and interleaved multimodal
rope; attention is causal GQA. After layers 0-2 the deepstack features are added to the first 128
visual rows.

The baseline is the deployed engine (TensorRT 10.16, one prefill per call). The oracle is an
independent PyTorch emulation of the exported graph: fp32 arithmetic (TF32 off), fp16 storage where
the graph stores its tensors, and ModelOpt's NVFP4 / FP8 quantization rules.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

SUBDIR = "groot-n17-libero-c235"
LAYERS, HIDDEN, Q_HEADS, KV_HEADS, HEAD_DIM = 16, 2048, 16, 8, 128
MROPE_SECTION = (24, 20, 20)  # rope frequency pairs per (t, h, w) position axis, interleaved
RMS_EPS = 1e-6
DEEPSTACK_LAYERS, DEEPSTACK_ROWS = 3, 128
INPUTS = (
    "inputs_embeds",
    "attention_mask",
    "position_ids",
    "visual_pos_masks",
    "deepstack_0",
    "deepstack_1",
    "deepstack_2",
)
NORMS = ("input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm")
NVFP4 = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "mlp.gateup_proj")
FP8 = ("self_attn.o_proj", "mlp.down_proj")
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def model_dir() -> Path:
    """The provisioned model directory (no download source exists)."""

    root = Path(os.environ.get("HF_MODEL_PATH", "/raid/catalyst/models")).expanduser()
    path = root / SUBDIR
    if not (path / "llm_manifest.json").is_file():
        raise FileNotFoundError(
            f"{path} has no LLM files; build them with groot-perf-diagnosis kda/tirx/m3_prepare.py "
            "and point HF_MODEL_PATH at the directory's parent"
        )
    return path


def load_model(device):
    """(flat weights on `device`, golden tensors on the CPU, golden call ids)."""

    from safetensors.torch import load_file

    path = model_dir()
    weights = {k: v.to(device) for k, v in load_file(str(path / "llm_weights.safetensors")).items()}
    golden = load_file(str(path / "llm_golden.safetensors"))
    calls = json.loads((path / "llm_manifest.json").read_text())["calls"]
    return weights, golden, calls


def weight_names():
    names = []
    for i in range(LAYERS):
        names += [f"layers.{i}.{n}.weight" for n in NORMS]
        for lin in NVFP4:
            names += [f"layers.{i}.{lin}.{f}" for f in ("w", "sf", "gsw", "gsx", "pq")]
        for lin in FP8:
            names += [f"layers.{i}.{lin}.{f}" for f in ("w", "sw", "sx", "sxd", "pq")]
    return [*names, "rope.inv_freq", "attn_scale"]


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


class TrtLlmPrefill:
    """The deployed engine bound to the task's data dict; launches on the current stream."""

    def __init__(self, data):
        trt = _tensorrt()
        blob = (model_dir() / "llm_nvfp4.engine").read_bytes()
        self.engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(blob)
        if self.engine is None:
            raise RuntimeError("cannot deserialize llm_nvfp4.engine")
        self.context = self.engine.create_execution_context()
        for name in INPUTS:
            self.context.set_input_shape(name, tuple(data[name].shape))
            self.context.set_tensor_address(name, data[name].data_ptr())
        self.context.set_tensor_address("embeddings", data["embeddings"].data_ptr())
        self.data = data

    def __call__(self):
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT enqueue failed")
        return self.data["embeddings"]


def _e2m1_rne(y):
    """Nearest e2m1 value, ties to the even mantissa, saturating at 6 (ModelOpt's cast)."""

    grid = torch.tensor(E2M1[:8], device=y.device)
    bounds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=y.device)
    odd_ties = torch.tensor([0.75, 1.75, 3.5], device=y.device)
    a = y.abs().clamp(max=6.0)
    idx = torch.searchsorted(bounds, a) + torch.isin(a, odd_ties).long()
    return torch.sign(y) * grid[idx]


def _qdq_nvfp4(x, gsx):
    """Dynamic NVFP4 quantize-dequantize of an activation along its last axis."""

    blocks = x.reshape(*x.shape[:-1], -1, 16)
    sf = (blocks.abs().amax(-1) / (6.0 * gsx)).clamp(max=448.0).to(torch.float8_e4m3fn).float()
    sf = torch.where(sf == 0, torch.ones_like(sf), sf)
    return (_e2m1_rne(blocks / (sf[..., None] * gsx)) * sf[..., None] * gsx).reshape(x.shape)


def _linear_nvfp4(data, prefix, x16):
    """fp16 in -> * pre-quant scale (fp16) -> NVFP4 q/dq -> fp32 GEMM -> fp16 out (no bias)."""

    x = (x16 * data[f"{prefix}.pq"]).float()
    w = data[f"{prefix}.w"]
    codes = torch.stack([w & 0xF, w >> 4], dim=-1).reshape(w.shape[0], -1)
    sf = data[f"{prefix}.sf"].view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1)
    weight = torch.tensor(E2M1, device=w.device)[codes.long()] * sf * data[f"{prefix}.gsw"].float()
    return (_qdq_nvfp4(x, data[f"{prefix}.gsx"].float()) @ weight.t()).half()


def _linear_fp8(data, prefix, x16):
    """fp16 in -> * pre-quant scale (fp16) -> static FP8 q/dq -> fp32 GEMM -> fp16 out (no bias)."""

    x = (x16 * data[f"{prefix}.pq"]).float()
    sx, sxd = data[f"{prefix}.sx"].float(), data[f"{prefix}.sxd"].float()
    xq = (x / sx).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * sxd
    weight = data[f"{prefix}.w"].view(torch.float8_e4m3fn).float() * data[f"{prefix}.sw"].float()
    return (xq @ weight.t()).half()


def _rmsnorm(x16, w16):
    x = x16.float()
    y = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + RMS_EPS)).half()
    return (w16.float() * y.float()).half()


def _rope_tables(position_ids, inv_freq):
    """Interleaved multimodal rope: frequency pair j uses the t position, except j % 3 == 1 (h)
    and j % 3 == 2 (w) for j < 3 * section. Returns fp16 cos / sin [L, 128]."""

    pos = position_ids[:, 0, :].float()  # [3, L]
    angles = pos[:, :, None] * inv_freq.float()[None, None, :]  # [3, L, 64]
    mixed = angles[0].clone()
    for axis in (1, 2):
        idx = torch.arange(axis, 3 * MROPE_SECTION[axis], 3, device=pos.device)
        mixed[:, idx] = angles[axis][:, idx]
    full = torch.cat((mixed, mixed), dim=-1)
    return full.cos().half(), full.sin().half()


def _rope(x16, cos, sin):
    x = x16.float()
    rotated = torch.cat((-x[..., HEAD_DIM // 2 :], x[..., : HEAD_DIM // 2]), dim=-1)
    return (x * cos.float() + rotated * sin.float()).half()


def _attention(q16, k16, v16, scale):
    """Causal GQA over [L, heads, 128] fp16 tensors; fp32 scores and softmax, fp16 probabilities."""

    n = q16.shape[0]
    group = Q_HEADS // KV_HEADS
    q = q16.float().transpose(0, 1)
    k = k16.float().transpose(0, 1).repeat_interleave(group, dim=0)
    v = v16.float().transpose(0, 1).repeat_interleave(group, dim=0)
    s = (q @ k.transpose(-1, -2)) * scale
    causal = torch.ones(n, n, dtype=torch.bool, device=q.device).tril()
    p = torch.softmax(s.masked_fill(~causal, float("-inf")), dim=-1).half().float()
    return (p @ v).transpose(0, 1).reshape(n, HIDDEN).half()


@torch.no_grad()
def reference(data):
    """Independent oracle: embeddings [1, L, 2048] fp16 from the inputs and weights."""

    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        h = data["inputs_embeds"][0].clone()
        n = h.shape[0]
        cos, sin = _rope_tables(data["position_ids"], data["rope.inv_freq"])
        cos, sin = cos[:, None, :], sin[:, None, :]
        rows = data["visual_pos_masks"][0].nonzero().squeeze(1)[:DEEPSTACK_ROWS]
        scale = data["attn_scale"].float()
        for i in range(LAYERS):
            p = f"layers.{i}"
            x = _rmsnorm(h, data[f"{p}.input_layernorm.weight"])
            q = _linear_nvfp4(data, f"{p}.self_attn.q_proj", x).view(n, Q_HEADS, HEAD_DIM)
            k = _linear_nvfp4(data, f"{p}.self_attn.k_proj", x).view(n, KV_HEADS, HEAD_DIM)
            v = _linear_nvfp4(data, f"{p}.self_attn.v_proj", x).view(n, KV_HEADS, HEAD_DIM)
            q = _rope(_rmsnorm(q, data[f"{p}.self_attn.q_norm.weight"]), cos, sin)
            k = _rope(_rmsnorm(k, data[f"{p}.self_attn.k_norm.weight"]), cos, sin)
            a = _attention(q, k, v, scale)
            h = (h.float() + _linear_fp8(data, f"{p}.self_attn.o_proj", a).float()).half()
            gu = _linear_nvfp4(
                data,
                f"{p}.mlp.gateup_proj",
                _rmsnorm(h, data[f"{p}.post_attention_layernorm.weight"]),
            ).float()
            gate, up = gu[:, : gu.shape[1] // 2], gu[:, gu.shape[1] // 2 :]
            s = ((gate * torch.sigmoid(gate)).half().float() * up).half()
            h = (h.float() + _linear_fp8(data, f"{p}.mlp.down_proj", s).float()).half()
            if i < DEEPSTACK_LAYERS:
                h[rows] = (h[rows].float() + data[f"deepstack_{i}"].float()).half()
        return h[None]
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
