#!/usr/bin/env python3
"""TensorRT timing baseline and oracle for GR00T N1.7 vl_self_attention + DiT cross-KV.

Once per frame the action head refines the 156 vision-language tokens with vl_self_attention (4
transformer blocks: affine LayerNorm, fp16 q / k / v / out projections with bias, unmasked attention
32 heads x 64, FP8 GELU FFN 8192) and projects the result to the keys and values of the DiT's 16
cross-attention blocks (32 FP8 linears with bias, output [16, 1, 32, 156, 48] each).

The baseline is the deployed pair of engines (TensorRT 10.16, vl_self_attention then cross-KV, back
to back on one stream). The oracle is an independent PyTorch emulation of the two exported graphs:
fp32 arithmetic (TF32 off), fp16 storage where the graphs store their tensors, and the static FP8
quantization the graphs apply.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

SUBDIR = "groot-n17-libero-c235"
BLOCKS, HIDDEN, HEADS, HEAD_DIM = 4, 2048, 32, 64
CROSS_BLOCKS, KV_HEADS, KV_DIM = 16, 32, 48
LN_EPS = 1e-5
QK_SCALE = float(
    torch.tensor(0.125**0.5, dtype=torch.float16)
)  # each of q and k, in fp16, as the graph does
GELU_C0 = float(torch.tensor(0.044715, dtype=torch.float16))  # the graph's fp16 constants
GELU_C1 = float(torch.tensor(0.7978845608, dtype=torch.float16))
INPUTS = ("hidden_states",)
OUTPUTS = ("cross_keys", "cross_values")
FP16_LINEARS = ("attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0")
FP8_LINEARS = ("ff.net.0.proj", "ff.net.2")
FP8_FIELDS = ("w", "sw", "sx", "sxd", "b")


def model_dir() -> Path:
    """The provisioned model directory (no download source exists)."""

    root = Path(os.environ.get("HF_MODEL_PATH", "/raid/catalyst/models")).expanduser()
    path = root / SUBDIR
    if not (path / "vlsa_manifest.json").is_file():
        raise FileNotFoundError(
            f"{path} has no vl_self_attention files; build them with groot-perf-diagnosis "
            "kda/tirx/m4_prepare.py and point HF_MODEL_PATH at the directory's parent"
        )
    return path


def load_model(device):
    """(flat weights on `device`, golden tensors on the CPU)."""

    from safetensors.torch import load_file

    path = model_dir()
    weights = {
        k: v.to(device) for k, v in load_file(str(path / "vlsa_weights.safetensors")).items()
    }
    return weights, load_file(str(path / "vlsa_golden.safetensors"))


def weight_names():
    names = []
    for b in range(BLOCKS):
        names += [f"vlsa.{b}.{n}.{f}" for n in ("norm1", "norm3") for f in ("weight", "bias")]
        names += [f"vlsa.{b}.{lin}.{f}" for lin in FP16_LINEARS for f in ("w", "b")]
        names += [f"vlsa.{b}.{lin}.{f}" for lin in FP8_LINEARS for f in FP8_FIELDS]
    for j in range(CROSS_BLOCKS):
        names += [f"xkv.{j}.{kv}.{f}" for kv in ("to_k", "to_v") for f in FP8_FIELDS]
    return names


def _tensorrt():
    # The engines need the TensorRT release that built them; it may sit beside an older image copy.
    site = os.environ.get("GROOT_TRT_SITE", "/opt/trt-10.16.1/site")
    if os.path.isdir(site) and site not in sys.path:
        sys.path.insert(0, site)
    import tensorrt as trt

    if not trt.__version__.startswith("10.16."):
        raise RuntimeError(
            f"TensorRT {trt.__version__} cannot load the 10.16 engines; set GROOT_TRT_SITE"
        )
    return trt


class _Engine:
    def __init__(self, trt, runtime, name, bindings):
        self.engine = runtime.deserialize_cuda_engine((model_dir() / name).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"cannot deserialize {name}")
        self.context = self.engine.create_execution_context()
        for tensor, value in bindings.items():
            if self.engine.get_tensor_mode(tensor) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(tensor, tuple(value.shape))
            self.context.set_tensor_address(tensor, value.data_ptr())

    def __call__(self, stream):
        if not self.context.execute_async_v3(stream):
            raise RuntimeError("TensorRT enqueue failed")


class TrtVlsaXkv:
    """The two deployed engines bound to the task's data dict; launch on the current stream."""

    def __init__(self, data):
        trt = _tensorrt()
        runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        x = data["hidden_states"]
        self.vl_embs = torch.empty_like(x)  # the engines' intermediate
        self.vlsa = _Engine(
            trt, runtime, "vl_self_attention.engine", {"hidden_states": x, "output": self.vl_embs}
        )
        self.xkv = _Engine(
            trt,
            runtime,
            "dit_cross_kv.engine",
            {
                "vl_embs": self.vl_embs,
                "cross_keys": data["cross_keys"],
                "cross_values": data["cross_values"],
            },
        )
        self.runtime = runtime
        self.data = data

    def __call__(self):
        stream = torch.cuda.current_stream().cuda_stream
        self.vlsa(stream)
        self.xkv(stream)
        return self.data["cross_keys"], self.data["cross_values"]


def _layernorm(data, prefix, x16):
    x = torch.nn.functional.layer_norm(
        x16.float(),
        (HIDDEN,),
        data[f"{prefix}.weight"].float(),
        data[f"{prefix}.bias"].float(),
        LN_EPS,
    )
    return x.half()


def _linear_fp16(data, prefix, x16):
    return (x16.float() @ data[f"{prefix}.w"].float().t() + data[f"{prefix}.b"].float()).half()


def _linear_fp8(data, prefix, x16):
    """fp16 in -> static FP8 q/dq -> fp32 GEMM -> + bias -> fp16 out."""

    sx, sxd = data[f"{prefix}.sx"].float(), data[f"{prefix}.sxd"].float()
    xq = (x16.float() / sx).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * sxd
    weight = data[f"{prefix}.w"].view(torch.float8_e4m3fn).float() * data[f"{prefix}.sw"].float()
    return (xq @ weight.t() + data[f"{prefix}.b"].float()).half()


def _gelu_tanh(x16):
    """0.5 x (1 + tanh(c1 (x + c0 x^3))) op by op in fp16, as the exported graph computes it."""

    def r(t):
        return t.half().float()

    x = x16.float()
    inner = r(x + r(GELU_C0 * r(x * r(x * x))))
    t = r(torch.tanh(r(GELU_C1 * inner)))
    return r(0.5 * r(x * r(1.0 + t))).half()


def _heads(x16, dim):
    return x16.view(x16.shape[0], -1, dim).transpose(0, 1)


@torch.no_grad()
def reference(data):
    """Independent oracle: (cross_keys, cross_values) [16, 1, 32, L, 48] fp16 from hidden_states."""

    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        h = data["hidden_states"][0]
        n = h.shape[0]
        for b in range(BLOCKS):
            p = f"vlsa.{b}"
            a = _layernorm(data, f"{p}.norm1", h)
            q = _linear_fp16(data, f"{p}.attn1.to_q", a)
            k = _linear_fp16(data, f"{p}.attn1.to_k", a)
            v = _linear_fp16(data, f"{p}.attn1.to_v", a)
            q = _heads((q.float() * QK_SCALE).half(), HEAD_DIM).float()
            k = _heads((k.float() * QK_SCALE).half(), HEAD_DIM).float()
            prob = torch.softmax(q @ k.transpose(-1, -2), dim=-1).half().float()
            o = (prob @ _heads(v, HEAD_DIM).float()).half().transpose(0, 1).reshape(n, HIDDEN)
            h = (h.float() + _linear_fp16(data, f"{p}.attn1.to_out.0", o).float()).half()
            f = _gelu_tanh(
                _linear_fp8(data, f"{p}.ff.net.0.proj", _layernorm(data, f"{p}.norm3", h))
            )
            h = (h.float() + _linear_fp8(data, f"{p}.ff.net.2", f).float()).half()
        keys, values = [], []
        for j in range(CROSS_BLOCKS):
            keys.append(_heads(_linear_fp8(data, f"xkv.{j}.to_k", h), KV_DIM))
            values.append(_heads(_linear_fp8(data, f"xkv.{j}.to_v", h), KV_DIM))
        return torch.stack(keys)[:, None], torch.stack(values)[:, None]
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
