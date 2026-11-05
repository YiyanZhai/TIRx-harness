#!/usr/bin/env python3
"""TensorRT timing baseline and oracle for the GR00T N1.7 action-head denoising loop.

One frame's loop, as the deployed runner (GR00T OptimizedActionHeadRunner._execute) runs it after
the backbone: actions start as the frame's Gaussian noise, and each of 4 steps runs
    features    = action_encoder(actions, timestep_embedding[step])            [1, 40, 1536]
    sa_embs     = cat(state_features, features + position_embedding)          [1, 41, 1536]
    model_out   = DiT(sa_embs, cross K/V, masks, adaln / output modulation of the step)
    prediction  = action_decoder(model_out)                                    [1, 41, 132]
    actions    += dt * prediction[:, 1:]
The task's outputs are the final actions and every step's DiT output (model_outputs[step]): the
final actions barely depend on the state token (it reaches the action rows only through
attention), so the DiT outputs are what the gate checks per step.
The action encoder / decoder are the M1 task's (fp16), the DiT is the M2 task's (32 NVFP4 W4A4
blocks); their oracles and TensorRT engines are reused from those tasks.

The baseline is the deployed sequence: the three TensorRT 10.16 engines with the runner's PyTorch
glue, 4 steps captured in one CUDA graph. The oracle chains the M1 task's float64 encoder and
decoder emulation with the M2 task's DiT emulation, with fp16 storage where the exported graphs
store their tensors.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from ..groot_ah_boundary_thor import baseline as ah
from ..groot_dit_step_thor import baseline as dit

SUBDIR = "groot-n17-libero-c235"
STEPS = 4
DT = ah.DT
EMBODIMENT = ah.EMBODIMENT
INPUTS = (  # new values on every call in deployment
    "initial_actions",
    "state_features",
    "cross_keys",
    "cross_values",
    "image_mask",
    "backbone_attention_mask",
    "embodiment_id",
)
CONSTANTS = ("timestep_embeddings", "adaln_modulations", "output_modulations", "position_embedding")
OUTPUTS = ("actions_out", "model_outputs")
FRAME = ("cross_keys", "cross_values", "image_mask", "backbone_attention_mask")


def vl_tokens(row):
    """The row's VL token count (128 image tokens + the instruction's text tokens)."""

    return int(row[3:]) if row.startswith("len") else 156


def row_tensor(golden, row, name):
    """One row's tensor: a len row's own `<row>.<name>`; otherwise the first rows' `<row>` or `<name>`."""

    if f"{row}.{name}" in golden:
        return golden[f"{row}.{name}"]
    return golden[row] if name == "initial_actions" else golden[name]


def model_dir() -> Path:
    """The provisioned model directory (no download source exists)."""

    root = Path(os.environ.get("HF_MODEL_PATH", "/raid/catalyst/models")).expanduser()
    path = root / SUBDIR
    if not all((path / name).is_file() for name in ("loop_manifest.json", "loop_lens_manifest.json")):
        raise FileNotFoundError(
            f"{path} has no denoising-loop files; build them with groot-perf-diagnosis "
            "kda/tirx/m1_prepare.py, m2_prepare.py, m5_prepare.py and m5_lens.py and point "
            "HF_MODEL_PATH at the directory's parent"
        )
    return path


def load_model(device):
    """(action-head and DiT weights on `device`, loop inputs on the CPU, DiT block types, rows)."""

    from safetensors.torch import load_file

    path = model_dir()
    weights = {k: v.to(device) for k, v in load_file(str(path / "weights.safetensors")).items()}
    weights.update(
        {k: v.to(device) for k, v in load_file(str(path / "dit_weights.safetensors")).items()}
    )
    golden = load_file(str(path / "loop_golden.safetensors"))
    golden.update(load_file(str(path / "loop_lens.safetensors")))
    block_types = json.loads((path / "dit_manifest.json").read_text())["block_types"]
    rows = json.loads((path / "loop_manifest.json").read_text())["noise_rows"]
    rows = [*rows, *json.loads((path / "loop_lens_manifest.json").read_text())["len_rows"]]
    return weights, golden, block_types, rows


def weight_names(block_types):
    return [*ah.WEIGHT_NAMES, *dit.weight_names(block_types)]


class TrtLoop:
    """The deployed loop on the task's data dict; launches on the current stream."""

    def __init__(self, data):
        trt = dit._tensorrt()
        path = model_dir()
        d = self.d = data
        dev = d["initial_actions"].device
        self.features = torch.empty(1, 40, 1536, dtype=torch.float16, device=dev)
        self.shifted = torch.empty_like(self.features)
        self.sa_embs = torch.empty(1, 41, 1536, dtype=torch.float16, device=dev)
        self.prediction = torch.empty(1, 41, 132, dtype=torch.float16, device=dev)
        self.encoder = ah._Engine(
            path / "action_encoder.engine",
            {
                "actions": d["actions_out"],
                "timestep_embedding": d["timestep_embeddings"][0],
                "embodiment_id": d["embodiment_id"],
            },
            self.features,
        )
        self.decoder = ah._Engine(
            path / "action_decoder.engine",
            {"model_output": d["model_outputs"][0], "embodiment_id": d["embodiment_id"]},
            self.prediction,
        )
        blob = (path / "dit_nvfp4.engine").read_bytes()
        self.dit = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(blob)
        if self.dit is None:
            raise RuntimeError("cannot deserialize dit_nvfp4.engine")
        self.dit_ctx = self.dit.create_execution_context()
        bound = {
            "sa_embs": self.sa_embs,
            **{name: d[name] for name in FRAME},
            "adaln_modulations": d["adaln_modulations"][0],
            "output_modulation": d["output_modulations"][0],
        }
        for name, tensor in bound.items():
            self.dit_ctx.set_input_shape(name, tuple(tensor.shape))
            self.dit_ctx.set_tensor_address(name, tensor.data_ptr())
        self.dit_ctx.set_tensor_address("output", d["model_outputs"][0].data_ptr())

    def __call__(self):
        d = self.d
        stream = torch.cuda.current_stream().cuda_stream
        actions = d["actions_out"]
        actions.copy_(d["initial_actions"])
        for step in range(STEPS):
            # The runner passes the step's slice of each constant; the bindings move with it.
            self.encoder.context.set_tensor_address(
                "timestep_embedding", d["timestep_embeddings"][step].data_ptr()
            )
            self.encoder.enqueue()
            torch.add(self.features, d["position_embedding"], out=self.shifted)
            torch.cat((d["state_features"], self.shifted), dim=1, out=self.sa_embs)
            self.dit_ctx.set_tensor_address(
                "adaln_modulations", d["adaln_modulations"][step].data_ptr()
            )
            self.dit_ctx.set_tensor_address(
                "output_modulation", d["output_modulations"][step].data_ptr()
            )
            # The DiT writes the step's model_outputs row, and the decoder reads it there.
            out = d["model_outputs"][step].data_ptr()
            self.dit_ctx.set_tensor_address("output", out)
            if not self.dit_ctx.execute_async_v3(stream):
                raise RuntimeError("TensorRT enqueue failed")
            self.decoder.context.set_tensor_address("model_output", out)
            self.decoder.enqueue()
            actions.add_(self.prediction[:, -40:], alpha=DT)
        return actions, d["model_outputs"]


def _f16(t):
    return t.to(torch.float16).double()


def _linear(data, t, w, b):
    return t @ data[w].double().T + data[b].double()


def encode(data, actions16, tau16, guard):
    """sa_embs [1, 41, 1536] fp16: the M1 oracle's action encoder, position embedding and concat."""

    a = _f16(_linear(data, actions16.double()[0], "W1", "b1"))
    z = _f16(_linear(data, torch.cat((a, tau16.double()[0]), dim=1), "W2", "b2"))
    h2 = _f16(z * torch.sigmoid(z))
    features = _f16(_linear(data, h2, "W3", "b3") + guard)
    shifted = _f16(features + data["position_embedding"].double()[0])
    return torch.cat((data["state_features"].double()[0], shifted), dim=0).half()[None]


def decode_update(data, actions16, model_out16, guard):
    """actions + dt * prediction[1:] in fp16: the M1 oracle's action decoder and Euler update."""

    h1 = _f16(torch.relu(_linear(data, model_out16.double()[0], "L1", "c1")))
    prediction = _f16(_linear(data, h1, "L2", "c2") + guard)
    return _f16(actions16.double()[0] + DT * prediction[1:]).half()[None]


@torch.no_grad()
def reference(data, block_types):
    """Independent oracle: (final actions [1, 40, 132], model_outputs [4, 1, 41, 1024]) fp16."""

    guard = 0.0 if int(data["embodiment_id"][0]) == EMBODIMENT else float("nan")
    actions = data["initial_actions"]
    step_data = dict(data)
    model_outputs = []
    for step in range(STEPS):
        step_data["sa_embs"] = encode(data, actions, data["timestep_embeddings"][step], guard)
        step_data["adaln_modulations"] = data["adaln_modulations"][step]
        step_data["output_modulation"] = data["output_modulations"][step]
        model_outputs.append(dit.reference(step_data, block_types))
        actions = decode_update(data, actions, model_outputs[-1], guard)
    return actions, torch.stack(model_outputs)
