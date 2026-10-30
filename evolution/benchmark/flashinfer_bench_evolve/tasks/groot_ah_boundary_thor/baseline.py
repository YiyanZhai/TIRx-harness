#!/usr/bin/env python3
"""TensorRT timing baseline and oracle for one GR00T N1.7 action-head step boundary.

The boundary is the work between two DiT denoising steps of the GR00T N1.7 action head
(batch 1, embodiment 2):

    prediction  = decoder(x)                                x: DiT output [1, 41, 1024]
    actions_out = actions_in + dt * prediction[:, 1:]       Euler update, dt = 0.25
    features    = encoder(actions_out, timestep_embedding)  the next step's embedding
    sa_embs     = cat(state_features, features + position_embedding)

The baseline is the deployed pipeline's own sequence: the TensorRT 10.16 action-decoder and
action-encoder engines with the PyTorch glue of the GR00T optimized runtime, captured in one
CUDA graph. The oracle is an independent float64 implementation of the exported graph, with
fp16 storage at every layer output.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

SUBDIR = "groot-n17-libero-c235"
DT = 0.25
EMBODIMENT = 2
WEIGHT_NAMES = ("L1", "c1", "L2", "c2", "W1", "b1", "W2", "b2", "W3", "b3")


def model_dir() -> Path:
    """The provisioned model directory: engines, weights, golden inputs (no download source)."""

    root = Path(os.environ.get("HF_MODEL_PATH", "/raid/catalyst/models")).expanduser()
    path = root / SUBDIR
    if not (path / "manifest.json").is_file():
        raise FileNotFoundError(
            f"{path} is not provisioned; build it with groot-perf-diagnosis kda/tirx/m1_prepare.py "
            "and point HF_MODEL_PATH at its parent"
        )
    return path


def load_model(device):
    """(weights on `device`, golden tensors on the CPU)."""

    from safetensors.torch import load_file

    path = model_dir()
    weights = {k: v.to(device) for k, v in load_file(str(path / "weights.safetensors")).items()}
    golden = load_file(str(path / "golden.safetensors"))
    return weights, golden


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
    """One deserialized engine with fixed bindings; enqueue() launches on the current stream."""

    def __init__(self, path, inputs, output):
        trt = _tensorrt()
        engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(
            path.read_bytes()
        )
        if engine is None:
            raise RuntimeError(f"cannot deserialize {path}")
        self.engine, self.context = engine, engine.create_execution_context()
        for name, tensor in inputs.items():
            self.context.set_input_shape(name, tuple(tensor.shape))
            self.context.set_tensor_address(name, tensor.data_ptr())
        self.context.set_tensor_address("output", output.data_ptr())
        self.keep = (inputs, output)

    def enqueue(self):
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT enqueue failed")


class TrtBoundary:
    """The deployed sequence (GR00T OptimizedActionHeadRunner._execute) on the task's data dict."""

    def __init__(self, data):
        path = model_dir()
        device = data["x"].device
        self.d = data
        self.prediction = torch.empty(1, 41, 132, dtype=torch.float16, device=device)
        self.features = torch.empty(1, 40, 1536, dtype=torch.float16, device=device)
        self.shifted = torch.empty_like(self.features)
        self.decoder = _Engine(
            path / "action_decoder.engine",
            {"model_output": data["x"], "embodiment_id": data["embodiment_id"]},
            self.prediction,
        )
        self.encoder = _Engine(
            path / "action_encoder.engine",
            {
                "actions": data["actions_out"],
                "timestep_embedding": data["timestep_embedding"],
                "embodiment_id": data["embodiment_id"],
            },
            self.features,
        )

    def __call__(self):
        d = self.d
        self.decoder.enqueue()
        # actions.add_(prediction[:, -40:], alpha=dt), written to actions_out (may alias actions_in)
        torch.add(d["actions_in"], self.prediction[:, -40:], alpha=DT, out=d["actions_out"])
        self.encoder.enqueue()
        torch.add(self.features, d["position_embedding"], out=self.shifted)
        torch.cat((d["state_features"], self.shifted), dim=1, out=d["sa_embs"])
        return d["actions_out"], d["sa_embs"]


def prepare(data):
    return TrtBoundary(data)


def run_prepared(runner):
    return runner()


@torch.no_grad()
def reference(data):
    """Independent oracle: (actions_out, sa_embs) from the inputs and weights in `data`.

    float64 arithmetic (independent of TF32 settings); fp16 storage after every layer, where
    the exported graph stores its activations; NaN decoder and encoder outputs unless
    embodiment_id equals the weights' embodiment.
    """

    def f16(t):
        return t.to(torch.float16).double()

    def linear(t, w, b):
        return t @ data[w].double().T + data[b].double()

    guard = 0.0 if int(data["embodiment_id"][0]) == EMBODIMENT else float("nan")
    x = data["x"].double()[0]
    h1 = f16(torch.relu(linear(x, "L1", "c1")))
    prediction = f16(linear(h1, "L2", "c2") + guard)
    actions = f16(data["actions_in"].double()[0] + DT * prediction[1:])
    a = f16(linear(actions, "W1", "b1"))
    tau = data["timestep_embedding"].double()[0]
    z = f16(linear(torch.cat((a, tau), dim=1), "W2", "b2"))
    h2 = f16(z * torch.sigmoid(z))
    features = f16(linear(h2, "W3", "b3") + guard)
    shifted = f16(features + data["position_embedding"].double()[0])
    sa_embs = torch.cat((data["state_features"].double()[0], shifted), dim=0)
    return actions.half().view(1, 40, 132), sa_embs.half().view(1, 41, 1536)
