"""verify_terminator.py - check terminator_vision.onnx against an independent PyTorch reference.

Runs the ONNX on onnxruntime (CPU EP), runs the same math as a torch.nn.Module built from the
model's metadata_props (edge_gain / edge_threshold / bg_luma), and requires
max |onnx - torch| < 1e-3. Also enforces the op whitelist and the fixed [1,3,H,W] shape the
proxy demands, and writes the stylized frame as a PNG.

Requires: pip install onnx onnxruntime torch numpy pillow
Usage:    python tools/ai/verify_terminator.py [--model config/terminator_vision.onnx]
              [--image some.png] [--out terminator_out.png]
"""
import argparse
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from PIL import Image

ALLOWED_OPS = {"Conv", "Mul", "Add", "Sqrt", "Clip", "Concat", "Slice", "Split"}
TOLERANCE = 1e-3


class TerminatorRef(torch.nn.Module):
    def __init__(self, edge_gain: float, edge_thr: float, bg_luma: float):
        super().__init__()
        self.gain, self.thr, self.bg = edge_gain, edge_thr, bg_luma
        self.register_buffer("luma", torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))
        sx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        sy = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
        self.register_buffer("sobel", torch.stack([sx, sy]).unsqueeze(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.conv2d(x, self.luma)
        g = F.conv2d(y, self.sobel, padding=1)
        mask = torch.zeros_like(y)
        mask[:, :, 1:-1, 1:-1] = 1.0
        g = g * mask
        mag = torch.sqrt((g * g).sum(dim=1, keepdim=True) + 1e-6)
        edges = torch.clamp(mag * self.gain - self.thr, 0.0, 1.0)
        r = torch.clamp(1.5 * self.bg * y + edges, 0.0, 1.0)
        gb = torch.clamp(0.7 * edges, 0.0, 1.0)
        return torch.cat([r, gb, gb], dim=1)


def synthetic(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[..., 0] = xx / w
    img[..., 1] = yy / h
    img[..., 2] = 0.5 + 0.5 * np.sin(xx * 0.05) * np.cos(yy * 0.07)
    img[h // 4:h // 2, w // 4:w // 2] = 0.9                    # hard-edged box
    cy, cx, rad = h * 0.65, w * 0.7, min(h, w) * 0.15
    img[(yy - cy) ** 2 + (xx - cx) ** 2 < rad * rad] = 0.15    # dark disc
    return img


def main() -> int:
    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(root, "config", "terminator_vision.onnx"))
    ap.add_argument("--image", default=None, help="RGB test image (resized to model res); synthetic if omitted")
    ap.add_argument("--out", default="terminator_out.png")
    a = ap.parse_args()

    model = onnx.load(a.model)
    ops = {n.op_type for n in model.graph.node}
    bad = ops - ALLOWED_OPS
    if bad:
        print(f"FAIL: non-whitelisted ops {sorted(bad)}")
        return 1
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        print("FAIL: model must have exactly one input and one output")
        return 1
    dims = [d.dim_value if d.HasField("dim_value") else -1
            for d in model.graph.input[0].type.tensor_type.shape.dim]
    if len(dims) != 4 or dims[0] != 1 or dims[1] != 3 or min(dims) <= 0:
        print(f"FAIL: input shape {dims} is not fixed [1,3,H,W]")
        return 1
    _, _, h, w = dims
    if w % 64 != 0:
        print(f"FAIL: width {w} is not a multiple of 64")
        return 1
    meta = {p.key: float(p.value) for p in model.metadata_props}
    gain = meta.get("edge_gain", 2.0)
    thr = meta.get("edge_threshold", 0.05)
    bg = meta.get("bg_luma", 0.35)

    if a.image:
        img = np.asarray(Image.open(a.image).convert("RGB").resize((w, h), Image.BILINEAR),
                         dtype=np.float32) / 255.0
    else:
        img = synthetic(h, w)
    x = np.ascontiguousarray(img.transpose(2, 0, 1)[None], dtype=np.float32)

    sess = ort.InferenceSession(a.model, providers=["CPUExecutionProvider"])
    out_onnx = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    with torch.no_grad():
        out_ref = TerminatorRef(gain, thr, bg)(torch.from_numpy(x)).numpy()

    err = float(np.max(np.abs(out_onnx - out_ref)))
    rng = (float(out_onnx.min()), float(out_onnx.max()))
    Image.fromarray((np.clip(out_onnx[0].transpose(1, 2, 0), 0, 1) * 255 + 0.5).astype(np.uint8)).save(a.out)
    print(f"model {a.model}: [1,3,{h},{w}] ops={sorted(ops)} gain={gain} thr={thr} bg={bg}")
    print(f"output range [{rng[0]:.4f}, {rng[1]:.4f}]  max|onnx-torch| = {err:.3e}  -> {a.out}")
    if err >= TOLERANCE or rng[0] < 0.0 or rng[1] > 1.0:
        print("FAIL")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
