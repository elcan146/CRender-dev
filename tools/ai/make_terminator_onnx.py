"""make_terminator_onnx.py - emit the fixed-shape "Terminator Vision" model for RetroMode::Terminator.

    frame  [1,3,H,W] fp32 (0..1 RGB, NCHW planar)
    Y      = Conv1x1(frame, [.299 .587 .114])                         [luminance]
    gxy    = Conv3x3(Y, [Sobel_x; Sobel_y], zero pad) * border_mask    [2ch gradients]
    G      = Sqrt(Conv1x1(gxy*gxy, [1 1]) + 1e-6)                      [magnitude]
    edges  = Clip(G*EDGE_GAIN - EDGE_THRESHOLD, 0, 1)                  [soft noise floor]
    R      = Clip(1.5*BG_LUMA*Y + edges, 0, 1)
    G = B  = Clip(0.7*edges, 0, 1)
    styled = Concat(R, G, B)  -> crimson void, white/pink outlines.

ONNX Conv only zero-pads and Pad(reflect) is outside the op budget, so the 1px border of
the gradient is multiplied by a constant 0/1 mask instead - no false edges at the frame rim.

Ops used: Conv, Mul, Add, Sqrt, Clip, Concat only (DirectML-friendly, no control flow,
no dynamic axes). Export params are stored in metadata_props so verify_terminator.py can
rebuild the exact reference.

Requires: pip install onnx numpy
Usage:    python tools/ai/make_terminator_onnx.py [--width 640] [--height 360]
              [--edge-gain 2.0] [--edge-threshold 0.05] [--bg-luma 0.35] [-o out.onnx]
"""
import argparse
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

LUMA = (0.299, 0.587, 0.114)
SOBEL_X = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
SOBEL_Y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
EPS = 1e-6
EDGE_TO_GB = 0.7
BG_TO_R = 1.5


def scalar(v: float, name: str):
    return numpy_helper.from_array(np.array(v, dtype=np.float32), name)


def f32(arr, name: str):
    return numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name)


def node(op, inputs, outputs, **attrs):
    return helper.make_node(op, inputs, outputs, **attrs)


def border_mask(h: int, w: int) -> np.ndarray:
    m = np.ones((1, 1, h, w), dtype=np.float32)
    m[:, :, 0, :] = 0.0
    m[:, :, -1, :] = 0.0
    m[:, :, :, 0] = 0.0
    m[:, :, :, -1] = 0.0
    return m


def build(width: int, height: int, edge_gain: float, edge_thr: float, bg_luma: float) -> onnx.ModelProto:
    sobel = np.array([SOBEL_X, SOBEL_Y], dtype=np.float32).reshape(2, 1, 3, 3)
    inits = [
        f32(np.array(LUMA, dtype=np.float32).reshape(1, 3, 1, 1), "w_luma"),
        f32(sobel, "w_sobel"),
        f32(border_mask(height, width), "k_mask"),
        f32(np.ones((1, 2, 1, 1), dtype=np.float32), "w_sum2"),
        scalar(EPS, "k_eps"),
        scalar(edge_gain, "k_gain"),
        scalar(-edge_thr, "k_negthr"),
        scalar(BG_TO_R * bg_luma, "k_bg"),
        scalar(EDGE_TO_GB, "k_gb"),
        scalar(0.0, "k_0"),
        scalar(1.0, "k_1"),
    ]
    nodes = [
        node("Conv", ["frame", "w_luma"], ["y"], kernel_shape=[1, 1]),
        node("Conv", ["y", "w_sobel"], ["gxy_raw"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        node("Mul", ["gxy_raw", "k_mask"], ["gxy"]),
        node("Mul", ["gxy", "gxy"], ["gxy2"]),
        node("Conv", ["gxy2", "w_sum2"], ["g2"], kernel_shape=[1, 1]),
        node("Add", ["g2", "k_eps"], ["g2e"]),
        node("Sqrt", ["g2e"], ["g"]),
        node("Mul", ["g", "k_gain"], ["gg"]),
        node("Add", ["gg", "k_negthr"], ["gt"]),
        node("Clip", ["gt", "k_0", "k_1"], ["edges"]),
        node("Mul", ["y", "k_bg"], ["ydim"]),
        node("Add", ["ydim", "edges"], ["r_raw"]),
        node("Clip", ["r_raw", "k_0", "k_1"], ["r"]),
        node("Mul", ["edges", "k_gb"], ["gb_raw"]),
        node("Clip", ["gb_raw", "k_0", "k_1"], ["gb"]),
        node("Concat", ["r", "gb", "gb"], ["styled"], axis=1),
    ]
    shape = [1, 3, height, width]
    graph = helper.make_graph(
        nodes, "terminator_vision",
        [helper.make_tensor_value_info("frame", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("styled", TensorProto.FLOAT, shape)],
        inits)
    model = helper.make_model(graph, producer_name="CRender",
                              opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8  # loadable by ORT 1.22.1 (last win-x86 DirectML release)
    for k, v in (("edge_gain", edge_gain), ("edge_threshold", edge_thr), ("bg_luma", bg_luma)):
        e = model.metadata_props.add()
        e.key, e.value = k, repr(float(v))
    onnx.checker.check_model(model)
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=640, help="model width (multiple of 64)")
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--edge-gain", type=float, default=2.0, help="Sobel magnitude gain (EDGE_GAIN)")
    ap.add_argument("--edge-threshold", type=float, default=0.05, help="soft noise floor (EDGE_THRESHOLD)")
    ap.add_argument("--bg-luma", type=float, default=0.35, help="world luminance scale in red (BG_LUMA)")
    ap.add_argument("-o", "--output", default=None)
    a = ap.parse_args()
    if a.width % 64 != 0 or a.width <= 0 or a.height <= 2:
        ap.error("--width must be a positive multiple of 64 (D3D12 copy row pitch), --height > 2")
    out = a.output or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "..", "config", "terminator_vision.onnx")
    out = os.path.normpath(out)
    onnx.save(build(a.width, a.height, a.edge_gain, a.edge_threshold, a.bg_luma), out)
    print(f"wrote {out}  (frame -> styled [1,3,{a.height},{a.width}] gain={a.edge_gain} "
          f"thr={a.edge_threshold} bg={a.bg_luma})")


if __name__ == "__main__":
    main()
