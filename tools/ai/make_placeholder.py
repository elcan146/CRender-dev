"""make_placeholder.py - emit a fixed-shape ONNX stylization model for RetroMode::Neural.

--style art (default): pixel-art converter, not a dumb pixelate filter
    frame  [1,3,H,W] fp32 (0..1 RGB, NCHW planar)
    Kuwahara smooth  = softmin-variance blend of 4 quadrant box means   [flat regions]
    edges            = Sobel on smoothed luminance, dilated             [outlines]
    pixelate         = AveragePool(px) -> Resize(nearest, x px)
    dither           = Bayer 4x4 ordered, tiled via nearest Resize      [indexed-color feel]
    styled           = Clip( Quantize(pix + dither) * (1 - edge*ostr) )
    -> flat shaded regions, chunky blocks, dithered palette, dark outlines.

--style pixel: simple blockify + saturate + contrast + posterize (no regions/edges)
--style edge:  original bring-up look (conv laplacian + sigmoid)

--two-input (temporal, AiTemporal:2): adds 'prev' input,
    styled = 0.7 * <graph>(frame) + 0.3 * prev

All ops are DirectML-friendly (Conv/Pool/Resize/elementwise/Softmax/Slice — no Where/Less).

Requires: pip install onnx numpy
Usage:    python tools/ai/make_placeholder.py [--width 640] [--height 360] [--style art]
              [--pixels 4] [--levels 8] [--outline 0.8] [--dither 1.0] [--two-input]
"""
import argparse

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def scalar(v: float, name: str):
    return numpy_helper.from_array(np.array(v, dtype=np.float32), name)


def i64(arr, name: str):
    return numpy_helper.from_array(np.asarray(arr, dtype=np.int64), name)


def f32(arr, name: str):
    return numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name)


def node(op, inputs, outputs, **attrs):
    return helper.make_node(op, inputs, outputs, **attrs)


# ----------------------------------------------------------------------------
# shared building blocks

def pixelate_nodes(x: str, out: str, px: int, H: int, W: int, tag: str,
                   nodes, inits):
    """out = ResizeNearest(AveragePool_px(x), sizes=HxW) — chunky blocks, exact HxW
    even when px doesn't divide the dims."""
    inits.append(i64([1, 3, H, W], f"{tag}_sz"))
    nodes += [
        node("AveragePool", [x], [f"{tag}_small"],
             kernel_shape=[px, px], strides=[px, px]),
        node("Resize", [f"{tag}_small", "", "", f"{tag}_sz"], [out],
             mode="nearest", coordinate_transformation_mode="asymmetric"),
    ]


def sat_contrast_nodes(x: str, out: str, sat: float, ctr: float, tag: str, nodes, inits):
    """out = (x + (x-meanC)*(sat-1) - .5)*ctr + .5 — color pop."""
    inits += [scalar(sat - 1.0, f"{tag}_sat"),
              scalar(0.5, f"{tag}_h"), scalar(ctr, f"{tag}_c")]
    nodes += [
        node("ReduceMean", [x], [f"{tag}_g"], axes=[1], keepdims=1),
        node("Sub", [x, f"{tag}_g"], [f"{tag}_d"]),
        node("Mul", [f"{tag}_d", f"{tag}_sat"], [f"{tag}_dw"]),
        node("Add", [x, f"{tag}_dw"], [f"{tag}_s"]),
        node("Sub", [f"{tag}_s", f"{tag}_h"], [f"{tag}_c1"]),
        node("Mul", [f"{tag}_c1", f"{tag}_c"], [f"{tag}_c2"]),
        node("Add", [f"{tag}_c2", f"{tag}_h"], [out]),
    ]


def quantize_nodes(x: str, out: str, levels: float, dither_amt: float,
                   H: int, W: int, tag: str, nodes, inits):
    """out = floor(x*(L-1) + bayer + .5)/(L-1) — ordered-dither palette snap."""
    bayer = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]],
                     dtype=np.float32).reshape(1, 1, 4, 4) / 16.0
    inits += [
        f32(bayer, f"{tag}_bayer"),
        i64([1, 1, H, W], f"{tag}_bsz"),
        scalar(dither_amt, f"{tag}_da"),
        scalar(levels - 1.0, f"{tag}_lm1"),
        scalar(0.5, f"{tag}_h"),
        scalar(1.0 / (levels - 1.0), f"{tag}_il"),
        scalar(0.0, f"{tag}_z"), scalar(1.0, f"{tag}_o"),
    ]
    nodes += [
        node("Resize", [f"{tag}_bayer", "", "", f"{tag}_bsz"], [f"{tag}_b"],
             mode="nearest", coordinate_transformation_mode="asymmetric"),
        node("Mul", [f"{tag}_b", f"{tag}_da"], [f"{tag}_bw"]),
        node("Mul", [x, f"{tag}_lm1"], [f"{tag}_q1"]),
        node("Add", [f"{tag}_q1", f"{tag}_bw"], [f"{tag}_q2"]),
        node("Add", [f"{tag}_q2", f"{tag}_h"], [f"{tag}_q3"]),
        node("Floor", [f"{tag}_q3"], [f"{tag}_q4"]),
        node("Mul", [f"{tag}_q4", f"{tag}_il"], [f"{tag}_q5"]),
        node("Clip", [f"{tag}_q5", f"{tag}_z", f"{tag}_o"], [out]),
    ]


# ----------------------------------------------------------------------------
# style: art — Kuwahara regions + Sobel outlines + pixelate + dither quantize

def art_nodes(px: int, levels: float, dither: float, outline: float,
              sat: float, ctr: float, edge_gain: float, edge_bias: float,
              sharp: float, H: int, W: int):
    nodes, inits = [], []

    # --- Kuwahara: 4 quadrant 3x3 box means via shifted 5x5 convs (pads=2) ---
    quad = {"tl": (0, 0), "tr": (0, 2), "bl": (2, 0), "br": (2, 2)}
    for k, (r0, c0) in quad.items():
        w = np.zeros((3, 3, 5, 5), dtype=np.float32)
        for c in range(3):
            w[c, c, r0:r0 + 3, c0:c0 + 3] = 1.0 / 9.0
        inits.append(f32(w, f"q_{k}"))
        nodes.append(node("Conv", ["frame", f"q_{k}"], [f"m_{k}"],
                          kernel_shape=[5, 5], pads=[2, 2, 2, 2]))
    nodes.append(node("Mul", ["frame", "frame"], ["fsq"]))
    for k in quad:
        nodes.append(node("Conv", ["fsq", f"q_{k}"], [f"s_{k}"],
                          kernel_shape=[5, 5], pads=[2, 2, 2, 2]))
        # var_k = mean(x^2) - mean^2, summed over channels -> [1,1,H,W]
        nodes += [
            node("Mul", [f"m_{k}", f"m_{k}"], [f"mm_{k}"]),
            node("Sub", [f"s_{k}", f"mm_{k}"], [f"v_{k}"]),
        ]
        inits.append(i64([1], f"rax_{k}"))
        nodes.append(node("ReduceSum", [f"v_{k}", f"rax_{k}"], [f"vr_{k}"], keepdims=1))
    # softmin weights over quadrant variances
    inits.append(scalar(-sharp, "k_negsharp"))
    nodes += [
        node("Concat", [f"vr_{k}" for k in quad], ["vcat"], axis=1),
        node("Mul", ["vcat", "k_negsharp"], ["vneg"]),
        node("Softmax", ["vneg"], ["wcat"], axis=1),
    ]
    # smooth = sum_k m_k * w_k
    for i, k in enumerate(quad):
        inits += [i64([i], f"ws0_{k}"), i64([i + 1], f"ws1_{k}"),
                  i64([1], f"wax_{k}")]
        nodes += [
            node("Slice", ["wcat", f"ws0_{k}", f"ws1_{k}", f"wax_{k}"], [f"w_{k}"]),
            node("Mul", [f"m_{k}", f"w_{k}"], [f"wm_{k}"]),
        ]
    nodes += [
        node("Add", ["wm_tl", "wm_tr"], ["wa0"]),
        node("Add", ["wm_bl", "wm_br"], ["wa1"]),
        node("Add", ["wa0", "wa1"], ["smooth"]),
    ]

    # --- outlines: Sobel on smoothed luminance, dilate with MaxPool ---
    lum_w = np.array([0.299, 0.587, 0.114], dtype=np.float32).reshape(1, 3, 1, 1)
    sobx = np.array([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=np.float32).reshape(1, 1, 3, 3)
    soby = np.array([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=np.float32).reshape(1, 1, 3, 3)
    inits += [f32(lum_w, "lum_w"), f32(sobx, "sobx"), f32(soby, "soby"),
              scalar(edge_gain, "k_eg"), scalar(edge_bias, "k_eb"),
              scalar(0.0, "k_ez"), scalar(1.0, "k_eo"), scalar(outline, "k_os"),
              scalar(1.0, "k_1")]
    nodes += [
        node("Conv", ["smooth", "lum_w"], ["lum"], kernel_shape=[1, 1]),
        node("Conv", ["lum", "sobx"], ["gx"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        node("Conv", ["lum", "soby"], ["gy"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        node("Abs", ["gx"], ["agx"]),
        node("Abs", ["gy"], ["agy"]),
        node("Add", ["agx", "agy"], ["g"]),
        node("Mul", ["g", "k_eg"], ["gs"]),
        node("Sub", ["gs", "k_eb"], ["gt"]),
        node("Clip", ["gt", "k_ez", "k_eo"], ["edge"]),
        node("MaxPool", ["edge"], ["edge_d"],
             kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1]),
    ]
    # pixelate the edge mask so outlines land on block boundaries
    pixelate_nodes("edge_d", "edge_pix", px, H, W, "epx", nodes, inits)

    # --- pixelate smoothed image, then pop + dithered posterize ---
    pixelate_nodes("smooth", "pix", px, H, W, "px", nodes, inits)
    sat_contrast_nodes("pix", "pop", sat, ctr, "pop", nodes, inits)
    quantize_nodes("pop", "quant", levels, dither, H, W, "q", nodes, inits)

    # styled = quant * (1 - edge_pix * outline)
    inits += [scalar(0.0, "k_fz"), scalar(1.0, "k_fo")]
    nodes += [
        node("Mul", ["edge_pix", "k_os"], ["em"]),
        node("Sub", ["k_1", "em"], ["ink"]),
        node("Mul", ["quant", "ink"], ["inked"]),
        node("Clip", ["inked", "k_fz", "k_fo"], ["styled"]),
    ]
    return nodes, inits


# ----------------------------------------------------------------------------
# style: pixel — simple blockify + pop + posterize (no regions/outlines)

def pixel_nodes(px: int, levels: float, dither: float, sat: float, ctr: float,
                H: int, W: int):
    nodes, inits = [], []
    pixelate_nodes("frame", "pix", px, H, W, "px", nodes, inits)
    sat_contrast_nodes("pix", "pop", sat, ctr, "pop", nodes, inits)
    quantize_nodes("pop", "styled", levels, dither, H, W, "q", nodes, inits)
    return nodes, inits


# ----------------------------------------------------------------------------
# style: edge — original conv+sigmoid bring-up look

def edge_nodes(strength: float):
    lap = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
    ident = np.zeros((3, 3), dtype=np.float32)
    ident[1, 1] = 1.0
    k = ident + 2.0 * lap
    w = np.zeros((3, 3, 3, 3), dtype=np.float32)
    for c in range(3):
        w[c, c] = strength * k
    b = np.full((3,), -0.5 * strength, dtype=np.float32)
    inits = [f32(w, "conv_w"), f32(b, "conv_b")]
    nodes = [
        node("Conv", ["frame", "conv_w", "conv_b"], ["conv"],
             kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        node("Sigmoid", ["conv"], ["styled"]),
    ]
    return nodes, inits


# ----------------------------------------------------------------------------

def build(width: int, height: int, two_input: bool, a) -> onnx.ModelProto:
    if a.style == "art":
        nodes, inits = art_nodes(a.pixels, a.levels, a.dither, a.outline,
                                 a.saturation, a.contrast, a.edge_gain,
                                 a.edge_bias, a.sharp, a.height, a.width)
    elif a.style == "pixel":
        nodes, inits = pixel_nodes(a.pixels, a.levels, a.dither,
                                   a.saturation, a.contrast, a.height, a.width)
    else:
        nodes, inits = edge_nodes(a.strength)

    shape = [1, 3, height, width]
    inputs = [helper.make_tensor_value_info("frame", TensorProto.FLOAT, shape)]
    if two_input:
        inputs.append(helper.make_tensor_value_info("prev", TensorProto.FLOAT, shape))
        inits += [scalar(0.7, "k_cur"), scalar(0.3, "k_prev")]
        nodes[-1].output[0] = "raw"  # reroute last node into the temporal blend
        nodes += [
            node("Mul", ["raw", "k_cur"], ["cur_w"]),
            node("Mul", ["prev", "k_prev"], ["prev_w"]),
            node("Add", ["cur_w", "prev_w"], ["styled"]),
        ]

    outputs = [helper.make_tensor_value_info("styled", TensorProto.FLOAT, shape)]
    graph = helper.make_graph(nodes, f"retro_{a.style}", inputs, outputs, inits)
    model = helper.make_model(graph, producer_name="CRender",
                              opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8  # loadable by every ORT release that still ships win-x86
    onnx.checker.check_model(model)
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--style", choices=["art", "pixel", "edge"], default="art")
    ap.add_argument("--pixels", type=int, default=4, help="pixel-art block size (px)")
    ap.add_argument("--levels", type=float, default=8.0, help="posterize levels per channel")
    ap.add_argument("--dither", type=float, default=1.0, help="ordered Bayer dither amount")
    ap.add_argument("--outline", type=float, default=0.8, help="edge darkening 0..1")
    ap.add_argument("--edge-gain", type=float, default=8.0, help="Sobel response gain")
    ap.add_argument("--edge-bias", type=float, default=0.5, help="Sobel threshold")
    ap.add_argument("--sharp", type=float, default=50.0, help="Kuwahara softmin sharpness")
    ap.add_argument("--saturation", type=float, default=1.35)
    ap.add_argument("--contrast", type=float, default=1.1)
    ap.add_argument("--strength", type=float, default=6.0, help="edge style sigmoid gain")
    ap.add_argument("--two-input", action="store_true", help="temporal variant with 'prev' input")
    ap.add_argument("-o", "--output", default=None)
    a = ap.parse_args()
    out = a.output or ("retro_ai_temporal.onnx" if a.two_input else "retro_ai.onnx")
    onnx.save(build(a.width, a.height, a.two_input, a), out)
    print(f"wrote {out}  ({'frame,prev' if a.two_input else 'frame'} -> styled  "
          f"[1,3,{a.height},{a.width}] style={a.style})")


if __name__ == "__main__":
    main()
