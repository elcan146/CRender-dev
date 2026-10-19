"""wrap_style.py - repackage a fixed-shape ONNX style model for RetroMode::Neural.

Wraps a model like fast_neural_style (zoo models expect [1,3,224,224] fp32 in 0..255)
into RetroIntercept's contract: input 'frame' / output 'styled', [1,3,H,W] fp32, 0..1.

    styled = Clip(Resize(Core(Resize(frame -> in) *255) -> out) /255, 0, 1)

- inner resize: bilinear down to the model's declared input HxW
- outer resize: bilinear back up to --width/--height (must match AiWidth/AiHeight config)
- model weights promoted from graph inputs to initializers when needed (IR<=3 style)

Requires: pip install onnx
Usage:    python tools/ai/wrap_style.py model.onnx [--width 640] [--height 360]
              [--in-scale 255] [--out-scale 255] [-o retro_ai.onnx]
"""
import argparse
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, version_converter


def promote_weights(model: onnx.ModelProto) -> None:
    """IR<=3 convention: initializers listed as graph inputs. Keep only real inputs."""
    init_names = {i.name for i in model.graph.initializer}
    keep = [i for i in model.graph.input if i.name not in init_names]
    del model.graph.input[:]
    model.graph.input.extend(keep)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="fixed-shape ONNX style model")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--in-scale", type=float, default=255.0, help="0..1 -> model range")
    ap.add_argument("--out-scale", type=float, default=255.0, help="model range -> 0..1")
    ap.add_argument("--target-opset", type=int, default=17)
    ap.add_argument("-o", "--output", default="retro_ai.onnx")
    a = ap.parse_args()

    core = onnx.load(a.model)
    if max(o.version for o in core.opset_import) < a.target_opset:
        core = version_converter.convert_version(core, a.target_opset)
    promote_weights(core)

    in_vi = core.graph.input[0]
    dims = [d.dim_value for d in in_vi.type.tensor_type.shape.dim]
    if len(dims) != 4 or not all(d > 0 for d in dims):
        sys.exit(f"unsupported input shape {[d.dim_param or d.dim_value for d in in_vi.type.tensor_type.shape.dim]}")
    _n, _c, ih, iw = dims
    in_name, out_name = in_vi.name, core.graph.output[0].name
    print(f"core: {in_name} [{_n},{_c},{ih},{iw}] -> {out_name}")

    W, H = a.width, a.height
    nodes = [
        helper.make_node("Mul", ["frame", "k_in"], ["x255"]),
        helper.make_node("Resize", ["x255", "", "", "sz_in"], ["core_in"],
                         mode="linear", coordinate_transformation_mode="asymmetric"),
    ]
    for n in core.graph.node:
        nodes.append(n)  # names preserved; we feed its input via 'core_in' below
    nodes += [
        helper.make_node("Resize", ["core_out", "", "", "sz_out"], ["y255"],
                         mode="linear", coordinate_transformation_mode="asymmetric"),
        helper.make_node("Mul", ["y255", "k_out"], ["y01"]),
        helper.make_node("Clip", ["y01", "k_zero", "k_one"], ["styled"]),
    ]
    # rewire: the core's io names become graph-internal tensors
    for n in nodes:
        for i, s in enumerate(n.input):
            if s == in_name:
                n.input[i] = "core_in"
        for i, s in enumerate(n.output):
            if s == out_name:
                n.output[i] = "core_out"

    inits = list(core.graph.initializer) + [
        numpy_helper.from_array(np.array(a.in_scale, dtype=np.float32), "k_in"),
        numpy_helper.from_array(np.array(1.0 / a.out_scale, dtype=np.float32), "k_out"),
        numpy_helper.from_array(np.array([1, 3, ih, iw], dtype=np.int64), "sz_in"),
        numpy_helper.from_array(np.array([1, 3, H, W], dtype=np.int64), "sz_out"),
        numpy_helper.from_array(np.array(0.0, dtype=np.float32), "k_zero"),
        numpy_helper.from_array(np.array(1.0, dtype=np.float32), "k_one"),
    ]

    shape = [1, 3, H, W]
    graph = helper.make_graph(nodes, f"wrapped_{core.graph.name or 'style'}",
                              [helper.make_tensor_value_info("frame", TensorProto.FLOAT, shape)],
                              [helper.make_tensor_value_info("styled", TensorProto.FLOAT, shape)],
                              inits)
    model = helper.make_model(graph, producer_name="RetroIntercept-wrap",
                              opset_imports=[helper.make_opsetid("", a.target_opset)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, a.output)
    print(f"wrote {a.output}  frame[1,3,{H},{W}] -> styled, core {iw}x{ih}")


if __name__ == "__main__":
    main()
