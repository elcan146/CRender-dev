"""render_model_graph.py — render an ONNX model's op graph as a Netron-style SVG/PNG.

Netron look: each node is a rounded "pill" — a dark header bar with the op name
(white text) stacked over white attribute rows (dtype/shape of folded initializers
and Constant inputs, plus node attributes like kernel_shape/pads/axis). Initializer
and few-consumer Constant nodes are folded INTO the consuming node's attribute
rows instead of appearing as graph nodes. Layout is a vertical layered DAG
(longest-path layering + barycenter ordering).

Text is measured with the real fonts (Segoe UI / Consolas via PIL) so node boxes
always fit their contents; rows too wide for the box are ellipsized.

Colors match Netron: black default, brown #59423b for tensor/transform ops,
dark red #702921 for activations, #eee for graph inputs/outputs.

Requires: pip install onnx matplotlib (Pillow optional — used for exact text metrics)
Usage:
    python tools/ai/render_model_graph.py config/terminator_vision.onnx
    python tools/ai/render_model_graph.py --all config docs/images
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, PathPatch
from matplotlib.path import Path
import numpy as np
import onnx
from onnx import numpy_helper

# ---- Netron-ish palette -------------------------------------------------------
HDR_DEFAULT = "#000000"
HDR_TENSOR  = "#59423b"    # Concat/Split/Slice/Transpose/Reshape/Gather/...
HDR_ACT     = "#702921"    # Sigmoid/Relu/Clip/Softmax/...
HDR_POOL    = "#394240"    # pools/reductions
IO_FILL     = "#eeeeee"    # graph inputs / outputs
BORDER      = "#333333"
EDGE_COLOR  = "#b0b0b0"
ATTR_TEXT   = "#333333"
FONT        = "Segoe UI"

TENSOR_OPS = {
    "Concat", "Split", "Slice", "Transpose", "Reshape", "Gather", "Unsqueeze",
    "Squeeze", "Shape", "Tile", "Expand", "Cast", "DepthToSpace", "SpaceToDepth",
    "Flatten", "Pad", "Upsample", "Resize", "Scatter", "ScatterND", "Where",
    "NonZero", "TopK", "GatherND", "Identity",
}
ACT_OPS = {
    "Sigmoid", "Relu", "LeakyRelu", "Clip", "Softmax", "Tanh", "Elu", "Selu",
    "Gelu", "HardSigmoid", "HardSwish", "Mish", "SiLU", "Softplus", "Softsign",
    "PRelu", "ThresholdedRelu", "LogSoftmax",
}
POOL_OPS = {
    "MaxPool", "AveragePool", "GlobalAveragePool", "GlobalMaxPool", "LpPool",
    "ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin", "ReduceProd", "ReduceL1",
    "ReduceL2", "ReduceSumSquare",
}

NODE_W_PAD = 24            # horizontal padding inside a node (px-equivalent units)
HDR_H = 18
ROW_H = 15
NODE_GAP_X = 26
LAYER_GAP_Y = 26           # vertical gap between layers
MAX_NODE_W = 340           # hard cap on node box width


def _fmt_num(v):
    if isinstance(v, float) and v == int(v) and abs(v) < 1e15:
        return str(int(v))
    return f"{v:g}"


def tensor_summary(arr):
    """'float32[32,3,3,3]' plus '= vals' for tiny tensors (Netron style)."""
    shape = "".join(f"[{d}]" for d in arr.shape)
    base = f"{arr.dtype}{shape}"
    if arr.size <= 8:
        vals = ", ".join(_fmt_num(float(v)) for v in arr.reshape(-1))
        return f"{base} = {vals}"
    return base


def attr_str(attr):
    """Render one onnx.AttributeProto concisely."""
    if attr.HasField("f"):
        return f"{attr.name}: {_fmt_num(attr.f)}"
    if attr.HasField("i"):
        return f"{attr.name}: {attr.i}"
    if attr.HasField("s"):
        s = attr.s.decode("utf-8", "replace")
        return f"{attr.name}: '{s}'"
    if attr.HasField("t"):
        return f"{attr.name}: {tensor_summary(numpy_helper.to_array(attr.t))}"
    if attr.floats:
        return f"{attr.name}: [{', '.join(_fmt_num(v) for v in attr.floats)}]"
    if attr.ints:
        return f"{attr.name}: [{', '.join(str(i) for i in attr.ints)}]"
    return attr.name


def hdr_color(op):
    if op in ACT_OPS:
        return HDR_ACT
    if op in TENSOR_OPS:
        return HDR_TENSOR
    if op in POOL_OPS:
        return HDR_POOL
    return HDR_DEFAULT


# ---- text measurement ---------------------------------------------------------
_FONTS = None

def _fonts():
    """Load the real TTFs so text can be measured exactly (px @ ~100 dpi scale)."""
    global _FONTS
    if _FONTS is None:
        _FONTS = {}
        try:
            from PIL import ImageFont
            from matplotlib import font_manager
            from matplotlib.font_manager import FontProperties
            # resolve the SAME ttf matplotlib will draw with, then over-measure
            # ~8-10% so the box keeps right-side headroom
            specs = {
                "hdr":  (FontProperties(family="Segoe UI", weight="bold"), 15),
                "io":   (FontProperties(family="Segoe UI"), 13),
                "attr": (FontProperties(family="Consolas"), 11),
            }
            for kind, (fp, px) in specs.items():
                try:
                    path = font_manager.findfont(fp, fallback_to_default=True)
                    _FONTS[kind] = ImageFont.truetype(path, px)
                except Exception:
                    _FONTS[kind] = None
        except ImportError:
            _FONTS = {"hdr": None, "io": None, "attr": None}
    return _FONTS


_MEAS = {}

def measure(s, kind):
    """True rendered width in px for the font used by that node part."""
    key = (kind, s)
    if key not in _MEAS:
        f = _fonts().get(kind)
        if f is not None:
            _MEAS[key] = f.getlength(s)
        else:  # heuristic fallback if PIL/fonts unavailable
            pt = {"hdr": 10, "io": 8.5, "attr": 7.5}[kind]
            w = sum(0.34 if c in "il.,:;|!'" else 0.62 if c.isupper() else 0.52
                    for c in s)
            _MEAS[key] = w * pt * 100.0 / 72.0
    return _MEAS[key]


def fit_row(s, max_w):
    """Truncate s with a trailing ellipsis until it fits max_w px."""
    if measure(s, "attr") <= max_w:
        return s
    lo, hi = 0, len(s)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if measure(s[:mid] + "…", "attr") <= max_w:
            lo = mid
        else:
            hi = mid - 1
    return s[:lo] + "…"


def dims_str(tensor):
    dims = []
    try:
        for d in tensor.type.tensor_type.shape.dim:
            dims.append(str(d.dim_value) if d.dim_value else (d.dim_param or "?"))
    except Exception:
        pass
    return "x".join(dims)


# ---- graph building -----------------------------------------------------------
def build(model):
    """Fold initializers/Constants into consumer attr rows.
    Returns nodes[{label,header,rows,kind}], edges."""
    g = model.graph
    init_map = {i.name: i for i in g.initializer}          # name -> TensorProto

    def val_for(name):
        t = init_map.get(name)
        return tensor_summary(numpy_helper.to_array(t)) if t is not None else None

    # Constant nodes: output name -> display value (fold into consumer)
    const_val = {}
    const_node = {}
    for idx, n in enumerate(g.node):
        if n.op_type == "Constant":
            txt = "const"
            for a in n.attribute:
                if a.name == "value" and a.HasField("t"):
                    txt = tensor_summary(numpy_helper.to_array(a.t))
            for o in n.output:
                const_val[o] = txt
                const_node[o] = idx
    const_consumers = {}
    for idx, n in enumerate(g.node):
        for inp in n.input:
            if inp in const_node:
                const_consumers.setdefault(inp, []).append(idx)
    folded_consts = {const_node[t] for t, cons in const_consumers.items()
                     if len(cons) <= 4}

    nodes, edges = [], set()
    producer = {}
    in_idx = {}

    # graph inputs (excluding weights)
    for vi in g.input:
        if vi.name in init_map:
            continue
        in_idx[vi.name] = len(nodes)
        d = dims_str(vi)
        nodes.append({"label": vi.name.split("/")[-1] or vi.name,
                      "header": "input",
                      "rows": [f"float32[{d}]"] if d else [],
                      "kind": "in"})

    # op nodes
    node_map = {}
    for ni, n in enumerate(g.node):
        if ni in folded_consts:
            continue
        rows = []
        for inp in n.input:
            if inp in init_map:
                nm = inp.split("/")[-1].split(".")[-1] or "init"
                rows.append(f"{nm}: {val_for(inp)}")
            elif inp in const_val and const_node.get(inp, -1) in folded_consts:
                nm = inp.split("/")[-1].split(".")[-1] or "const"
                rows.append(f"{nm}: {const_val[inp]}")
        for a in n.attribute:
            rows.append(attr_str(a))
        if len(rows) > 4:
            rows = rows[:3] + [f"(+{len(rows) - 3} attrs)"]
        node_map[ni] = len(nodes)
        nodes.append({"label": n.op_type, "header": hdr_color(n.op_type),
                      "rows": rows, "kind": "op"})
        for o in n.output:
            producer[o] = ni

    # edges (input -> consumer via producers / graph inputs)
    for ni, n in enumerate(g.node):
        if ni not in node_map:
            continue
        me = node_map[ni]
        for inp in n.input:
            if not inp:
                continue
            if inp in producer and producer[inp] in node_map:
                edges.add((node_map[producer[inp]], me))
            elif inp in in_idx:
                edges.add((in_idx[inp], me))

    # graph outputs
    for vo in g.output:
        d = dims_str(vo)
        idx = len(nodes)
        nodes.append({"label": vo.name.split("/")[-1] or vo.name,
                      "header": "output",
                      "rows": [f"float32[{d}]"] if d else [],
                      "kind": "out"})
        if vo.name in producer and producer[vo.name] in node_map:
            edges.add((node_map[producer[vo.name]], idx))
    return nodes, sorted(edges)


# ---- layout -------------------------------------------------------------------
def layout(nodes, edges):
    """Vertical layered layout. Returns pos{i:(x,y)}, widths, heights, W, H."""
    succ, pred = {}, {}
    for a, b in edges:
        succ.setdefault(a, []).append(b)
        pred.setdefault(b, []).append(a)

    # Kahn topological order
    indeg = {i: len(pred.get(i, [])) for i in range(len(nodes))}
    queue = [i for i in range(len(nodes)) if indeg[i] == 0]
    topo = []
    while queue:
        u = queue.pop()
        topo.append(u)
        for v in succ.get(u, []):
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
    seen = set(topo)
    topo.extend(i for i in range(len(nodes)) if i not in seen)

    # longest-path layer assignment; outputs hang one below their producer
    layer = [0] * len(nodes)
    for u in topo:
        for v in succ.get(u, []):
            if layer[v] < layer[u] + 1:
                layer[v] = layer[u] + 1
    for i, n in enumerate(nodes):
        if n["kind"] == "out" and pred.get(i):
            layer[i] = max(layer[p] for p in pred[i]) + 1

    by_layer = {}
    for i, l in enumerate(layer):
        by_layer.setdefault(l, []).append(i)

    # measured widths; rows ellipsized so text always fits its box
    widths = []
    for n in nodes:
        w = measure(n["label"], "io" if n["kind"] != "op" else "hdr")
        fitted = []
        for r in n["rows"]:
            r = fit_row(r, MAX_NODE_W - NODE_W_PAD)
            fitted.append(r)
            w = max(w, measure(r, "attr"))
        n["rows"] = fitted
        widths.append(max(64, min(MAX_NODE_W, w + NODE_W_PAD)))
    heights = [HDR_H + 4 + ROW_H * len(n["rows"]) + 4 for n in nodes]

    # per-layer y accumulation (variable node heights)
    layer_y = {}
    y = 0.0
    for l in sorted(by_layer):
        layer_y[l] = y
        y += max(heights[i] for i in by_layer[l]) + LAYER_GAP_Y
    total_h = y - LAYER_GAP_Y

    # column ordering via barycenter sweeps (reduces edge crossings)
    col = {}
    for l, members in by_layer.items():
        for k, i in enumerate(sorted(members)):
            col[i] = k
    for _ in range(6):
        for l in sorted(by_layer):
            members = by_layer[l]
            members.sort(key=lambda i: (float(np.mean([col[p] for p in
                                         pred.get(i, [i])])), i))
            for k, i in enumerate(members):
                col[i] = k
        for l in sorted(by_layer, reverse=True):
            members = by_layer[l]
            members.sort(key=lambda i: (float(np.mean([col[s] for s in
                                         succ.get(i, [i])])), i))
            for k, i in enumerate(members):
                col[i] = k

    # columns -> x with per-node widths; center each row
    pos, row_w = {}, {}
    for l, members in by_layer.items():
        x = 0.0
        for i in sorted(members, key=lambda i: col[i]):
            pos[i] = (x, layer_y[l])
            x += widths[i] + NODE_GAP_X
        row_w[l] = x - NODE_GAP_X
    max_row_w = max(row_w.values()) if row_w else 0.0
    for l, members in by_layer.items():
        off = (max_row_w - row_w[l]) / 2.0
        for i in members:
            pos[i] = (pos[i][0] + off, pos[i][1])

    return pos, widths, heights, max_row_w, total_h


# ---- render -------------------------------------------------------------------
def render(model, title, out_png, out_svg=None):
    nodes, edges = build(model)
    pos, widths, heights, total_w, total_h = layout(nodes, edges)

    fig_w = max(4.0, (total_w + 80) / 100.0)
    fig_h = min(max(3.0, (total_h + 90) / 100.0), 200.0)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])  # full-bleed: 1 data unit == 1 px @ dpi 100
    ax.set_xlim(-40, total_w + 40)
    ax.set_ylim(total_h + 60, -44)   # inverted: layer 0 on top
    ax.axis("off")

    ax.text(0, -34, title, fontsize=12, color="#111", family=FONT,
            fontweight="bold")
    ax.text(0, -18, f"{len(nodes)} nodes · {len(edges)} edges",
            fontsize=9, color="#555", family=FONT)

    # edges: bottom-center of producer -> top-center of consumer (soft curve)
    for a, b in edges:
        xa = pos[a][0] + widths[a] / 2.0
        ya = pos[a][1] + heights[a]
        xb = pos[b][0] + widths[b] / 2.0
        yb = pos[b][1]
        mid = (ya + yb) / 2.0
        path = Path([(xa, ya), (xa, mid), (xb, mid), (xb, yb)],
                    [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
        ax.add_patch(PathPatch(path, fill=False, edgecolor=EDGE_COLOR,
                               lw=0.9, zorder=1))

    for i, n in enumerate(nodes):
        x, y = pos[i]
        w, h = widths[i], heights[i]
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0,rounding_size=5",
                     linewidth=0.8, edgecolor=BORDER,
                     facecolor="white", zorder=2))
        if n["kind"] in ("in", "out"):
            hh = min(HDR_H, h)
            ax.add_patch(FancyBboxPatch((x, y), w, hh,
                         boxstyle="round,pad=0,rounding_size=5",
                         linewidth=0, edgecolor="none",
                         facecolor=IO_FILL, zorder=3))
            ax.add_patch(Rectangle((x, y + hh - 5), w, 5, facecolor=IO_FILL,
                         edgecolor="none", zorder=3))
            ax.text(x + 6, y + hh / 2, n["label"], fontsize=8.5,
                    color="#333", family=FONT, va="center", zorder=5)
        else:
            ax.add_patch(FancyBboxPatch((x, y), w, HDR_H,
                         boxstyle="round,pad=0,rounding_size=5",
                         linewidth=0, edgecolor="none",
                         facecolor=n["header"], zorder=3))
            ax.add_patch(Rectangle((x, y + HDR_H - 5), w, 5,
                         facecolor=n["header"], edgecolor="none", zorder=3))
            ax.text(x + 6, y + HDR_H / 2, n["label"], fontsize=10,
                    color="white", family=FONT, va="center", zorder=5,
                    fontweight="bold")
        yy = y + (HDR_H if n["kind"] == "op" else min(HDR_H, h))
        for r in n["rows"]:
            ax.plot([x, x + w], [yy, yy], color="#d8d8d8", lw=0.6, zorder=4)
            ax.text(x + 6, yy + ROW_H / 2, r, fontsize=7.5,
                    color=ATTR_TEXT, family="Consolas", va="center", zorder=5)
            yy += ROW_H

    fig.savefig(out_png, bbox_inches="tight", facecolor="white", dpi=150)
    if out_svg:
        fig.savefig(out_svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return len(nodes), len(edges)


# ---- cli ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", help="path to .onnx")
    ap.add_argument("-o", "--out",
                    help="output image path (default docs/images/modelgraph_<name>.png)")
    ap.add_argument("--all", nargs=2, metavar=("DIR", "OUTDIR"),
                    help="render every .onnx in DIR into OUTDIR")
    args = ap.parse_args()

    jobs = []
    if args.all:
        indir, outdir = args.all
        os.makedirs(outdir, exist_ok=True)
        for f in sorted(os.listdir(indir)):
            if f.lower().endswith(".onnx"):
                base = os.path.splitext(f)[0]
                jobs.append((os.path.join(indir, f),
                             os.path.join(outdir, f"modelgraph_{base}.png")))
    elif args.model:
        base = os.path.splitext(os.path.basename(args.model))[0]
        out = args.out or os.path.join("docs", "images", f"modelgraph_{base}.png")
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        jobs.append((args.model, out))
    else:
        ap.error("give a model path or --all DIR OUTDIR")

    for src, dst in jobs:
        try:
            model = onnx.load(src)
            svg = os.path.splitext(dst)[0] + ".svg"
            n, e = render(model, os.path.basename(src), dst, svg)
            print(f"{os.path.basename(src):28s} -> {dst} + .svg  "
                  f"({n} nodes, {e} edges)")
        except Exception as ex:
            print(f"{src}: FAILED — {ex}", file=sys.stderr)


if __name__ == "__main__":
    main()
