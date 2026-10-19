"""fetch_detector.py — produce config/yolov8n.onnx for the Terminator detector.

Exports Ultralytics YOLOv8n (COCO-80) to a fixed-shape, NMS-free ONNX graph:
  input  [1,3,640,640] fp32 0..1
  output [1,84,8400]   fp32  cx,cy,w,h (normalized) + 80 class scores

Requires: pip install ultralytics onnx
Usage:    python tools/ai/fetch_detector.py [--model yolov8n.pt] [--size 640]
              [-o config/yolov8n.onnx]

Any COCO-80 YOLOv8/v11 checkpoint works (.pt); for a weapon-trained model pass its
.pt path (class names must still resolve through the detector's table — use numeric
ids in DetClasses/DetWeapon when the model isn't COCO-80).
"""
import argparse
import os
import shutil
import tempfile


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pt", help="ultralytics checkpoint (.pt)")
    ap.add_argument("--size", type=int, default=640, help="square input size (multiple of 64)")
    ap.add_argument("-o", "--output", default=None,
                    help="output .onnx path (default: <repo>/config/<model>.onnx)")
    args = ap.parse_args()

    from ultralytics import YOLO  # pip install ultralytics
    import onnx

    out = args.output
    if out is None:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        stem = os.path.splitext(os.path.basename(args.model))[0]
        out = os.path.join(root, "config", f"{stem}.onnx")

    print(f"exporting {args.model} @ {args.size}px ...")
    m = YOLO(args.model)
    exported = m.export(format="onnx", imgsz=args.size, opset=17, simplify=True,
                        dynamic=False, nms=False)  # raw [1,84,N] — decoded in C++
    if not os.path.isfile(exported):
        raise SystemExit(f"export failed: {exported}")

    # Verify the contract Detect.cpp decodes.
    mod = onnx.load(exported)
    sig = {o.name: [d.dim_value or -1 for d in o.type.tensor_type.shape.dim]
           for o in mod.graph.output}
    print(f"outputs: {sig}")
    dims = next(iter(sig.values()))
    if len(dims) != 3 or dims[0] != 1 or min(dims[1], dims[2]) != 4 + 80:
        raise SystemExit("unexpected output layout — expected [1,84,N] or [1,N,84]")

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    if os.path.abspath(exported) != os.path.abspath(out):
        shutil.copyfile(exported, out)
    print(f"wrote {out} ({os.path.getsize(out) // 1024} KiB)")
    print("set \"DetModel\" in retro.json to this filename and rebuild")


if __name__ == "__main__":
    main()
