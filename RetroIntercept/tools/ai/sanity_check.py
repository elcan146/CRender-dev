"""Run the detector models on a real image — no game required.

  python tools\\ai\\sanity_check.py photo.jpg [photo2.jpg ...]

Uses ultralytics' ONNX backend (pip install ultralytics onnx onnxruntime).
Writes annotated copies next to each input as <name>_<model>.jpg.
"""
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
MODELS = [
    ("coco",    ROOT / "config" / "yolov8n.onnx",    {0: "person"}),
    ("weapons", ROOT / "config" / "det_weapons.onnx", {0: "guns", 1: "knife"}),
]


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: sanity_check.py <image> [more images]")
    models = [(tag, YOLO(str(mp), task="detect"), names)
              for tag, mp, names in MODELS if mp.exists()]
    if not models:
        sys.exit("no models found in config/ (yolov8n.onnx, det_weapons.onnx)")
    for img_arg in sys.argv[1:]:
        img = Path(img_arg)
        print(f"\n== {img.name} ==")
        for tag, m, names in models:
            r = m.predict(str(img), conf=0.25, verbose=False)[0]
            for b in r.boxes:
                cls = int(b.cls[0])
                xy = [round(v) for v in b.xyxy[0].tolist()]
                print(f"  [{tag}] {names.get(cls, r.names.get(cls, cls)):<8} "
                      f"{float(b.conf[0]):.2f} box={xy}")
            out = img.with_name(f"{img.stem}_{tag}{img.suffix}")
            r.save(filename=str(out))
            print(f"  [{tag}] -> {out.name}")


if __name__ == "__main__":
    main()
