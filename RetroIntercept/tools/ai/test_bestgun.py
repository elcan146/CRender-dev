"""Test f:\\dovuland\\best.onnx (BecayeSoft Gun-Detection YOLOv8, class 'gun')
on the local test images. Writes annotated copies as <name>_bestgun.<ext>.

  python tools\\ai\\test_bestgun.py [conf]
"""
import sys
from pathlib import Path

from ultralytics import YOLO

MODEL = Path(r"f:\dovuland\best.onnx")
TESTIMG = Path(__file__).resolve().parent / "testimg"
IMAGES = ["teste.jpg", "guns_batch.jpg", "bus.jpg",
          "repo_test-image.jpg", "repo_test_hd.jpg", "repo_Guns.jpg"]


def main() -> None:
    conf = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
    print(f"model: {MODEL} (exists={MODEL.exists()}, {MODEL.stat().st_size//1024} KB)")
    m = YOLO(str(MODEL), task="detect")
    print(f"names: {m.names}")
    for name in IMAGES:
        img = TESTIMG / name
        if not img.exists():
            print(f"== {name}: MISSING ==")
            continue
        r = m.predict(str(img), conf=conf, verbose=False)[0]
        print(f"== {name}: {len(r.boxes)} det @conf{conf} ==")
        for b in r.boxes:
            xy = [round(v) for v in b.xyxy[0].tolist()]
            cls = int(b.cls[0])
            print(f"   {r.names[cls]} conf={float(b.conf[0]):.3f} box={xy}")
        out = img.with_name(f"{img.stem}_bestgun{img.suffix}")
        r.save(filename=str(out))
        print(f"   -> {out.name}")


if __name__ == "__main__":
    main()
