"""Test f:\\dovuland\\best.onnx (BecayeSoft Gun-Detection YOLOv8, class 'gun')
on the local test images, or on a video file. Writes annotated copies as
<name>_bestgun.<ext>.

  python tools\\ai\\test_bestgun.py [conf]
  python tools\\ai\\test_bestgun.py video.mp4 [conf]
  python tools\\ai\\test_bestgun.py screen[:all] [conf]   # transparent overlay
"""
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # hide GPU — CPU provider, no CUDA spam

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import onnxruntime as ort
    ort.set_default_logger_severity(3)   # hide remaining provider noise
except Exception:
    pass

MODEL = Path(r"f:\dovuland\best.onnx")
TESTIMG = Path(__file__).resolve().parent / "testimg"
IMAGES = ["teste.jpg", "guns_batch.jpg", "bus.jpg",
          "repo_test-image.jpg", "repo_test_hd.jpg", "repo_Guns.jpg"]


def run_video(m: YOLO, video: Path, conf: float) -> None:
    """Frame-by-frame inference; writes <name>_bestgun.<ext> annotated."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"{video}: cannot open")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = video.with_name(f"{video.stem}_bestgun{video.suffix}")
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = hits = 0
    best = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        r = m.predict(frame, conf=conf, verbose=False)[0]
        vw.write(r.plot())
        for b in r.boxes:
            hits += 1
            c = float(b.conf[0])
            best = max(best, c)
            xy = [round(v) for v in b.xyxy[0].tolist()]
            print(f"   f{n:05d} gun conf={c:.3f} box={xy}")
    cap.release()
    vw.release()
    print(f"{video.name}: {n} frames, {hits} gun dets, best {best:.2f} -> {out.name}")


# --- mini tracker: same semantics as the in-game term::Tracker ----------
_IOU_MATCH, _DIST_MATCH, _EMA, _COAST = 0.25, 0.08, 0.35, 0.35
_MIN_HITS = 4   # flicker gate — same as in-game weapon tracks


def _iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


class _Tracker:
    """IoU/centroid match + EMA smoothing + coast — normalized 0..1 coords."""

    def __init__(self) -> None:
        self.tracks: list[dict] = []
        self._last = 0.0
        self._next_id = 1

    def update(self, dets, now: float) -> None:
        dt = min(max(now - self._last, 0.0), 1.0) if self._last else 0.0
        self._last = now
        used = set()
        for box, score in sorted(dets, key=lambda d: -d[1]):
            best, best_iou = -1, _IOU_MATCH
            for i, tr in enumerate(self.tracks):
                if i in used:
                    continue
                v = _iou(box, tr["box"])
                if v > best_iou:
                    best_iou, best = v, i
            if best < 0:                      # centroid-distance fallback
                cx = (box[0]+box[2])*0.5
                cy = (box[1]+box[3])*0.5
                best_d = _DIST_MATCH * _DIST_MATCH
                for i, tr in enumerate(self.tracks):
                    if i in used:
                        continue
                    dx = cx - (tr["box"][0]+tr["box"][2])*0.5
                    dy = cy - (tr["box"][1]+tr["box"][3])*0.5
                    if dx*dx + dy*dy < best_d:
                        best_d, best = dx*dx + dy*dy, i
            if best >= 0:
                tr = self.tracks[best]
                tr["box"] = tuple(tr["box"][k] + (box[k]-tr["box"][k])*_EMA
                                  for k in range(4))
                tr["unseen"] = 0.0
                tr["score"] = score
                tr["hits"] += 1
                used.add(best)
            else:
                self.tracks.append({"box": box, "score": score, "unseen": 0.0,
                                    "hits": 1, "id": self._next_id})
                self._next_id = self._next_id % 99 + 1
                used.add(len(self.tracks) - 1)
        kept = []
        for i, tr in enumerate(self.tracks):
            if i not in used:
                tr["unseen"] += dt
            if tr["unseen"] <= _COAST:
                kept.append(tr)
        self.tracks = kept

    def view(self):
        return [t for t in self.tracks if t["hits"] >= _MIN_HITS]


def run_screen(m: YOLO, conf: float, all_screens: bool) -> None:
    """Transparent click-through overlay over the desktop — draws tracked
    gun boxes like the in-game HUD (EMA + coast + hits>=4 + sqrt conf).
    Inference on a worker thread. Esc quits."""
    import ctypes
    import threading
    from PIL import ImageGrab
    from PyQt6.QtCore import Qt, QRectF, QTimer
    from PyQt6.QtGui import QColor, QFont, QPainter, QPen
    from PyQt6.QtWidgets import QApplication, QWidget

    lock = threading.Lock()
    stop = threading.Event()
    shared: dict = {"tracks": [], "frames": 0}
    trk = _Tracker()

    def worker() -> None:
        while not stop.is_set():
            shot = ImageGrab.grab(all_screens=all_screens)
            frame = cv2.cvtColor(np.asarray(shot), cv2.COLOR_RGB2BGR)
            r = m.predict(frame, conf=conf, verbose=False)[0]
            h, w = frame.shape[:2]
            dets = []
            for b in r.boxes:
                x0, y0, x1, y1 = (v / s for v, s in zip(b.xyxy[0].tolist(), (w, h, w, h)))
                dets.append(((x0, y0, x1, y1), float(b.conf[0]) ** 0.5))  # sqrt, like in-game
            trk.update(dets, time.perf_counter())
            with lock:
                shared["tracks"] = trk.view()
                shared["frames"] += 1

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    app = QApplication.instance() or QApplication(sys.argv)

    class Overlay(QWidget):
        def __init__(self) -> None:
            super().__init__(flags=(Qt.WindowType.FramelessWindowHint
                                    | Qt.WindowType.WindowStaysOnTopHint
                                    | Qt.WindowType.Tool))
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
            geo = app.primaryScreen().geometry()
            if all_screens:
                for s in app.screens():
                    geo = geo.united(s.geometry())
            self.setGeometry(geo)
            qt = QTimer(self)
            qt.timeout.connect(self._tick)
            qt.start(33)  # ~30fps repaint

        def _tick(self) -> None:
            if ctypes.windll.user32.GetAsyncKeyState(0x1B) & 0x8000:  # VK_ESCAPE
                app.quit()
            self.update()

        def paintEvent(self, _e) -> None:
            with lock:
                tracks = list(shared["tracks"])
            W, H = self.width(), self.height()
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            # Corner chip so the transparent overlay is visibly alive at 0 dets.
            p.setPen(QPen(QColor(255, 40, 40, 90)))
            p.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
            p.drawText(8, 20, "GUN-SCAN")
            if not tracks:
                p.end()
                return
            p.setPen(QPen(QColor(255, 40, 40), 3))
            p.setFont(QFont("Consolas", 11, QFont.Weight.Bold))
            for tr in tracks:
                x0, y0, x1, y1 = tr["box"]
                r = QRectF(x0 * W, y0 * H, (x1 - x0) * W, (y1 - y0) * H)
                p.drawRect(r)
                p.drawText(int(r.left()), max(int(r.top()) - 6, 14),
                           f"GUN {tr['score'] * 100:.0f}%")
            p.end()

    ov = Overlay()
    ov.show()
    print("overlay running — Esc to quit (look for the GUN-SCAN chip, top-left)")
    t0 = time.perf_counter()
    try:
        app.exec()
    finally:
        stop.set()
        t.join(timeout=1.0)
    dt = time.perf_counter() - t0
    n = shared["frames"]
    print(f"screen: {n} infer passes in {dt:.0f}s ({n / dt:.1f} infer fps)")


def main() -> None:
    conf = 0.25
    video = None
    screen_all = None   # None = off; bool = all-monitors flag
    for a in sys.argv[1:]:
        low = a.lower()
        if low.startswith("screen"):
            screen_all = low == "screen:all"
            continue
        try:
            conf = float(a)
        except ValueError:
            video = Path(a)
    print(f"model: {MODEL} (exists={MODEL.exists()}, {MODEL.stat().st_size//1024} KB)")
    m = YOLO(str(MODEL), task="detect")
    print(f"names: {m.names}")
    if screen_all is not None:
        run_screen(m, conf, screen_all)
        return
    if video is not None:
        run_video(m, video, conf)
        return
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
