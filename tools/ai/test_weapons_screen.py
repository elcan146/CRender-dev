"""test_weapons_screen.py — RAW-ish weapons_m.onnx on the live screen.

No pipeline config, no tracker state beyond a light persist check, no person
session. Per frame: grab -> letterbox 640 -> infer -> class-AGNOSTIC NMS (an
overlapping GUN+RIFLE pair on one object collapses to the best box) -> size
gates (icons < 0.15% of frame and >55% fill-hallucinations drop) -> draw only
dets seen in >=persist consecutive passes (flicker FPs die, real guns stay).

  python tools/ai/test_weapons_screen.py [conf] [iou] [persist] [model.onnx] [all] [knife] [wide]

  conf     raw score floor (default 0.40)
  iou      NMS threshold (default 0.45)
  persist  consecutive passes required to draw (default 2; 1 = pure raw)
  model    config/<name>.onnx to load (default weapons_m.onnx —
           try weapons_m_1280.onnx: 4x input pixels = surer small-gun dets)
  all      cover all monitors instead of primary only
  knife    include class 3 (off by default — most screen FPs)
  wide     aspect gate: w/h >= 1.3 — drops square icon/text FPs
           (may also drop guns pointed at the camera)
  agree    draw only [WB] dets — boxes BOTH weapons_m and best.onnx fire on

CASCADE: best.onnx scouts the full frame at a low floor; the exact
proposal box (no margin) is cropped and letterboxed to weapons_m, which
re-detects on the zoomed content. Only crops weapons_m confirms draw (green [WB]). Anything best.onnx
misses never reaches stage 2 — union mode removed for this test.

Esc quits. Corner chip prints ms/det.
"""
import ctypes
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

MODEL = Path(__file__).resolve().parents[2] / "config" / "weapons_m.onnx"
MODEL_B = MODEL.parent / "best.onnx"          # corroborator (1-class gun)
NAMES = ["GUN", "RIFLE", "HEAVY", "KNIFE"]
VOTE_CONF = 0.20                              # best.onnx floor — calibrated low
SZ = 640
MIN_AREA, MAX_AREA = 0.0015, 0.55   # full-frame gates (~74px / half-screen)
MIN_FILL = 0.10                     # crop gate: det must fill >=10% of crop


def _iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


def letterbox(img, sz=SZ):
    """Top-left letterbox into sz x sz; returns (padded, scale)."""
    h, w = img.shape[:2]
    sc = min(sz / w, sz / h)
    nw, nh = int(w * sc), int(h * sc)
    pad = np.full((sz, sz, 3), 114, np.uint8)
    pad[:nh, :nw] = cv2.resize(img, (nw, nh))
    return pad, sc


def ensemble(d_w, d_b):
    """Union of both models' dets, source-tagged. A weapons_m det a best.onnx
    det overlaps at IoU>0.3 is corroborated -> 'WB'. A best det overlapping
    nothing weapons_m saw -> appended 'B'. The rest stay 'W'."""
    out, used_b = [], set()
    for d in d_w:
        tag = "W"
        for j, b in enumerate(d_b):
            if _iou(d[:4], b[:4]) > 0.30:
                tag = "WB"
                used_b.add(j)
                break
        out.append((*d[:6], tag))
    for j, b in enumerate(d_b):
        if j not in used_b and not any(_iou(b[:4], w[:4]) > 0.45 for w in d_w):
            out.append((*b[:6], "B"))
    return out


def _model_nc(sess) -> int:
    """Class count from the model's 'names' metadata (0 = unknown)."""
    import ast
    try:
        raw = sess.get_modelmeta().custom_metadata_map.get("names", "")
        if raw.strip().startswith("{"):
            return len(ast.literal_eval(raw))
        return sum(1 for t in raw.split(",") if t.strip().startswith(tuple("0123456789")))
    except Exception:
        return 0


def detect(sess, frame, conf, iou_thr, classes, wide=False, crop=False):
    """Raw decode -> class-agnostic NMS -> size gates. One physical object
    can't be both GUN and RIFLE: overlapping cross-class boxes collapse to
    the highest score. Returns [(x0,y0,x1,y1,cls,score)] normalized.

    crop=True (stage-2 proposal check): the full-frame area caps invert — a
    real det fills most of the crop, so max-area is off and MIN_FILL demands
    the det cover >=10% of the crop (a corner icon can't confirm a box)."""
    inp = sess.get_inputs()[0]
    sh = inp.shape
    if (len(sh) != 4 or sh[1] != 3 or not isinstance(sh[2], int)
            or sh[2] != sh[3]):
        raise ValueError(f"not a detector input {sh} — "
                         "pick a YOLO detect export, not a classifier")
    sz = int(sh[2])                                # 640 or 1280 export
    lb, sc = letterbox(frame, sz)
    x = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None]
    o = sess.run(None, {inp.name: x})[0][0]          # 'images' or 'frame' etc
    if o.ndim != 2 or min(o.shape) < 5 or max(o.shape) < 16:
        raise ValueError(f"not a detector output {o.shape} — "
                         "pick a YOLO detect export, not a classifier")
    if o.shape[0] > o.shape[1]:                      # anchors-first (yolov5 export)
        o = o.T                                      # -> (rows, anchors)
    nc = _model_nc(sess)
    if nc > 0 and o.shape[0] == nc + 5:              # v5: box4 + objectness + nc
        cls_sc = o[5:] * o[4:5]                      # score = obj * cls
    else:                                            # v8+: box4 + nc
        cls_sc = o[4:]                               # (nCls, N) post-sigmoid
    best_cls = cls_sc.argmax(0)
    best_sc = cls_sc.max(0)
    keep = [k for k in np.where(best_sc >= conf)[0] if best_cls[k] in classes]
    if not keep:
        return []
    keep = np.asarray(keep)[np.argsort(-best_sc[keep])][:300]
    taken = []                                        # [(box, cls, score)]
    for k in keep:
        cx, cy, w, h = o[0, k], o[1, k], o[2, k], o[3, k]
        b = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        ok = True
        for t, _tc, _ts in taken:
            ix0, iy0 = max(b[0], t[0]), max(b[1], t[1])
            ix1, iy1 = min(b[2], t[2]), min(b[3], t[3])
            if ix1 > ix0 and iy1 > iy0:
                inter = (ix1 - ix0) * (iy1 - iy0)
                union = ((b[2]-b[0]) * (b[3]-b[1])
                         + (t[2]-t[0]) * (t[3]-t[1]) - inter)
                if union > 0 and inter / union > iou_thr:
                    ok = False
                    break
        if ok:
            taken.append((b, int(best_cls[k]), float(best_sc[k])))
    H, W = frame.shape[:2]
    dets = []
    for b, c, s_ in taken:
        x0, y0 = float(np.clip(b[0] / sc / W, 0, 1)), float(np.clip(b[1] / sc / H, 0, 1))
        x1, y1 = float(np.clip(b[2] / sc / W, 0, 1)), float(np.clip(b[3] / sc / H, 0, 1))
        a = (x1 - x0) * (y1 - y0)
        if crop:
            if a < MIN_FILL:
                continue
        elif not (MIN_AREA <= a <= MAX_AREA):
            continue
        if wide and (x1 - x0) / max(y1 - y0, 1e-6) < 1.3:
            continue
        dets.append((x0, y0, x1, y1, c, s_))
    return dets


def main() -> None:
    conf, iou, persist = 0.40, 0.45, 2
    nums = [a for a in sys.argv[1:] if a.replace(".", "", 1).isdigit()]
    if len(nums) > 0:
        conf = float(nums[0])
    if len(nums) > 1:
        iou = float(nums[1])
    if len(nums) > 2:
        persist = int(float(nums[2]))
    flags = [a.lower() for a in sys.argv[1:]]
    all_screens = "all" in flags
    classes = [0, 1, 2] + ([3] if "knife" in flags else [])
    wide = "wide" in flags
    mdl = next((a for a in sys.argv[1:] if a.lower().endswith(".onnx")), None)
    model = (MODEL.parent / mdl) if mdl else MODEL
    assert model.exists(), f"missing {model}"
    sess = ort.InferenceSession(str(model),
                                providers=["DmlExecutionProvider",
                                           "CPUExecutionProvider"])
    sess_b = (ort.InferenceSession(str(MODEL_B),
                                   providers=["DmlExecutionProvider",
                                              "CPUExecutionProvider"])
              if MODEL_B.exists() else None)
    agree = "agree" in flags
    print(f"{model.name} conf={conf} iou={iou} persist={persist} "
          f"classes={classes} wide={wide} "
          f"scout={'best.onnx@%.2f' % VOTE_CONF if sess_b else 'none'} "
          f"[{sess.get_providers()[0]}]")

    from PIL import ImageGrab
    from PyQt6.QtCore import Qt, QRectF, QTimer
    from PyQt6.QtGui import QColor, QFont, QPainter, QPen
    from PyQt6.QtWidgets import QApplication, QWidget

    lock = threading.Lock()
    stop = threading.Event()
    shared: dict = {"dets": [], "frames": 0, "ms": 0.0}

    prev: list = []        # [{box,cls,n}] — last pass's dets

    def worker() -> None:
        nonlocal prev
        while not stop.is_set():
            shot = ImageGrab.grab(all_screens=all_screens)
            frame = cv2.cvtColor(np.asarray(shot), cv2.COLOR_RGB2BGR)
            t0 = time.perf_counter()
            props = (detect(sess_b, frame, VOTE_CONF, iou, [0], wide)
                     if sess_b else detect(sess, frame, conf, iou, classes, wide))
            Hf, Wf = frame.shape[:2]
            dets = []
            for d in props[:4]:                      # cap crops/pass
                px0, py0 = d[0] * Wf, d[1] * Hf
                px1, py1 = d[2] * Wf, d[3] * Hf
                cx, cy = (px0 + px1) / 2, (py0 + py1) / 2
                cw, ch = px1 - px0, py1 - py0   # exact box, no margin
                X0 = max(0, int(cx - cw / 2)); Y0 = max(0, int(cy - ch / 2))
                X1 = min(Wf, int(cx + cw / 2)); Y1 = min(Hf, int(cy + ch / 2))
                if X1 - X0 < 24 or Y1 - Y0 < 24:
                    continue
                d2 = detect(sess, frame[Y0:Y1, X0:X1], conf, iou,
                            classes, wide, crop=True)
                if not d2:
                    continue
                top = max(d2, key=lambda z: z[5])    # crop-space best det
                bx0 = (X0 + top[0] * (X1 - X0)) / Wf
                by0 = (Y0 + top[1] * (Y1 - Y0)) / Hf
                bx1 = (X0 + top[2] * (X1 - X0)) / Wf
                by1 = (Y0 + top[3] * (Y1 - Y0)) / Hf
                dets.append((bx0, by0, bx1, by1, top[4], top[5], "WB"))
            ms = (time.perf_counter() - t0) * 1000.0
            cur = []
            for d in dets:
                n, box, sc_ = 1, d[:4], d[5]
                for p in prev:
                    if p["cls"] == d[4] and _iou(d[:4], p["box"]) > 0.30:
                        n = p["n"] + 1
                        box = tuple(p["box"][i] + (d[i] - p["box"][i]) * 0.4
                                    for i in range(4))   # EMA, like the HUD
                        sc_ = p["score"] + (d[5] - p["score"]) * 0.4
                        break
                cur.append({"box": box, "cls": d[4], "score": sc_,
                            "n": n, "src": d[6]})
            prev = cur
            view = [(*r["box"], r["cls"], r["score"], r["src"]) for r in cur
                    if r["n"] >= persist and (not agree or r["src"] == "WB")]
            with lock:
                shared["dets"] = view
                shared["frames"] += 1
                shared["ms"] = ms

    threading.Thread(target=worker, daemon=True).start()

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
            qt.start(33)

        def _tick(self) -> None:
            if ctypes.windll.user32.GetAsyncKeyState(0x1B) & 0x8000:
                app.quit()
            self.update()

        def paintEvent(self, _e) -> None:
            with lock:
                dets, ms = list(shared["dets"]), shared["ms"]
            W, H = self.width(), self.height()
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
            p.setPen(QPen(QColor(255, 40, 40, 90)))
            p.drawText(8, 20, f"WEAPONS-RAW {ms:.0f}ms/det")
            if not dets:
                p.end()
                return
            p.setFont(QFont("Consolas", 11, QFont.Weight.Bold))
            colors = {"WB": QColor(80, 255, 80), "W": QColor(255, 40, 40),
                      "B": QColor(255, 200, 40)}
            for x0, y0, x1, y1, c, sc_, src in dets:
                p.setPen(QPen(colors.get(src, QColor(255, 40, 40)), 3))
                r = QRectF(x0 * W, y0 * H, (x1 - x0) * W, (y1 - y0) * H)
                p.drawRect(r)
                p.drawText(int(r.left()), max(int(r.top()) - 6, 14),
                           f"{NAMES[c]} {sc_ * 100:.0f}% [{src}]")
            p.end()

    ov = Overlay()
    ov.show()
    print("raw weapons overlay — Esc to quit")
    t0 = time.perf_counter()
    try:
        app.exec()
    finally:
        stop.set()
    dt = time.perf_counter() - t0
    print(f"{shared['frames']} det passes in {dt:.0f}s "
          f"({shared['frames'] / max(dt, 0.01):.1f}/s)")


if __name__ == "__main__":
    main()
