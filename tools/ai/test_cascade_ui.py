"""test_cascade_ui.py — configurable cascade tester, PyQt6 panel + overlay.

Pick any config/*.onnx per stage. Stage 1 scouts the full frame; each stage-2
selection gets exact-box crops of stage-1 dets and must confirm them; stage 3
confirms stage-2 dets the same way. Stages left as (none) are skipped —
stage 1 alone = raw single-model view.

Per stage: model, conf floor, NMS IoU, class csv (blank = all).
Global: persist-N consecutive passes before a det draws (EMA-smoothed),
show-proposals toggle (dim yellow = stage-1 raw dets), Esc/Quit exits.

  python tools/ai/test_cascade_ui.py
"""
import ast
import ctypes
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np


def _ort_ok() -> bool:
    """Probe onnxruntime in a SUBPROCESS — importing it here would lock
    its .pyd files and pip couldn't repair them. Returns True if healthy."""
    import subprocess as _sp
    r = _sp.run([sys.executable, "-c",
                 "import onnxruntime as o; exit(0 if "
                 "hasattr(o, 'InferenceSession') else 1)"],
                capture_output=True)
    return r.returncode == 0


if not _ort_ok():   # ultralytics keeps shadowing ort with plain onnxruntime
    print("onnxruntime shadowed — repairing before import...")
    import subprocess as _sp
    _sp.run([sys.executable, "-m", "pip", "install", "-q",
             "--force-reinstall", "--no-deps", "onnxruntime-directml"],
            capture_output=True)

import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_weapons_screen import detect, letterbox, _iou   # noqa: E402

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
PROV = ["DmlExecutionProvider", "CPUExecutionProvider"]
SESS_CACHE: dict = {}


def get_sess(path: Path):
    s = SESS_CACHE.get(str(path))
    if s is None:
        s = ort.InferenceSession(str(path), providers=PROV)
        SESS_CACHE[str(path)] = s
    return s


_U32, _G32 = ctypes.windll.user32, ctypes.windll.gdi32
_VP, _I, _U = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32
_U32.GetDC.argtypes = (_VP,); _U32.GetDC.restype = _VP
_U32.ReleaseDC.argtypes = (_VP, _VP)
_G32.CreateCompatibleDC.argtypes = (_VP,); _G32.CreateCompatibleDC.restype = _VP
_G32.CreateCompatibleBitmap.argtypes = (_VP, _I, _I)
_G32.CreateCompatibleBitmap.restype = _VP
_G32.SelectObject.argtypes = (_VP, _VP); _G32.SelectObject.restype = _VP
_G32.BitBlt.argtypes = (_VP, _I, _I, _I, _I, _VP, _I, _I, _U)
_G32.GetDIBits.argtypes = (_VP, _VP, _U, _U, _VP, _VP, _U)
_G32.DeleteObject.argtypes = (_VP,)
_G32.DeleteDC.argtypes = (_VP,)


def grab_screen():
    """BGR screen frame via plain SRCCOPY BitBlt.
    BitBlt without CAPTUREBLT never includes layered windows, so our
    translucent overlay can never leak into detection frames — no
    display-affinity API needed. Returns uint8 (H, W, 3)."""
    u32, g32 = _U32, _G32
    x0 = u32.GetSystemMetrics(76)                   # SM_XVIRTUALSCREEN
    y0 = u32.GetSystemMetrics(77)                   # SM_YVIRTUALSCREEN
    w = u32.GetSystemMetrics(78)                    # SM_CXVIRTUALSCREEN
    h = u32.GetSystemMetrics(79)                    # SM_CYVIRTUALSCREEN
    sdc, mdc = u32.GetDC(0), g32.CreateCompatibleDC(0)
    bmp = g32.CreateCompatibleBitmap(sdc, w, h)
    g32.SelectObject(mdc, bmp)
    g32.BitBlt(mdc, 0, 0, w, h, sdc, x0, y0, 0x00CC0020)  # SRCCOPY only

    class _BMI(ctypes.Structure):                   # BITMAPINFOHEADER
        _fields_ = [("biSize", ctypes.c_uint32),
                    ("biWidth", ctypes.c_int32), ("biHeight", ctypes.c_int32),
                    ("biPlanes", ctypes.c_uint16), ("biBitCount", ctypes.c_uint16),
                    ("biCompression", ctypes.c_uint32), ("biSizeImage", ctypes.c_uint32),
                    ("biXPelsPerMeter", ctypes.c_int32), ("biYPelsPerMeter", ctypes.c_int32),
                    ("biClrUsed", ctypes.c_uint32), ("biClrImportant", ctypes.c_uint32)]

    bmi = _BMI()
    bmi.biSize, bmi.biWidth, bmi.biHeight = ctypes.sizeof(_BMI), w, -h
    bmi.biPlanes, bmi.biBitCount = 1, 32
    buf = (ctypes.c_ubyte * (w * h * 4))()
    g32.GetDIBits(mdc, bmp, 0, h, buf, ctypes.byref(bmi), 0)
    g32.DeleteObject(bmp)
    g32.DeleteDC(mdc)
    u32.ReleaseDC(0, sdc)
    return np.frombuffer(buf, np.uint8).reshape(h, w, 4)[:, :, :3].copy()


def class_names(sess) -> dict:
    """Embedded 'names' metadata -> {id: label}. Fallback CLS n."""
    try:
        raw = sess.get_modelmeta().custom_metadata_map.get("names", "")
        d = ast.literal_eval(raw) if raw.strip().startswith("{") else {}
        if not d:                                   # "0: 'gun', 1: ..." form
            for tok in raw.split(","):
                k, _, v = tok.partition(":")
                if k.strip().isdigit():
                    d[int(k.strip())] = v.strip().strip("'\"")
        return {int(k): str(v).upper() for k, v in d.items()}
    except Exception:
        return {}


def n_classes(sess) -> int:
    od = sess.get_outputs()[0].shape                 # [1,R,N] or [1,N,R]
    return min(od[1], od[2]) - 4 if len(od) == 3 else 0


def stage_dets(stage: dict, img, crop: bool = False) -> list:
    """Run every model selected for this stage independently on the same
    input; class ids resolved per-model -> (x0,y0,x1,y1,cls_name,score,mi).
    Union merge: cross-model overlaps collapse to the higher score.
    stage['tiles']>1 (full-frame only): NxN overlapping windows — a small
    weapon fills more of the 640 input, scoring higher at the same conf."""
    W0, H0 = img.shape[1], img.shape[0]
    views = [(0, 0, W0, H0, img)]
    n = int(stage.get("tiles", 1))
    if not crop and n > 1:
        views = []
        tw, th = W0 // n, H0 // n
        ovx, ovy = int(tw * 0.25), int(th * 0.25)    # 25% border overlap
        for gy in range(n):
            for gx in range(n):
                x0 = max(0, gx * tw - ovx)
                y0 = max(0, gy * th - ovy)
                x1 = min(W0, (gx + 1) * tw + ovx)
                y1 = min(H0, (gy + 1) * th + ovy)
                views.append((x0, y0, x1, y1, img[y0:y1, x0:x1]))
    dets = []
    for mi, mp in enumerate(stage["models"]):
        sess = get_sess(mp)
        names = class_names(sess)
        for X0, Y0, X1, Y1, view in views:
            for d in detect(sess, view, stage["conf"], stage["iou"],
                            stage["classes"], crop=crop):
                # tile-normalized -> frame-normalized coords
                bx0 = (X0 + d[0] * (X1 - X0)) / W0
                by0 = (Y0 + d[1] * (Y1 - Y0)) / H0
                bx1 = (X0 + d[2] * (X1 - X0)) / W0
                by1 = (Y0 + d[3] * (Y1 - Y0)) / H0
                dets.append((bx0, by0, bx1, by1,
                             names.get(d[4], f"CLS{d[4]}"), d[5], mi))
    dets.sort(key=lambda z: -z[5])
    keep = []
    for d in dets:
        if all(_iou(d[:4], k[:4]) <= 0.50 for k in keep):
            keep.append(d)
    return keep


def chain(cfg, frame):
    """Run the stage chain on a frame.
    Returns (final_dets, stage1_proposals) — each det (x0,y0,x1,y1,cls_name,
    score) in normalized frame coords."""
    st = cfg["stages"]
    props = stage_dets(st[0], frame)
    s1_props = [p[:4] for p in props]                # stage-1 raw boxes
    final = props
    Hf, Wf = frame.shape[:2]
    for si in (1, 2):
        if not st[si]["models"] or not props:
            break
        nxt = []
        for d in props[:4]:                          # cap 4 crops/pass
            X0, Y0 = max(0, int(d[0] * Wf)), max(0, int(d[1] * Hf))
            X1, Y1 = min(Wf, int(d[2] * Wf)), min(Hf, int(d[3] * Hf))
            if X1 - X0 < 24 or Y1 - Y0 < 24:
                continue
            for top in stage_dets(st[si], frame[Y0:Y1, X0:X1], crop=True):
                bx0 = (X0 + top[0] * (X1 - X0)) / Wf
                by0 = (Y0 + top[1] * (Y1 - Y0)) / Hf
                bx1 = (X0 + top[2] * (X1 - X0)) / Wf
                by1 = (Y0 + top[3] * (Y1 - Y0)) / Hf
                nxt.append((bx0, by0, bx1, by1, top[4], top[5], top[6]))
        props = nxt                                  # chain: next stage crops these
        final = nxt
    return final, s1_props


def main() -> None:
    from PyQt6.QtCore import Qt, QRectF, QTimer, pyqtSignal, QObject
    from PyQt6.QtGui import QColor, QFont, QPainter, QPen
    from PyQt6.QtWidgets import (QApplication, QWidget, QHBoxLayout,
                                 QVBoxLayout, QLabel, QComboBox,
                                 QDoubleSpinBox, QSpinBox, QLineEdit,
                                 QCheckBox, QPushButton)

    models = (sorted(p.name for p in CONFIG_DIR.glob("*.onnx"))
              + sorted(p.name for p in CONFIG_DIR.glob("*.pt")))
    if not models:
        raise SystemExit(f"no .onnx/.pt in {CONFIG_DIR}")

    class Bus(QObject):
        changed = pyqtSignal()
        status = pyqtSignal(str)

    bus = Bus()
    cfg_lock = threading.Lock()
    cfg: dict = {"stages": [{"models": [], "conf": 0.20, "iou": 0.45,
                             "tiles": 1,
                             "classes": list(range(256))} for _ in range(3)],
                 "persist": 2, "show_props": True}
    shared: dict = {"dets": [], "props": [], "ms": 0.0, "frames": 0}
    stop = threading.Event()

    app = QApplication.instance() or QApplication(sys.argv)

    # ---------- control panel ----------
    class Panel(QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("cascade test")
            self.rows = []
            lay = QVBoxLayout(self)
            # default preset — S1: test-2 PERSON+WEAPON scout, tiled (small
            # weapons stay large, people show as yellow proposals = visible
            # liveness); S2: weapons_m confirms at 0.50 — only weapon boxes
            # survive to final, so cross-family agreement kills most FPs.
            defaults = [
                ("test-2.onnx", "(none)", 0.25, 0.45, "0,1", 2, "640"),
                ("weapons_m.onnx", "(none)", 0.50, 0.45, "", 1, "640"),
                (None, "(none)", 0.40, 0.45, "", 1, "640"),
            ]
            for si in range(3):
                dm1, dm2, dcf, dio, dcl, dtl, dsz = defaults[si]
                row = QHBoxLayout()
                row.addWidget(QLabel(f"S{si + 1}"))
                cb = QComboBox()
                if si:
                    cb.addItem("(none)")
                cb.addItems(models)
                cb.setCurrentText(dm1 if dm1 in models
                                  else (models[0] if si == 0 else "(none)"))
                cb2 = QComboBox()                     # secondary model (optional)
                cb2.addItem("(none)")
                cb2.addItems(models)
                if dm2 in models:
                    cb2.setCurrentText(dm2)
                conf = QDoubleSpinBox(); conf.setRange(0.01, 1.0)
                conf.setSingleStep(0.05); conf.setValue(dcf)
                iou = QDoubleSpinBox(); iou.setRange(0.05, 0.95)
                iou.setSingleStep(0.05); iou.setValue(dio)
                cls = QLineEdit(); cls.setPlaceholderText("cls csv (all)")
                cls.setFixedWidth(70); cls.setText(dcl)
                tl = QSpinBox(); tl.setRange(1, 4); tl.setValue(dtl)
                tl.setToolTip("S1 tiles: NxN overlapping windows — small "
                              "objects stay large (4x slower per +1)")
                sz = QComboBox()                      # export/input size
                sz.addItems(["320", "480", "640", "960", "1280"])
                sz.setCurrentText(dsz)
                sz.setToolTip("export size for .pt picks (bigger = small "
                              "objects detect better, slower); ignored "
                              "for .onnx")
                for w, lbl in ((cb, None), (cb2, "+"), (conf, "conf"),
                               (iou, "iou"), (tl, "tl"), (sz, "sz")):
                    if lbl:
                        row.addWidget(QLabel(lbl))
                    row.addWidget(w)
                row.addWidget(cls)
                lay.addLayout(row)
                self.rows.append((cb, cb2, conf, iou, cls, tl, sz))
                for w in (cb, cb2, conf, iou, tl, sz):
                    sig = (w.currentTextChanged if isinstance(w, QComboBox)
                           else w.valueChanged)
                    sig.connect(bus.changed)
                cls.textChanged.connect(bus.changed)
            row = QHBoxLayout()
            row.addWidget(QLabel("persist"))
            ps = QSpinBox(); ps.setRange(1, 10); ps.setValue(2)
            ps.valueChanged.connect(bus.changed)
            row.addWidget(ps)
            pr = QCheckBox("show stage-1 proposals"); pr.setChecked(True)
            pr.toggled.connect(bus.changed)
            row.addWidget(pr)
            qb = QPushButton("Quit (Esc)")
            qb.clicked.connect(app.quit)
            row.addWidget(qb)
            lay.addLayout(row)
            self.persist = ps
            self.props = pr
            self._exporting: set = set()
            self.status = QLabel("ready")
            self.hb = QLabel("-")                   # worker heartbeat
            hb_t = QTimer(self)
            hb_t.timeout.connect(self._hb)
            hb_t.start(500)
            lay.addWidget(self.hb)
            lay.addWidget(self.status)
            bus.status.connect(self.status.setText)
            bus.changed.connect(self.apply)
            self.apply()

        def queue_export(self, pt_name: str, target: Path,
                         size: int = 640) -> None:
            """pt -> onnx at `size` via fetch_detector.py; then repair the
            onnxruntime/onnxruntime-directml shadow ultralytics keeps causing."""
            if str(target) in self._exporting:
                return
            self._exporting.add(str(target))
            bus.status.emit(f"exporting {pt_name} ...")
            import subprocess

            def run() -> None:
                tools_dir = CONFIG_DIR.parent / "tools" / "ai"
                r = subprocess.run(
                    [sys.executable, str(tools_dir / "fetch_detector.py"),
                     "--model", str(CONFIG_DIR / pt_name),
                     "--size", str(size), "-o", str(target)],
                    capture_output=True, text=True)
                ok = r.returncode == 0 and target.exists()
                if not ok:
                    # legacy checkpoint (yolov5/v7 pickle or old ultralytics
                    # namespace) — rescue-export it instead
                    bus.status.emit(f"{pt_name}: fetch_detector failed, "
                                    "trying legacy export ...")
                    r2 = subprocess.run(
                        [sys.executable, str(tools_dir / "export_legacy.py"),
                         "--model", str(CONFIG_DIR / pt_name),
                         "--size", str(size), "-o", str(target)],
                        capture_output=True, text=True)
                    ok = r2.returncode == 0 and target.exists()
                    if not ok:
                        print(r2.stdout[-1500:], r2.stderr[-1500:])
                # only repair when the export actually broke ort — an
                # unconditional reinstall can't swap .pyds our own
                # process already has loaded anyway.
                if not _ort_ok():
                    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                                    "--force-reinstall", "--no-deps",
                                    "onnxruntime-directml"],
                                   capture_output=True)
                self._exporting.discard(str(target))
                bus.status.emit(("ready: " + target.name) if ok else
                                f"export FAILED {pt_name} (see console)")
                if not ok:
                    print(r.stdout[-2000:], r.stderr[-2000:])
                bus.changed.emit()          # re-apply -> resolve the .onnx
            threading.Thread(target=run, daemon=True).start()

        def _hb(self) -> None:
            """Heartbeat: proves the grab→chain loop is alive."""
            with cfg_lock:
                f, np_, nd, ms = (shared["frames"], len(shared["props"]),
                                  len(shared["dets"]), shared["ms"])
            self.hb.setText(f"frames {f}  props {np_}  dets {nd}  "
                            f"{ms:.0f}ms")

        def _resolve(self, mdl: str, size: int):
            """Combo text -> usable model path (.pt auto-exports to .onnx)."""
            if mdl not in models or mdl == "(none)":
                return None
            if mdl.endswith(".pt"):
                stem = Path(mdl).stem
                target = CONFIG_DIR / (stem + ".onnx" if size == 640 else
                                       f"{stem}_{size}.onnx")
                if target.exists():
                    return target
                self.queue_export(mdl, target, size)
                return None
            return CONFIG_DIR / mdl

        def apply(self) -> None:
            st = []
            for cb, cb2, conf, iou, cls, tl, sz in self.rows:
                size = int(sz.currentText())
                ms = [p for p in (self._resolve(cb.currentText(), size),
                                  self._resolve(cb2.currentText(), size))
                      if p]
                csv = [int(x) for x in cls.text().split(",") if x.strip().isdigit()]
                st.append({"models": ms, "conf": conf.value(),
                           "iou": iou.value(), "tiles": tl.value(),
                           "classes": csv or list(range(256))})
            with cfg_lock:
                cfg["stages"] = st
                cfg["persist"] = self.persist.value()
                cfg["show_props"] = self.props.isChecked()

    # ---------- worker ----------
    prev: list = []
    last_err: list = [""]

    def worker() -> None:
        nonlocal prev
        while not stop.is_set():
            with cfg_lock:
                c = {"stages": [dict(s) for s in cfg["stages"]],
                     "persist": cfg["persist"],
                     "show_props": cfg["show_props"]}
            if not c["stages"][0]["models"]:
                time.sleep(0.05)
                continue
            t0 = time.perf_counter()
            try:
                frame = grab_screen()
                final, props = chain(c, frame)
                last_err[0] = ""
            except Exception as e:
                msg = str(e)
                if msg != last_err[0]:
                    last_err[0] = msg
                    print("chain err:", e)
                final, props = [], []
            ms = (time.perf_counter() - t0) * 1000.0
            cur = []
            for d in final:
                n, box, sc_ = 1, d[:4], d[5]
                for p in prev:
                    if p["cls"] == d[4] and _iou(d[:4], p["box"]) > 0.30:
                        n = p["n"] + 1
                        box = tuple(p["box"][i] + (d[i] - p["box"][i]) * 0.4
                                    for i in range(4))
                        sc_ = p["score"] + (d[5] - p["score"]) * 0.4
                        break
                cur.append({"box": box, "cls": d[4], "score": sc_, "n": n,
                            "mi": d[6]})
            prev = cur
            view = [(*r["box"], r["cls"], r["score"], r["mi"]) for r in cur
                    if r["n"] >= c["persist"]]
            with cfg_lock:
                shared["dets"] = view
                shared["props"] = props if c["show_props"] else []
                shared["ms"] = ms
                shared["frames"] += 1

    threading.Thread(target=worker, daemon=True).start()

    # ---------- overlay ----------
    class Overlay(QWidget):
        def __init__(self) -> None:
            super().__init__(flags=(Qt.WindowType.FramelessWindowHint
                                    | Qt.WindowType.WindowStaysOnTopHint
                                    | Qt.WindowType.Tool))
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
            # span the whole virtual screen (same coords grab_screen uses,
            # so normalized det coords map 1:1 onto this widget)
            self.setGeometry(_U32.GetSystemMetrics(76),
                             _U32.GetSystemMetrics(77),
                             _U32.GetSystemMetrics(78),
                             _U32.GetSystemMetrics(79))
            qt = QTimer(self)
            qt.timeout.connect(self._tick)
            qt.start(33)

        def _tick(self) -> None:
            if ctypes.windll.user32.GetAsyncKeyState(0x1B) & 0x8000:
                app.quit()
            self.update()

        def paintEvent(self, _e) -> None:
            with cfg_lock:
                dets = list(shared["dets"])
                props = list(shared["props"])
                ms = shared["ms"]
            W, H = self.width(), self.height()
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
            p.setPen(QPen(QColor(255, 40, 40, 90)))
            p.drawText(8, 20, f"CASCADE {ms:.0f}ms/pass")
            p.setPen(QPen(QColor(255, 200, 40, 90), 1))
            for x0, y0, x1, y1 in props:
                p.drawRect(QRectF(x0 * W, y0 * H,
                                  (x1 - x0) * W, (y1 - y0) * H))
            p.setFont(QFont("Consolas", 11, QFont.Weight.Bold))
            for x0, y0, x1, y1, name, sc_, mi in dets:
                # primary model = green, secondary = blue
                col = QColor(80, 255, 80) if mi == 0 else QColor(90, 170, 255)
                p.setPen(QPen(col, 3))
                r = QRectF(x0 * W, y0 * H, (x1 - x0) * W, (y1 - y0) * H)
                p.drawRect(r)
                p.drawText(int(r.left()), max(int(r.top()) - 6, 14),
                           f"{name} {sc_ * 100:.0f}%")
            p.end()

    panel = Panel()
    panel.show()
    ov = Overlay()
    ov.show()

    def _shield(retry: bool = True) -> None:
        """Best-effort belt-and-suspenders: also hide the overlay from
        CAPTUREBLT grabs (mss/obs). grab_screen() already excludes it."""
        u32 = ctypes.windll.user32
        hwnd = int(ov.winId())
        # NOTE: no WDA_MONITOR fallback — it blacks a fullscreen window
        # out of EVERY capture, which blinded the detector entirely.
        if u32.SetWindowDisplayAffinity(hwnd, 0x11):   # EXCLUDEFROMCAPTURE
            print("overlay capture-excluded (WDA_EXCLUDEFROMCAPTURE)")
            return
        if retry:                       # HWND may not be realized yet
            QTimer.singleShot(400, lambda: _shield(retry=False))
        # else: harmless — grab_screen() already excludes layered windows

    _shield()
    print("cascade ui — pick S1/S2/S3 models; Esc quits")
    try:
        app.exec()
    finally:
        stop.set()


if __name__ == "__main__":
    main()
