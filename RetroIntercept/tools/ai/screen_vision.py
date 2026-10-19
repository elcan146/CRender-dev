"""Screen Vision — faithful desktop port of RetroIntercept's Terminator
pipeline. Same decode (raw ONNX output, letterbox undo, per-class NMS 0.45,
crop-fill rejection, cross-crop dedup, sqrt weapon remap, optional stage-2
classifier), same tracker (IoU/EMA/coast/lock), same HUD (8x8 glyph atlas,
columns, reticle, OBB brackets, sight lines) — drawn on a transparent
click-through overlay over your live desktop, or in a styled mirror window.

  python tools\\ai\\screen_vision.py                    # Terminator HUD overlay
  python tools\\ai\\screen_vision.py --style term       # Terminator base mirror
  python tools\\ai\\screen_vision.py --style retro      # RetroFX mirror
  python tools\\ai\\screen_vision.py --style ascii      # ASCII mirror
  python tools\\ai\\screen_vision.py --monitor 2
  python tools\\ai\\screen_vision.py --all
  python tools\\ai\\screen_vision.py --conf 0.30 --stride 2

Hotkeys (polled globally, like in-game): F8 cycle palette, F9 cycle style,
F10 panic hide/show, F11 reload retro.json, End quit.
Defaults come from config/retro.json. Windows are WDA_EXCLUDEFROMCAPTURE so
capture never sees its own HUD (no feedback loop).
"""
import argparse
import ctypes
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"      # physical px == capture px

import cv2
import numpy as np
import onnxruntime as ort
from PIL import ImageGrab
from PyQt6.QtCore import Qt, QRectF, QTimer
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PyQt6.QtWidgets import QApplication, QWidget

try:
    import mss                                    # fast BitBlt grabber
except Exception:
    mss = None

try:
    import dxcam                                  # DXGI desktop dup, 60+ fps
except Exception:
    dxcam = None

ort.set_default_logger_severity(4)  # fatal only — silences CUDA probe noise
ROOT = Path(__file__).resolve().parents[2]
CFG_PATH = ROOT / "config" / "retro.json"
# Auto-pick the best EP: DirectML (GPU — same backend the in-game ORT path
# uses) > CUDA > CPU. DML init is fast and needs no CUDA runtime DLLs.
AVAIL = ort.get_available_providers()
PROVIDERS = [p for p in ("DmlExecutionProvider", "CUDAExecutionProvider",
                         "CPUExecutionProvider") if p in AVAIL]

# ---------------------------------------------------------------- tracker ----
# Faithful port of src/term/Tracker.cpp.
IOU_MATCH, DIST_MATCH, EMA, COAST, LOCK_AFTER = 0.25, 0.08, 0.35, 0.35, 1.2
SESS1_BASE = 1000                                   # sess-1 cls offset (kSess1Base)


def _iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = ((a[2] - a[0]) * (a[3] - a[1])
             + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


class Tracker:
    def __init__(self) -> None:
        self.tracks: list[dict] = []
        self._next_id = 1

    def update(self, dets, dt, max_tracks) -> None:
        dt = min(max(dt, 0.0), 1.0)
        used = set()
        for d in sorted(dets, key=lambda x: -x["score"]):
            box = d["box"]
            best, best_iou = -1, IOU_MATCH
            for i, tr in enumerate(self.tracks):
                if i in used:
                    continue
                v = _iou(box, tr["box"])
                if v > best_iou:
                    best_iou, best = v, i
            if best < 0:  # centroid-distance fallback
                cx, cy = (box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5
                best_d = DIST_MATCH * DIST_MATCH
                for i, tr in enumerate(self.tracks):
                    if i in used:
                        continue
                    dx = cx - (tr["box"][0] + tr["box"][2]) * 0.5
                    dy = cy - (tr["box"][1] + tr["box"][3]) * 0.5
                    if dx * dx + dy * dy < best_d:
                        best_d, best = dx * dx + dy * dy, i
            if best >= 0:
                tr = self.tracks[best]
                tr["box"] = tuple(tr["box"][k] + (box[k] - tr["box"][k]) * EMA
                                  for k in range(4))
                tr.update(unseen=0.0, cls=d["cls"], score=d["score"],
                          ang=d.get("ang", 0.0))
                tr["hits"] += 1
                used.add(best)
            elif len(self.tracks) < max_tracks:
                self.tracks.append({"box": box, "id": self._next_id, "age": 0.0,
                                    "unseen": 0.0, "lock_age": 0.0,
                                    "locked": False, "cls": d["cls"],
                                    "score": d["score"],
                                    "ang": d.get("ang", 0.0), "hits": 1})
                self._next_id = self._next_id % 99 + 1
                used.add(len(self.tracks) - 1)
        kept = []
        for i, tr in enumerate(self.tracks):
            tr["age"] += dt
            if i not in used:
                tr["unseen"] += dt
            if not tr["locked"] and tr["age"] >= LOCK_AFTER:
                tr["locked"], tr["lock_age"] = True, 0.0
            elif tr["locked"]:
                tr["lock_age"] += dt
            if tr["unseen"] <= COAST and len(kept) < max_tracks:
                kept.append(tr)
        self.tracks = kept


# ---------------------------------------------------------------- config -----
COCO = ("person bicycle car motorcycle airplane bus train truck boat "
        "traffic light fire hydrant stop sign parking meter bench bird cat "
        "dog horse sheep cow elephant bear zebra giraffe backpack umbrella "
        "handbag tie suitcase frisbee skis snowboard sports ball kite "
        "baseball bat baseball glove skateboard surfboard tennis racket "
        "bottle wine glass cup fork knife spoon bowl banana apple sandwich "
        "orange broccoli carrot hot dog pizza donut cake chair couch "
        "potted plant bed dining table toilet tv laptop mouse remote "
        "keyboard cell phone microwave oven toaster sink refrigerator book "
        "clock vase scissors teddy bear hair drier toothbrush").split(" ")
SESS1_NAMES = ["GUN", "KNIFE"]


def _class_id(tok: str) -> int:
    tok = tok.strip()
    if not tok:
        return -1
    if tok.isdigit():
        return int(tok)
    low = tok.lower()
    return COCO.index(low) if low in COCO else -1


def _csv_ids(csv: str) -> list[int]:
    return [i for i in (_class_id(t) for t in (csv or "").split(",")) if i >= 0][:16]


def _label(cls: int) -> str:
    if cls >= SESS1_BASE:
        i = cls - SESS1_BASE
        return SESS1_NAMES[i] if i < len(SESS1_NAMES) else f"DET {i:02d}"
    return COCO[cls] if 0 <= cls < 80 else f"OBJ {cls:02d}"


class Config:
    """retro.json values; CLI args override. Reloadable subset via load()."""

    def __init__(self, args) -> None:
        self.args = args
        self.load()

    def load(self) -> None:
        cfg = {}
        if CFG_PATH.exists():
            try:
                cfg = json.loads(CFG_PATH.read_text())
            except Exception:
                pass
        a = self.args
        # detector (SetConfig clamps preserved)
        self.model = a.model or cfg.get("DetModel", "yolov8n.onnx")
        self.model2 = a.model2 or cfg.get("DetModel2", "det_weapons.onnx")
        self.model3 = a.model3 or cfg.get("DetModel3", "")
        c = a.conf if a.conf is not None else cfg.get("DetConf", 0.35)
        self.conf = min(max(c, 0.05), 0.95)
        s = a.stride if a.stride is not None else cfg.get("DetStride", 1)
        self.stride = min(max(s, 1), 30)
        cc = a.cls_conf if a.cls_conf is not None else cfg.get("DetClsConf", 0.85)
        self.cls_conf = min(max(cc, 0.50), 0.99)
        self.max_dets = 32
        self.max_targets = int(cfg.get("HudMaxTargets", 6))
        self.classes = _csv_ids(a.classes or cfg.get("DetClasses", "person")) or [0]
        self.weapons = set(_csv_ids(a.weapons or cfg.get("DetWeapon", "")))
        self.classes2 = _csv_ids(a.classes2 or cfg.get("DetClasses2", "0,1")) or [0, 1]
        self.weapons2 = set(_csv_ids(a.weapons2 or cfg.get("DetWeapon2", "0,1")))
        self.person_cls = 0                        # g_cfg.personClass
        # HUD toggles
        self.hud = bool(cfg.get("TermHud", True))
        self.brackets = bool(cfg.get("TermBrackets", True))
        # AI stylizer (AiPass)
        self.ai_model = cfg.get("AiModel", "")
        self.ai_w = int(cfg.get("AiWidth", 640))
        self.ai_h = int(cfg.get("AiHeight", 360))
        self.ai_temporal = int(cfg.get("AiTemporal", 0))
        self.ai_blend = float(cfg.get("AiBlend", 0.5))
        self.ai_stride = min(max(int(cfg.get("AiStride", 1)), 1), 30)
        self.term_base = bool(cfg.get("TermBase", True))
        self.term_base_onnx = bool(cfg.get("TermBaseOnnx", False))
        self.edge_gain = float(cfg.get("TermEdgeGain", 1.0))
        tint = cfg.get("TermTint", [1.0, 0.0, 0.0])
        self.tint = tuple(float(v) for v in tint[:3])
        lens = cfg.get("TerminatorLens", {})
        self.lens_on = bool(cfg.get("TermLens", True))
        self.term_crt = float(lens.get("Crt", 0.10))
        self.term_vig = float(lens.get("Vignette", 0.45))
        self.term_scan = float(lens.get("Scanlines", 0.35))
        # retro/ascii params
        self.pixel = int(cfg.get("PixelPx", 4))
        self.cell = max(int(cfg.get("CellPx", 8)), 4)
        self.crt = float(cfg.get("Crt", 0.12))
        self.vignette = float(cfg.get("Vignette", 0.35))
        self.palette = str(cfg.get("Palette", "green"))

    def is_weapon(self, cls: int) -> bool:
        return ((cls - SESS1_BASE) in self.weapons2 if cls >= SESS1_BASE
                else cls in self.weapons)


# -------------------------------------------------------------- sessions -----
class Sess:
    """One ORT session mirroring Detect.cpp Sess: input [1,3,H,W] fp32, output
    [1,R,N] / [1,N,R] (detectors) or [1,C] logits (sess2 classifier)."""

    def __init__(self, path: Path, idx: int) -> None:
        self.idx = idx
        self.path = path
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = max(os.cpu_count() - 2, 2)
        self.sess = ort.InferenceSession(str(path), sess_options=so,
                                         providers=PROVIDERS)
        inp = self.sess.get_inputs()[0]
        self.in_name = inp.name
        self.in_names = [i.name for i in self.sess.get_inputs()]
        shp = [int(v) for v in inp.shape]
        if len(shp) != 4 or shp[1] != 3:
            raise ValueError(f"{path.name}: detector input must be [1,3,H,W]")
        self.inH, self.inW = shp[2], shp[3]
        self.classifier = idx == 2
        self.obb = "obb" in path.name.lower()      # in-game filename sniff
        od = [int(v) for v in self.sess.get_outputs()[0].shape]
        if self.classifier:
            if len(od) != 2 or od[0] != 1 or not (1 <= od[1] <= 8):
                raise ValueError(f"{path.name}: classifier output must be [1,C]")
            self.nCls = od[1]
        else:
            r = min(od[1], od[2])
            self.nCls = r - (5 if self.obb else 4)
        self.rect = (0.0, 0.0, 1.0, 1.0)

    def run(self, tensor) -> np.ndarray:
        return self.sess.run(None, {self.in_name: tensor})[0]


def make_input(frame, rect, in_w: int, in_h: int) -> np.ndarray:
    """Port of kPsInSrc: aspect-preserving letterbox of the src-rect crop into
    a centered content region of the model input; black borders; NCHW fp32."""
    rx, ry, rw, rh = rect
    fh, fw = frame.shape[:2]
    x0, y0 = max(int(rx * fw), 0), max(int(ry * fh), 0)
    x1, y1 = min(int((rx + rw) * fw), fw), min(int((ry + rh) * fh), fh)
    crop = frame[y0:y1, x0:x1]
    out = np.zeros((in_h, in_w, 3), np.float32)
    if crop.size:
        in_a = in_w / in_h
        bb_a = crop.shape[1] / crop.shape[0]       # bbA = crop dims
        cw = min(1.0, bb_a / in_a)
        ch = min(1.0, in_a / bb_a)
        tw = max(int(round(cw * in_w)), 1)
        th = max(int(round(ch * in_h)), 1)
        rs = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_LINEAR)
        ox, oy = (in_w - tw) // 2, (in_h - th) // 2
        out[oy:oy + th, ox:ox + tw] = rs.astype(np.float32) / 255.0
    return out[..., ::-1].transpose(2, 0, 1)[None].copy()


def decode(s: Sess, out: np.ndarray, conf: float, classes: list[int],
           cls_base: int, bb_w: int, bb_h: int, cls_set=None) -> list[dict]:
    """Exact port of Detect.cpp Decode(): layout sniff, per-class best score,
    letterbox undo + src-rect map, OBB corner rotation, size/crop-fill gates,
    score-sort, per-class NMS 0.45. Returns normalized dets (<=64 raw)."""
    out = out[0]
    if out.shape[0] > out.shape[1]:
        out = out.T                                # -> (R, N)
    R, N = out.shape
    need_rows = s.nCls + (5 if s.obb else 4)
    if R != need_rows or N <= 0:
        return []
    valid = [c for c in classes if 0 <= c < s.nCls]
    if not valid:
        return []

    # Ultralytics exports input-PIXEL coords; sniff normalized vs pixels.
    mx = float(out[:4, :min(N, 64)].max())
    x_s = 1.0 / s.inW if mx > 1.5 else 1.0
    y_s = 1.0 / s.inH if mx > 1.5 else 1.0

    scores = out[4:4 + s.nCls][valid]              # (K, N)
    bi = scores.argmax(0)
    best = scores[bi, np.arange(N)]
    cand = np.nonzero(best > conf)[0]              # strict > like best=conf seed

    in_a = s.inW / s.inH
    rx, ry, rw, rh = s.rect
    src_h = rh * bb_h
    bb_a = (rw * bb_w) / src_h if src_h > 0 else in_a
    cw = min(bb_a / in_a, 1.0)
    ch = min(in_a / bb_a, 1.0)
    px, py = (1.0 - cw) * 0.5, (1.0 - ch) * 0.5
    c01 = lambda v: min(max(v, 0.0), 1.0)

    raw = []
    for c in cand:
        cls = valid[int(bi[c])]
        sc = float(best[c])
        cx, cy = float(out[0, c]) * x_s, float(out[1, c]) * y_s
        bw, bh = float(out[2, c]) * x_s, float(out[3, c]) * y_s
        ang = 0.0
        if s.obb:
            ang = float(out[4 + s.nCls, c])
            ca, sa = math.cos(ang), math.sin(ang)
            ax, ay = ca * bw * 0.5, sa * bw * 0.5
            bx, by = -sa * bh * 0.5, ca * bh * 0.5
            u0 = u1 = rx + (cx - px) / cw * rw
            v0 = v1 = ry + (cy - py) / ch * rh
            for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                mu = rx + (cx + ax * su + bx * sv - px) / cw * rw
                mv = ry + (cy + ay * su + by * sv - py) / ch * rh
                u0, u1 = min(u0, mu), max(u1, mu)
                v0, v1 = min(v0, mv), max(v1, mv)
        else:
            u0 = rx + (cx - bw * 0.5 - px) / cw * rw
            u1 = rx + (cx + bw * 0.5 - px) / cw * rw
            v0 = ry + (cy - bh * 0.5 - py) / ch * rh
            v1 = ry + (cy + bh * 0.5 - py) / ch * rh
        d = {"box": (c01(u0), c01(v0), c01(u1), c01(v1)),
             "score": math.sqrt(sc) if s.idx == 1 else sc,
             "cls": cls + cls_base, "ang": ang}
        if d["box"][2] - d["box"][0] < 0.002 or d["box"][3] - d["box"][1] < 0.002:
            continue
        # Crop-fill rejection (sess1 partial crops only).
        if s.idx == 1 and rw < 0.99 and \
                (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]) \
                > 0.60 * rw * rh:
            continue
        raw.append(d)
        if len(raw) >= 64:
            break

    raw.sort(key=lambda d: -d["score"])
    # Per-class NMS IoU 0.45 (suppression only from strictly-higher scores —
    # `raw[j].score > raw[i].score` skips, equal scores die like in-game).
    dead = [False] * len(raw)
    for i in range(len(raw)):
        if dead[i]:
            continue
        for j in range(i + 1, len(raw)):
            if (dead[j] or raw[j]["cls"] != raw[i]["cls"]
                    or raw[j]["score"] > raw[i]["score"]):
                continue
            if _iou(raw[i]["box"], raw[j]["box"]) > 0.45:
                dead[j] = True
    return [d for i, d in enumerate(raw) if not dead[i]]


class Detector:
    """The Run() pass: sess0 full frame; sess1 (when both live) defers to
    torso-anchored crops of this pass's person dets (max 3), full-frame
    fallback; cross-crop dedup; optional sess2 classifier gate on weapons."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.sess: list[Sess | None] = [None, None, None]
        for i, name in enumerate((cfg.model, cfg.model2, cfg.model3)):
            if not name:
                continue
            p = Path(name)
            if not p.is_absolute():
                p = ROOT / "config" / name
            if not p.exists():
                print(f"sess{i}: model not found: {p} — off")
                continue
            try:
                t0 = time.perf_counter()
                self.sess[i] = Sess(p, i)
                actual = self.sess[i].sess.get_providers()[0]
                print(f"sess{i}: {p.name} [1,3,{self.sess[i].inH},"
                      f"{self.sess[i].inW}] nCls={self.sess[i].nCls}"
                      + (" classifier" if i == 2 else "")
                      + (" obb" if self.sess[i].obb else "")
                      + f" [{actual}]"
                      + f" ({(time.perf_counter() - t0) * 1000:.0f} ms)")
            except Exception as e:
                print(f"sess{i}: {p.name} failed: {e} — off")

    @property
    def live(self) -> bool:
        return any(s is not None for s in self.sess[:2])

    def infer(self, frame) -> list[dict]:
        h, w = frame.shape[:2]
        s0, s1, s2 = self.sess
        dets: list[dict] = []
        defer = s0 is not None and s1 is not None
        # sess0 — always the full frame.
        if s0 is not None:
            s0.rect = (0.0, 0.0, 1.0, 1.0)
            out = s0.run(make_input(frame, s0.rect, s0.inW, s0.inH))
            dets += decode(s0, out, self.cfg.conf, self.cfg.classes, 0, w, h)
        if s1 is not None:
            # Person-anchored crops when deferring (torso expansion, max 3),
            # else one full-frame pass.
            if defer:
                rects = []
                for d in dets:
                    if d["cls"] != self.cfg.person_cls or len(rects) >= 3:
                        continue
                    px0, py0, px1, py1 = d["box"]
                    pw, ph = px1 - px0, py1 - py0
                    rx = max(px0 - pw * 0.45, 0.0)
                    ry = max(py0 - ph * 0.15, 0.0)
                    rects.append((rx, ry, min(pw * 1.90, 1.0 - rx),
                                  min(ph * 1.15, 1.0 - ry)))
                if not rects:
                    rects = [(0.0, 0.0, 1.0, 1.0)]
            else:
                rects = [(0.0, 0.0, 1.0, 1.0)]
            for rect in rects:
                s1.rect = rect
                out = s1.run(make_input(frame, rect, s1.inW, s1.inH))
                dets += decode(s1, out, self.cfg.conf, self.cfg.classes2,
                               SESS1_BASE, w, h)
        # Cross-crop dedup (same class; IoU 0.3 weapons / 0.6 other).
        dets.sort(key=lambda d: -d["score"])
        out_d = []
        for d in dets:
            lim = 0.3 if d["cls"] >= SESS1_BASE else 0.6
            if any(k["cls"] == d["cls"] and _iou(d["box"], k["box"]) > lim
                   for k in out_d):
                continue
            out_d.append(d)
        dets = out_d[: self.cfg.max_dets]
        # Stage-2 classifier gate: P(weapon|25%-expanded crop) >= clsConf.
        if s2 is not None and dets:
            kept = []
            for d in dets:
                if not self.cfg.is_weapon(d["cls"]):
                    kept.append(d)
                    continue
                x0, y0, x1, y1 = d["box"]
                mx, my = (x1 - x0) * 0.25, (y1 - y0) * 0.25
                rx, ry = max(x0 - mx, 0.0), max(y0 - my, 0.0)
                rect = (rx, ry, min(x1 - x0 + mx * 2, 1.0 - rx),
                        min(y1 - y0 + my * 2, 1.0 - ry))
                s2.rect = rect
                logits = s2.run(make_input(frame, rect, s2.inW, s2.inH))[0]
                if logits.size == 1:
                    p = 1.0 / (1.0 + math.exp(-float(logits[0])))
                else:
                    e = np.exp(logits - logits.max())
                    p = float(e[1] / e.sum())      # softmax class-1
                if p >= self.cfg.cls_conf or p < 0.0:
                    kept.append(d)
            dets = kept
        return dets


# --------------------------------------------------------------- stylizer ----
class AiStylizer:
    """Port of AiPass: NCHW fp32 stretch-sampled input at AiW x AiH, fp32 CHW
    output bilinear-gathered back to frame res, temporal modes, Terminator
    look re-grade (e = g/0.7), AiStride skip. lens applied by caller."""

    def __init__(self, cfg: Config) -> None:
        p = Path(cfg.ai_model)
        if not p.is_absolute():
            p = ROOT / "config" / cfg.ai_model
        if not cfg.ai_model or not p.exists():
            raise FileNotFoundError(f"AI model missing: {p}")
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = max(os.cpu_count() - 2, 2)
        self.s = ort.InferenceSession(str(p), sess_options=so,
                                      providers=PROVIDERS)
        ins = self.s.get_inputs()
        self.in_names = [i.name for i in ins]
        shp = [int(v) for v in ins[0].shape]
        # cfg dims when the model is dynamic; declared dims otherwise.
        self.h = shp[2] if shp[2] > 0 else cfg.ai_h
        self.w = shp[3] if shp[3] > 0 else cfg.ai_w
        self.cfg = cfg
        self.prev = None                            # last raw output (model res)
        self.last_img = None                        # composed BGR frame
        self.n = 0

    def __call__(self, frame):
        cfg = self.cfg
        self.n += 1
        if self.last_img is not None and self.n % cfg.ai_stride != 0:
            return self.last_img
        rs = cv2.resize(frame, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        t = rs.astype(np.float32)[..., ::-1].transpose(2, 0, 1)[None] / 255.0
        feed = {self.in_names[0]: t}
        if len(self.in_names) > 1 and cfg.ai_temporal == 2:
            feed[self.in_names[1]] = (self.prev if self.prev is not None
                                      else np.zeros_like(t))
        out = self.s.run(None, feed)[0]             # [1,3,H,W] fp32
        cur = out.copy()
        if cfg.ai_temporal == 1 and self.prev is not None:
            cur = cur * (1.0 - cfg.ai_blend) + self.prev * cfg.ai_blend
        self.prev = cur if cfg.ai_temporal != 0 else out
        c = np.clip(out[0].transpose(1, 2, 0), 0, 1)     # HWC RGB
        if cfg.term_base and cfg.term_base_onnx:
            # lookOn re-grade: e = g/0.7, e2 = e*gain, c = max(r-e,0)*tint + ...
            e = np.clip(c[..., 1] / 0.7, 0, 1)
            e2 = np.clip(e * cfg.edge_gain, 0, 1)
            tr, tg, tb = cfg.tint
            base = np.clip(c[..., 0] - e, 0, 1)
            c = np.stack([base * tr + e2, base * tg + 0.7 * e2,
                          base * tb + 0.7 * e2], -1)
        img = (np.clip(c, 0, 1) * 255).astype(np.uint8)[..., ::-1]  # -> BGR
        h, w = frame.shape[:2]
        self.last_img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        return self.last_img


# --------------------------------------------------------------- styling -----
_BARREL_CACHE: dict = {}


def _barrel(w: int, h: int, k: float):
    key = (w, h, k)
    m = _BARREL_CACHE.get(key)
    if m is None:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        nx, ny = (xx / w - 0.5) * 2, (yy / h - 0.5) * 2
        r2 = nx * nx + ny * ny
        mx = ((nx * (1 + k * r2)) * 0.5 + 0.5) * w
        my = ((ny * (1 + k * r2)) * 0.5 + 0.5) * h
        m = (mx.astype(np.float32), my.astype(np.float32))
        _BARREL_CACHE[key] = m
    return m


def _crt(img, k: float):
    if k <= 0:
        return img
    mx, my = _barrel(img.shape[1], img.shape[0], k)
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)


def _scanlines(img, strength):
    img[1::2] *= 1.0 - strength
    return img


def _vignette(img, strength):
    h, w = img.shape[:2]
    y, x = np.mgrid[0:h, 0:w]
    d = np.sqrt(((x / w - 0.5) * 2) ** 2 + ((y / h - 0.5) * 2) ** 2)
    v = 1.0 - strength * np.clip(d - 0.6, 0, 1) ** 2
    return img * v[..., None]


def _term_lens(img, cfg: Config):
    """TerminatorLens pass: CRT barrel + scanlines + vignette."""
    img = _crt(img, cfg.term_crt)
    img = img.astype(np.float32) / 255.0
    img = _scanlines(img, cfg.term_scan)
    img = _vignette(img, cfg.term_vig)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def _term_hlsl(frame, cfg: Config):
    """HLSL fallback base (TermBaseOnnx=false): Sobel edges over dim bg."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    ex = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    ey = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    e = np.clip(np.hypot(ex, ey) * cfg.edge_gain, 0, 1)
    tr, tg, tb = cfg.tint
    out = np.zeros((*g.shape, 3), np.float32)      # BGR buffer
    out[..., 0] = np.clip(g * 0.25 * tb + e * 0.7 * tb + e * tb * 0.0, 0, 1)
    out[..., 1] = np.clip(g * 0.25 * tg + e * 0.7 * tg + e * 0.7 * tg * 0.0, 0, 1)
    out[..., 2] = np.clip(g * 0.25 * tr + e * tr, 0, 1)
    # equivalent: c = max(r? - e,0)*tint + (e2, .7e2, .7e2) with bg term
    base = g * 0.25
    e2 = np.clip(e * cfg.edge_gain, 0, 1)
    out = np.stack([np.clip(base * tb + 0.7 * e2 * tb, 0, 1),
                    np.clip(base * tg + 0.7 * e2 * tg, 0, 1),
                    np.clip(base * tr + e2 * tr, 0, 1)], -1)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def style_term(stylizer: "AiStylizer | None"):
    def fn(frame, cfg):
        img = stylizer(frame) if stylizer is not None else _term_hlsl(frame, cfg)
        return _term_lens(img, cfg) if cfg.lens_on else img
    return fn


def style_retro(frame, cfg: Config):
    """RetroFX: pixelate + 3-bit palette + CRT + scanlines + vignette."""
    px = max(cfg.pixel, 1)
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (max(w // px, 1), max(h // px, 1)),
                       interpolation=cv2.INTER_LINEAR)
    out = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    out = (out.astype(np.uint16) // 85 * 85).astype(np.uint8)
    out = _crt(out, cfg.crt)
    f = out.astype(np.float32) / 255.0
    f = _scanlines(f, 0.4)
    f = _vignette(f, cfg.vignette)
    return (np.clip(f, 0, 1) * 255).astype(np.uint8)


RAMP = " .'`^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
_TILE_CACHE: dict = {}


def _glyph_tiles(c: int) -> np.ndarray:
    lut = _TILE_CACHE.get(c)
    if lut is not None:
        return lut
    lut = np.zeros((256, c, c), np.float32)
    for v in range(256):
        ch = RAMP[min(v * len(RAMP) // 256, len(RAMP) - 1)]
        if ch == " ":
            continue
        tile = np.zeros((c, c), np.uint8)
        cv2.putText(tile, ch, (0, c - 1), cv2.FONT_HERSHEY_PLAIN,
                    c / 9.0, 255, 1, cv2.LINE_AA)
        lut[v] = tile.astype(np.float32) / 255.0
    _TILE_CACHE[c] = lut
    return lut


def style_ascii(frame, cfg: Config):
    """GPU ASCII pass look: CellPx square grid, mean luminance -> ramp glyph,
    palette or per-cell mean color."""
    h, w = frame.shape[:2]
    c = max(cfg.cell, 4)
    gh, gw = h // c, w // c
    if gh == 0 or gw == 0:
        return frame
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(g, (gw, gh), interpolation=cv2.INTER_AREA)
    tiles = _glyph_tiles(c)
    grid = tiles[small]                            # (gh, gw, c, c)
    pal = {"green": (0, 1.0, 0.31), "amber": (1.0, 0.69, 0),
           "cyan": (0, 0.86, 1.0), "mono": (0.78, 0.78, 0.78)}.get(
        cfg.palette.lower())
    if pal is None:                                # "color": per-cell mean BGR
        col = cv2.resize(frame, (gw, gh),
                         interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        grid = grid[..., None] * col[:, :, None, None, :]
    else:
        grid = grid[..., None] * np.asarray(pal, np.float32)
    out = grid.transpose(0, 2, 1, 3, 4).reshape(gh * c, gw * c, 3) * 255
    full = np.zeros((h, w, 3), np.uint8)
    full[: gh * c, : gw * c] = np.clip(out, 0, 255).astype(np.uint8)
    return full


# ------------------------------------------------------------ HUD painter ----
COL_NORM = (255, 209, 209, 242)
COL_RED = (255, 31, 20, 255)


def _qcol(c, a_mul=1.0):
    return QColor(int(c[0]), int(c[1]), int(c[2]), min(int(c[3] * a_mul), 255))


class HudPainter:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self._font = QFont("Segoe UI")
        self._font.setBold(True)

    def draw(self, p: QPainter, w: int, h: int, cfg: Config, state: dict) -> None:
        if not cfg.hud or w < 64 or h < 64:
            return
        s = max((h + 240) // 480, 1)
        line_h = 10 * s
        self._font.setPixelSize(max(8 * s, 10))
        p.setFont(self._font)
        if cfg.brackets and state["tracks"]:
            self._brackets(p, state["tracks"], w, h, s, line_h)

    def _rect(self, p, x0, y0, x1, y1, col):
        p.fillRect(QRectF(x0, y0, x1 - x0, y1 - y0), _qcol(col))

    def _brackets(self, p, tracks, w, h, s, line_h):
        for tr in tracks:
            x0, y0, x1, y1 = (tr["box"][0] * w, tr["box"][1] * h,
                              tr["box"][2] * w, tr["box"][3] * h)
            k = min(max(tr["age"] / 0.25, 0.0), 1.0)
            ease = 1.0 - (1.0 - k) ** 3
            sc = 1.0 + 0.6 * (1.0 - ease)
            cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
            hw, hh = (x1 - x0) * 0.5 * sc, (y1 - y0) * 0.5 * sc
            x0, x1, y0, y1 = cx - hw, cx + hw, cy - hh, cy + hh
            flash = (tr["locked"] and tr["lock_age"] < 0.6
                     and (tr["lock_age"] * 6.0) % 1.0 < 0.5)
            col = list(COL_RED if flash else COL_NORM)
            col[3] *= ease * (0.5 if tr["unseen"] > 0.0 else 1.0)
            lx = x0
            ang = tr.get("ang", 0.0)
            if ang != 0.0:
                # OBB: envelope -> true w/h, then rotate the corners (Hud.cpp).
                ca, sa = abs(math.cos(ang)), abs(math.sin(ang))
                det2 = ca * ca - sa * sa
                ex, ey = x1 - x0, y1 - y0
                rw, rh = ex, ey
                if abs(det2) > 0.05:
                    rw = max((ex * ca - ey * sa) / det2, 2.0)
                    rh = max((ey * ca - ex * sa) / det2, 2.0)
                ax, ay = math.cos(ang) * rw * 0.5, math.sin(ang) * rw * 0.5
                bx, by = -math.sin(ang) * rh * 0.5, math.cos(ang) * rh * 0.5
                q = [(cx + ax * su + bx * sv, cy + ay * su + by * sv)
                     for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
                for e_ in range(4):
                    xa, ya = q[e_]
                    xb, yb = q[(e_ + 1) & 3]
                    edx, edy = xb - xa, yb - ya
                    elen = math.hypot(edx, edy)
                    if elen < 1.0:
                        continue
                    for d in range(min(int(elen / (3.0 * s)), 96)):
                        u = d * 3.0 * s
                        px, py = xa + edx * (u / elen), ya + edy * (u / elen)
                        self._rect(p, px - s, py - s, px + s, py + s, col)
                lx, ly = q[0]
                for qx, qy in q[1:]:
                    if qy < ly:
                        ly, lx = qy, qx
                ly -= line_h
            else:
                m2 = min(hw, hh) * 2.0
                L = max(m2 * 0.25, 4.0 * s)
                th = s
                for r in ((x0, y0, x0 + L, y0 + th), (x0, y0, x0 + th, y0 + L),
                          (x1 - L, y0, x1, y0 + th), (x1 - th, y0, x1, y0 + L),
                          (x0, y1 - th, x0 + L, y1), (x0, y1 - L, x0 + th, y1),
                          (x1 - L, y1 - th, x1, y1),
                          (x1 - th, y1 - L, x1, y1)):
                    self._rect(p, *r, col)
                ly = y0 - line_h if y0 - line_h >= 0 else y1 + 2.0 * s
            pct = int(tr["score"] * 100.0 + 0.5)
            txt = (f"{_label(tr['cls'])} {pct:02d}% LOCK" if tr["locked"]
                   else f"{_label(tr['cls'])} {pct:02d}%")
            p.setPen(_qcol(COL_RED if tr["locked"] else col))
            p.drawText(int(lx), int(ly + 8 * s), txt)


# ---------------------------------------------------------------- windows ----
def _exclude_from_capture(widget: QWidget) -> None:
    try:
        ctypes.windll.user32.SetWindowDisplayAffinity(int(widget.winId()),
                                                    0x00000011)
    except Exception:
        pass


def _geo(args, app) -> tuple[int, int, int, int]:
    if args.all:
        u = ctypes.windll.user32
        return (u.GetSystemMetrics(76), u.GetSystemMetrics(77),
                u.GetSystemMetrics(78), u.GetSystemMetrics(79))
    screens = app.screens()
    idx = min(max(args.monitor - 1, 0), len(screens) - 1)
    g = screens[idx].geometry()
    return g.x(), g.y(), g.width(), g.height()


STYLES = ("hud", "term", "retro", "ascii")
PALETTES = ("green", "amber", "cyan", "mono", "color")


class Overlay(QWidget):
    """Transparent click-through overlay. Draws the styled mirror frame (when
    the style restyles pixels) then the Terminator HUD on top."""

    VK = {"end": 0x23, "f8": 0x77, "f9": 0x78, "f10": 0x79, "f11": 0x7A}

    def __init__(self, geo, hud: HudPainter, cfg: Config, shared, lock,
                 stop) -> None:
        super().__init__(flags=(Qt.WindowType.FramelessWindowHint
                                | Qt.WindowType.WindowStaysOnTopHint
                                | Qt.WindowType.Tool))
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.hud, self.cfg = hud, cfg
        self.shared, self.lock, self.stop = shared, lock, stop
        self._down = set()
        self.setGeometry(*geo)
        qt = QTimer(self)
        qt.timeout.connect(self._tick)
        qt.start(16)                              # ~60 Hz repaint

    def _hotkey(self, name: str) -> bool:
        down = bool(ctypes.windll.user32.GetAsyncKeyState(self.VK[name])
                    & 0x8000)
        edge = down and name not in self._down
        if down:
            self._down.add(name)
        else:
            self._down.discard(name)
        return edge

    def _tick(self) -> None:
        u = ctypes.windll.user32
        if u.GetAsyncKeyState(self.VK["end"]) & 0x8000:
            QApplication.instance().quit()
            return
        if self._hotkey("f9"):
            with self.lock:
                self.shared["style"] = STYLES[(STYLES.index(
                    self.shared["style"]) + 1) % len(STYLES)]
        if self._hotkey("f8"):
            with self.lock:
                self.shared["palette"] = PALETTES[(PALETTES.index(
                    self.shared["palette"]) + 1) % len(PALETTES)]
        if self._hotkey("f10"):                    # panic hide/show
            self.setVisible(not self.isVisible())
        if self._hotkey("f11"):
            with self.lock:
                self.shared["reload"] = True
        self.update()

    def _snapshot(self) -> dict:
        with self.lock:
            return {"tracks": [dict(t) for t in self.shared["tracks"]],
                    "styled": self.shared.get("styled")}

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        st = self._snapshot()
        if st["styled"] is not None:
            p.drawImage(0, 0, st["styled"].scaled(self.width(), self.height()))
        self.hud.draw(p, self.width(), self.height(), self.cfg, st)
        p.end()


# ----------------------------------------------------------------- capture ---
class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", ctypes.c_ulong), ("biWidth", ctypes.c_long),
                ("biHeight", ctypes.c_long), ("biPlanes", ctypes.c_ushort),
                ("biBitCount", ctypes.c_ushort),
                ("biCompression", ctypes.c_ulong),
                ("biSizeImage", ctypes.c_ulong),
                ("biXPelsPerMeter", ctypes.c_long),
                ("biYPelsPerMeter", ctypes.c_long),
                ("biClrUsed", ctypes.c_ulong),
                ("biClrImportant", ctypes.c_ulong)]


class _BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", _BITMAPINFOHEADER),
                ("bmiColors", ctypes.c_ulong * 3)]


class WinGrabber:
    """Dependency-free fast capture: BitBlt + GetDIBits into a top-down DIB.
    ~10x faster than PIL ImageGrab. Returns a fresh BGR ndarray per call."""

    def __init__(self, x: int, y: int, w: int, h: int) -> None:
        u, g = ctypes.windll.user32, ctypes.windll.gdi32
        self.w, self.h = w, h
        self._sdc = u.GetDC(0)
        self._mdc = g.CreateCompatibleDC(self._sdc)
        self._bmp = g.CreateCompatibleBitmap(self._sdc, w, h)
        if not (self._sdc and self._mdc and self._bmp):
            raise OSError("screen DC setup failed")
        self._old = g.SelectObject(self._mdc, self._bmp)
        self._src_x, self._src_y = x, y
        bmi = _BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = w
        bmi.bmiHeader.biHeight = -h          # negative = top-down rows
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        bmi.bmiHeader.biCompression = 0      # BI_RGB
        self._bmi = bmi
        self._buf = np.empty((h, w, 4), np.uint8)
        self._g = g

    def grab(self):
        g = self._g
        g.BitBlt(self._mdc, 0, 0, self.w, self.h, self._sdc,
                 self._src_x, self._src_y, 0x40CC0020)   # SRCCOPY|CAPTUREBLT
        g.GetDIBits(self._mdc, self._bmp, 0, self.h,
                    ctypes.c_void_p(self._buf.ctypes.data),
                    ctypes.byref(self._bmi), 0)
        return self._buf[:, :, :3].copy()    # BGRA -> BGR


class DxGrabber:
    """dxcam DXGI Desktop Duplication — GPU-side capture at monitor refresh.
    start() runs the dup loop on its own thread; get_latest_frame() just
    copies the newest BGR frame."""

    def __init__(self, out_idx: int, w: int, h: int) -> None:
        if dxcam is None:
            raise RuntimeError("dxcam not installed")
        self._cam = dxcam.create(output_idx=out_idx, output_color="BGR")
        if self._cam is None:
            raise RuntimeError("dxcam.create failed")
        # region is relative to the output's own origin — full output here.
        self._cam.start(region=(0, 0, w, h), target_fps=60, video_mode=True)
        self._w, self._h = w, h

    def grab(self):
        f = self._cam.get_latest_frame()
        if f is None:                              # before first dup frame
            return np.zeros((self._h, self._w, 3), np.uint8)
        return f


class Grabber:
    """dxcam (DXGI, 60fps) > mss > raw BitBlt (WinGrabber) > PIL ImageGrab."""

    def __init__(self, bbox, out_idx=None) -> None:
        self.bbox = bbox
        x, y = bbox[0], bbox[1]
        w, h = bbox[2] - x, bbox[3] - y
        self._dx = self._mss = self._win = None
        if out_idx is not None and dxcam is not None:
            try:
                self._dx = DxGrabber(out_idx, w, h)
            except Exception as e:
                print(f"dxcam: {e} — falling back")
        if self._dx is None and mss is not None:
            self._mss = mss.mss()
        self._mon = {"left": x, "top": y, "width": w, "height": h}
        if self._dx is None and self._mss is None:
            try:
                self._win = WinGrabber(x, y, w, h)
            except Exception:
                self._win = None

    def grab(self):
        if self._dx is not None:
            return self._dx.grab()
        if self._mss is not None:
            return cv2.cvtColor(np.asarray(self._mss.grab(self._mon)),
                                cv2.COLOR_BGRA2BGR)
        if self._win is not None:
            return self._win.grab()
        shot = ImageGrab.grab(bbox=self.bbox, all_screens=True)
        return cv2.cvtColor(np.asarray(shot), cv2.COLOR_RGB2BGR)


# ------------------------------------------------------------------ main -----
def main() -> None:
    ap = argparse.ArgumentParser(description="Screen Vision — faithful desktop "
                                     "port of the RetroIntercept Terminator pipeline")
    ap.add_argument("--style", choices=list(STYLES), default="hud",
                    help="hud = transparent overlay; others = styled mirror")
    ap.add_argument("--monitor", type=int, default=1, help="1-based monitor index")
    ap.add_argument("--all", action="store_true", help="union of all monitors")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--stride", type=int, default=None,
                    help="run detection every Nth grab (DetStride)")
    ap.add_argument("--cls-conf", type=float, default=None,
                    help="stage-2 classifier keep threshold (DetClsConf)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--model2", default=None)
    ap.add_argument("--model3", default=None, help="stage-2 classifier onnx")
    ap.add_argument("--classes", default=None)
    ap.add_argument("--weapons", default=None)
    ap.add_argument("--classes2", default=None)
    ap.add_argument("--weapons2", default=None)
    ap.add_argument("--no-hud", action="store_true",
                    help="styled window without HUD")
    ap.add_argument("--gpu", action="store_true",
                    help="no-op: DirectML GPU is automatic when installed")
    args = ap.parse_args()

    cfg = Config(args)
    if args.no_hud:
        cfg.hud = False
    det = Detector(cfg)
    if not det.live:
        sys.exit("no detector models found — check config/retro.json / --model")

    stylizer = None
    if cfg.term_base_onnx and cfg.ai_model:
        try:
            stylizer = AiStylizer(cfg)
            print(f"ai: {cfg.ai_model} [{stylizer.w}x{stylizer.h}] "
                  f"temporal={cfg.ai_temporal} blend={cfg.ai_blend}")
        except Exception as e:
            print(f"ai: {e} — term style falls back to HLSL Sobel")

    app = QApplication(sys.argv)
    geo = _geo(args, app)
    bbox = (geo[0], geo[1], geo[0] + geo[2], geo[1] + geo[3])
    # dxcam only drives a single DXGI output — use it for --monitor, not --all.
    grabber = Grabber(bbox, out_idx=None if args.all else args.monitor - 1)

    lock = threading.Lock()
    stop = threading.Event()
    shared: dict = {"tracks": [], "styled": None,
                    "frames": 0, "style": args.style, "palette": cfg.palette,
                    "reload": False}
    trk = Tracker()
    last_dets: list[dict] = []
    last = {"t": time.perf_counter()}
    fps_ema = {"grab": 0.0, "infer": 0.0, "style": 0.0}
    state = {"det": det, "cfg": cfg, "stylizer": stylizer}

    def run_style(style: str, frame, c: Config):
        if style == "term":
            return style_term(state["stylizer"])(frame, c)
        if style == "retro":
            return style_retro(frame, c)
        if style == "ascii":
            return style_ascii(frame, c)
        return None

    # Inference runs on its own thread — grabbing/tracker/HUD stay smooth
    # while the (slow) CPU sessions run at whatever rate they can manage.
    latest = {"frame": None}

    def infer_worker() -> None:
        while not stop.is_set():
            with lock:
                frame = latest["frame"]
                latest["frame"] = None
                d, c = state["det"], state["cfg"]
            if frame is None:
                stop.wait(0.003)
                continue
            ti = time.perf_counter()
            try:
                dets = d.infer(frame)
            except Exception:
                continue
            fps_ema["infer"] = (fps_ema["infer"] * 0.9
                                + (time.perf_counter() - ti) * 100.0)
            with lock:
                last_dets[:] = dets

    def worker() -> None:
        n = 0
        while not stop.is_set():
            with lock:
                do_reload = shared.pop("reload", False)
            if do_reload:
                c2 = Config(args)
                if args.no_hud:
                    c2.hud = False
                try:
                    d2 = Detector(c2)
                    if d2.live:
                        state["det"], state["cfg"] = d2, c2
                        try:
                            state["stylizer"] = (AiStylizer(c2)
                                                 if c2.term_base_onnx
                                                 and c2.ai_model else None)
                        except Exception:
                            state["stylizer"] = None
                        cfg.__dict__.update(c2.__dict__)   # painter sees it
                        print("config reloaded")
                except Exception as e:
                    print(f"reload failed: {e}")

            t0 = time.perf_counter()
            frame = grabber.grab()
            n += 1
            c = state["cfg"]
            with lock:
                if (n - 1) % c.stride == 0:
                    latest["frame"] = frame
                cur_dets = list(last_dets)
            now = time.perf_counter()
            trk.update(cur_dets, now - last["t"], c.max_targets)
            last["t"] = now
            view = [t for t in trk.tracks
                    if t["hits"] >= (4 if c.is_weapon(t["cls"]) else 2)]

            with lock:
                style = shared["style"]
                c.palette = shared["palette"]
            img = None
            if style != "hud":
                ti = time.perf_counter()
                styled = run_style(style, frame, c)
                if styled is not None:
                    img = QImage(styled.data, styled.shape[1], styled.shape[0],
                                 styled.strides[0],
                                 QImage.Format.Format_BGR888).copy()
                fps_ema["style"] = (fps_ema["style"] * 0.9
                                    + (time.perf_counter() - ti) * 100.0)
            fps_ema["grab"] = (fps_ema["grab"] * 0.9
                               + (time.perf_counter() - t0) * 100.0)
            with lock:
                shared["tracks"] = view
                shared["styled"] = img
                shared["frames"] += 1
            # Pace to ~60 Hz — leaves CPU for the inference thread.
            stop.wait(max(0.0, 0.016 - (time.perf_counter() - t0)))

    it = threading.Thread(target=infer_worker, daemon=True)
    it.start()
    t = threading.Thread(target=worker, daemon=True)
    t.start()

    hud = HudPainter()
    win = Overlay(geo, hud, cfg, shared, lock, stop)
    win.show()
    _exclude_from_capture(win)
    print(f"providers: {PROVIDERS}")
    print(f"{args.style} on {geo[2]}x{geo[3]} @({geo[0]},{geo[1]})"
          f" — F8 palette, F9 style, F10 panic, F11 reload, End quits")
    t0 = time.perf_counter()
    try:
        app.exec()
    finally:
        stop.set()
        t.join(timeout=1.5)
        it.join(timeout=1.5)
    dt = time.perf_counter() - t0
    print(f"{shared['frames']} grabs in {dt:.0f}s "
          f"({shared['frames'] / max(dt, 1e-3):.1f} grab/s)")


if __name__ == "__main__":
    main()
