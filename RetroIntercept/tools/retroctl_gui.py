"""RetroIntercept controller - pick ANY game folder, Apply, Rewind to undo.

    pip install PyQt6
    python tools/retroctl_gui.py

- Finds the real game .exe anywhere under the picked folder (e.g. Yakuza 0\\media,
  UE <Game>\\Binaries\\Win64) and installs next to it with the matching x86/x64 DLL.
- Writes a per-game retro.json from the settings below (Pixel scale = RenderScale:
  the game renders at 1/N resolution -> big FPS gain when GPU-bound).
- Every file we overwrite is backed up as <name>.retro-bak; Rewind restores it.
"""
import json
import os
import re
import struct
import sys
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton, QSpinBox,
    QVBoxLayout, QWidget,
)

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ".retroctl.json"
BAK_SUFFIX = ".retro-bak"
TEMPLATE = ROOT / "config" / "retro.json"

DLL_BY_ARCH = {
    "x86": ROOT / "build" / "win32" / "Release" / "d3d11.dll",
    "x64": ROOT / "build" / "x64" / "Release" / "d3d11.dll",
}
ORT_BIN = {
    "x86": ROOT / "third_party" / "onnxruntime" / "bin",
    "x64": ROOT / "third_party" / "onnxruntime" / "bin64",
}
NEURAL_MODELS = [
    (ROOT / "config" / "retro_ai.onnx", "retro_ai.onnx"),
    (ROOT / "config" / "retro_ai_temporal.onnx", "retro_ai_temporal.onnx"),
    (ROOT / "config" / "terminator_vision.onnx", "terminator_vision.onnx"),
    (ROOT / "config" / "yolov8n.onnx", "yolov8n.onnx"),
    (ROOT / "config" / "det_weapons.onnx", "det_weapons.onnx"),
    (ROOT / "config" / "best.onnx", "best.onnx"),  # BecayeSoft gun detector (1 class)
]

def neural_files(arch: str) -> list[tuple[Path, str]]:
    """ORT/DirectML runtime for the arch (if fetched) + arch-neutral models."""
    bindir = ORT_BIN[arch]
    rt = [(bindir / n, n) for n in ("onnxruntime.dll", "onnxruntime_providers_shared.dll",
                                    "DirectML.dll") if (bindir / n).exists()]
    return rt + [(s, n) for s, n in NEURAL_MODELS if s.exists()]


MODES = ["Off", "RetroFX", "Ascii", "Terminator"]   # config alias: "Neural" -> Terminator

SCAN_DEPTH = 5
SKIP_DIRS = {"_redist", "redist", "commonredist", "_commonredist", "directx", "__installer",
             "installer", "installers", "support", "engine", "prereqs", "vcredist", "dotnet"}
SKIP_EXE = re.compile(
    r"(unins|setup|install|redist|vcredist|dxsetup|dxwebsetup|crash|report|quicksfv|"
    r"prereq|dotnet|directx|easyanticheat|battleye|beservice|helper|cefprocess|"
    r"errorreporter|updater|patcher|benchmarktool|uploader)", re.IGNORECASE)


def exe_arch(exe: Path) -> str | None:
    """PE machine field -> 'x86' / 'x64' / None (reads only the header)."""
    try:
        with open(exe, "rb") as f:
            head = f.read(4096)
            if head[:2] != b"MZ":
                return None
            pe = struct.unpack_from("<I", head, 0x3C)[0]
            f.seek(pe)
            sig = f.read(6)
            if sig[:4] != b"PE\0\0":
                return None
            machine = struct.unpack_from("<H", sig, 4)[0]
        return {0x014C: "x86", 0x8664: "x64"}.get(machine)
    except Exception:
        return None


def mentions_d3d11(path: Path) -> bool:
    """Chunked case-insensitive scan for 'd3d11' in a binary."""
    pat = re.compile(rb"d3d11", re.IGNORECASE)
    tail = b""
    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1 << 22)
                if not chunk:
                    return False
                if pat.search(tail + chunk):
                    return True
                tail = chunk[-8:]
    except Exception:
        return False


def find_game_exe(folder: Path) -> Path | None:
    """Largest plausible game .exe under folder (depth-limited, junk filtered)."""
    best: tuple[int, Path] | None = None
    base_depth = len(folder.parts)
    for dirpath, dirnames, filenames in os.walk(folder):
        depth = len(Path(dirpath).parts) - base_depth
        dirnames[:] = [d for d in dirnames
                       if d.lower() not in SKIP_DIRS and depth < SCAN_DEPTH]
        for fn in filenames:
            if not fn.lower().endswith(".exe") or SKIP_EXE.search(fn):
                continue
            p = Path(dirpath) / fn
            try:
                size = p.stat().st_size
            except OSError:
                continue
            if exe_arch(p) is None:
                continue
            if best is None or size > best[0]:
                best = (size, p)
    return best[1] if best else None


def find_manifest(folder: Path) -> Path | None:
    if (folder / MANIFEST).exists():
        return folder / MANIFEST
    base_depth = len(folder.parts)
    for dirpath, dirnames, filenames in os.walk(folder):
        depth = len(Path(dirpath).parts) - base_depth
        dirnames[:] = [d for d in dirnames if depth < SCAN_DEPTH]
        if MANIFEST in filenames:
            return Path(dirpath) / MANIFEST
    return None


class Window(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("RetroIntercept")
        self.setMinimumWidth(520)
        self.folder: Path | None = None
        self.exe: Path | None = None
        self.target: Path | None = None   # dir the game exe lives in = install dir

        self.folder_label = QLabel("No game folder selected")
        self.folder_label.setWordWrap(True)
        browse = QPushButton("Select game folder…")
        browse.clicked.connect(self.pick_folder)
        pick_exe = QPushButton("Pick .exe manually…")
        pick_exe.clicked.connect(self.pick_exe)
        top = QHBoxLayout()
        top.addWidget(browse)
        top.addWidget(pick_exe)

        self.mode = QComboBox()
        self.mode.addItems(MODES)
        self.mode.setCurrentText("RetroFX")
        self.scale = QSpinBox()
        self.scale.setRange(1, 8)
        self.scale.setValue(4)
        self.scale.setSuffix("x  (1 = native res)")
        self.scale.valueChanged.connect(self.sync_pixel)
        self.pixel = QSpinBox()
        self.pixel.setRange(1, 32)
        self.pixel.setValue(4)
        self.pixel.setSuffix(" px")
        self.skip_shadows = QCheckBox("Skip shadow-map passes (extra FPS, retro modes only)")
        self.skip_shadows.setChecked(True)
        self.patch_cb = QCheckBox("Compat: patch screen-size constants (try if image is cropped/zoomed)")
        self.neural = QCheckBox("Include Terminator AI files (ONNX Runtime + DirectML)")
        self.term_fx = QCheckBox("Terminator HUD + lens (brackets, reticle, scanlines)")
        self.term_fx.setChecked(True)

        form = QFormLayout()
        form.addRow("Start mode (F9 cycles):", self.mode)
        form.addRow("Pixel scale / RenderScale:", self.scale)
        form.addRow("RetroFX block size:", self.pixel)
        form.addRow(self.skip_shadows)
        form.addRow(self.patch_cb)
        form.addRow(self.neural)
        form.addRow(self.term_fx)
        box = QGroupBox("Settings (written to retro.json)")
        box.setLayout(form)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self.apply)
        self.rewind_btn = QPushButton("⟲ Rewind")
        self.rewind_btn.clicked.connect(self.rewind)
        btns = QHBoxLayout()
        btns.addWidget(self.apply_btn)
        btns.addWidget(self.rewind_btn)

        lay = QVBoxLayout()
        lay.addWidget(self.folder_label)
        lay.addLayout(top)
        lay.addWidget(box)
        lay.addLayout(btns)
        lay.addWidget(self.status)
        w = QWidget()
        w.setLayout(lay)
        self.setCentralWidget(w)
        self.refresh()

    # ---- selection ---------------------------------------------------------
    def sync_pixel(self, v: int) -> None:
        if self.pixel.value() < v:
            self.pixel.setValue(v)

    def pick_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select the game's install folder")
        if not d:
            return
        self.folder = Path(d)
        self.status.setText("Scanning…")
        QApplication.processEvents()
        man = find_manifest(self.folder)
        if man is not None:
            self.target = man.parent
            self.exe = find_game_exe(self.target)
        else:
            self.exe = find_game_exe(self.folder)
            self.target = self.exe.parent if self.exe else self.folder
        self.load_installed_settings()
        self.refresh()

    def pick_exe(self) -> None:
        start = str(self.folder) if self.folder else ""
        f, _ = QFileDialog.getOpenFileName(self, "Select the game .exe", start, "Executables (*.exe)")
        if not f:
            return
        self.exe = Path(f)
        self.folder = self.folder or self.exe.parent
        self.target = self.exe.parent
        self.load_installed_settings()
        self.refresh()

    # ---- manifest ----------------------------------------------------------
    def manifest_path(self) -> Path:
        return self.target / MANIFEST

    def installed_entries(self) -> list[dict]:
        try:
            files = json.loads(self.manifest_path().read_text())["files"]
        except Exception:
            return []
        return [{"name": f, "backup": None} if isinstance(f, str) else f for f in files]

    def is_installed(self) -> bool:
        return self.target is not None and self.manifest_path().exists()

    def load_installed_settings(self) -> None:
        if not self.is_installed():
            return
        try:
            cfg = json.loads((self.target / "retro.json").read_text())
        except Exception:
            return
        mode = cfg.get("Mode")
        if mode == "Neural":
            mode = "Terminator"
        if mode in MODES:
            self.mode.setCurrentText(mode)
        self.scale.setValue(int(cfg.get("RenderScale", 1)))
        self.pixel.setValue(int(cfg.get("PixelPx", self.scale.value())))
        self.skip_shadows.setChecked(bool(cfg.get("SkipShadows", 0)))
        self.patch_cb.setChecked(bool(cfg.get("PatchScreenCB", 0)))
        self.term_fx.setChecked(bool(cfg.get("TermHud", 1)) and bool(cfg.get("TermLens", 1)))

    def refresh(self) -> None:
        have = self.target is not None
        installed = self.is_installed()
        self.apply_btn.setEnabled(have)
        self.apply_btn.setText("Update settings" if installed else "Apply")
        self.rewind_btn.setEnabled(installed)
        if not have:
            self.status.setText("")
            return
        arch = exe_arch(self.exe) if self.exe else None
        exe_txt = f"{self.exe.name} ({arch or '?'})" if self.exe else "no game .exe found"
        self.folder_label.setText(f"Game: <b>{exe_txt}</b><br>Install dir: {self.target}")
        if installed:
            parts = [e["name"] + (" (original backed up)" if e.get("backup") else "")
                     for e in self.installed_entries()]
            self.status.setText(f"<b style='color:#4caf50'>Active</b> — installed: {', '.join(parts)}")
        else:
            self.status.setText("<b style='color:#888'>Not installed</b> — press Apply to enable.")

    # ---- apply / rewind ----------------------------------------------------
    def build_config(self) -> bytes:
        cfg = json.loads(TEMPLATE.read_text())
        cfg["Mode"] = self.mode.currentText()
        cfg["RenderScale"] = self.scale.value()
        cfg["PixelPx"] = self.pixel.value()
        cfg["SkipShadows"] = 1 if self.skip_shadows.isChecked() else 0
        cfg["PatchScreenCB"] = 1 if self.patch_cb.isChecked() else 0
        cfg["TermHud"] = 1 if self.term_fx.isChecked() else 0
        cfg["TermLens"] = 1 if self.term_fx.isChecked() else 0
        if cfg["Mode"] == "Terminator":
            cfg["AiModel"] = "terminator_vision.onnx"
            # Weapon session: best.onnx (1-class 'gun') scanned on person crops.
            # In-game scores sit at 0.15-0.45 raw, so keep the low conf floor —
            # the DLL's crop-fill/flicker/dedup gates do the FP filtering.
            # Inert when the ONNX files aren't deployed.
            cfg["DetModel2"] = "best.onnx"
            cfg["DetClasses2"] = "0"
            cfg["DetWeapon2"] = "0"
            cfg["DetConf"] = 0.15
            cfg["DetSight"] = False   # no weapon->owner sight lines
            cfg["DetDump"] = 0
        return (json.dumps(cfg, indent=2) + "\n").encode()

    def apply(self) -> None:
        arch = exe_arch(self.exe) if self.exe else None
        if arch is None:
            r = QMessageBox.question(
                self, "No game .exe found",
                f"No 32/64-bit game .exe detected in\n{self.target}\n\n"
                "Use 'Pick .exe manually…' for best results. Continue with the x64 build?")
            if r != QMessageBox.StandardButton.Yes:
                return
            arch = "x64"
        dll_src = DLL_BY_ARCH[arch]
        if not dll_src.exists():
            preset = "x64-release" if arch == "x64" else "win32-release"
            QMessageBox.critical(self, "Not built",
                                 f"{dll_src} not found. Build first:\ncmake --build --preset {preset}")
            return
        if self.exe and not self.is_installed() and not mentions_d3d11(self.exe):
            dlls = [p for p in self.target.glob("*.dll") if p.stat().st_size < (256 << 20)]
            if not any(mentions_d3d11(p) for p in dlls):
                r = QMessageBox.question(
                    self, "Direct3D 11 not detected",
                    "This game doesn't reference d3d11 — it may use D3D9/D3D12/Vulkan/OpenGL, "
                    "which RetroIntercept can't hook. Install anyway?")
                if r != QMessageBox.StandardButton.Yes:
                    return

        ai_files = neural_files(arch) if self.neural.isChecked() else []
        if (self.neural.isChecked() and self.mode.currentText() == "Terminator"
                and not any(n == "onnxruntime.dll" for _, n in ai_files)):
            QMessageBox.information(
                self, "AI not available",
                f"No {arch} ONNX Runtime in third_party/onnxruntime — run "
                "tools/ai/fetch_ort.ps1 first. Deploying without AI files "
                "(the HLSL fallback still works).")
        payload: list[tuple[str, bytes]] = [
            ("d3d11.dll", dll_src.read_bytes()),
            ("retro.json", self.build_config()),
        ]
        payload += [(n, s.read_bytes()) for s, n in ai_files]

        prev = {e["name"]: e for e in self.installed_entries()}
        installed = []
        try:
            for name, data in payload:
                dst = self.target / name
                bak = self.target / (name + BAK_SUFFIX)
                backup = prev[name].get("backup") if name in prev else None
                if name not in prev and dst.exists():
                    if bak.exists():
                        backup = bak.name          # keep the first/original backup
                        dst.unlink()
                    elif dst.read_bytes() != data:
                        dst.rename(bak)            # foreign file: preserve it for Rewind
                        backup = bak.name
                dst.write_bytes(data)
                installed.append({"name": name, "backup": backup})
        except PermissionError:
            QMessageBox.critical(self, "File in use",
                                 "A file is locked — close the game first, then Apply again.")
            return
        finally:
            done = {e["name"] for e in installed}
            installed += [e for n, e in prev.items() if n not in done]
            if installed:
                self.manifest_path().write_text(json.dumps({"files": installed}, indent=1))

        backed = [e["name"] for e in installed if e["backup"]]
        msg = (f"Installed ({arch}) into {self.target}: {', '.join(e['name'] for e in installed)}"
               f"<br>RenderScale={self.scale.value()} — scene renders at 1/{self.scale.value()} res. "
               "F9 cycles modes in-game. RenderScale changes need a game restart.")
        if backed:
            msg += f"<br>Backed up: {', '.join(backed)} → *.retro-bak (restored on Rewind)"
        self.refresh()
        self.status.setText(msg)

    def rewind(self) -> None:
        restored = []
        remaining = []
        for e in self.installed_entries():
            dst = self.target / e["name"]
            try:
                dst.unlink(missing_ok=True)
                if e.get("backup"):
                    bak = self.target / e["backup"]
                    if bak.exists():
                        bak.rename(dst)
                        restored.append(e["name"])
            except PermissionError:
                remaining.append(e)
        if remaining:
            self.manifest_path().write_text(json.dumps({"files": remaining}, indent=1))
            QMessageBox.critical(
                self, "File in use",
                f"{', '.join(e['name'] for e in remaining)} locked — close the game, then Rewind again.")
            self.refresh()
            return
        self.manifest_path().unlink(missing_ok=True)
        (self.target / "RetroIntercept.log").unlink(missing_ok=True)
        msg = "Rewound — the game runs stock again."
        if restored:
            msg += f" Restored originals: {', '.join(restored)}"
        self.refresh()
        self.status.setText(msg)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec())
