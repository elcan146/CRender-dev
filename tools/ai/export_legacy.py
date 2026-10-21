"""export_legacy.py — rescue-export .pt checkpoints ultralytics can't load.

Two failure classes, both shimmed here:
  * YOLOv5-era pickles: reference a `models` module (the yolov5 repo layout).
    Fixed by putting the installed `yolov5` pip package dir on sys.path so
    `import models.yolo` resolves to its vendored code, plus a pkg_resources
    stub (setuptools>=81 removed it; yolov5 only needs parse_version).
  * Old-ultralytics pickles: reference `ultralytics.yolo.*` (8.0.x namespace,
    moved to ultralytics.models.yolo.* in 8.1+). Fixed with sys.modules aliases.

Exports RAW detect output (no NMS) to onnx, batch 1, fixed square input —
the same contract fetch_detector.py produces, so detect()/Detect.cpp decode
it identically. Class names are injected into onnx metadata ('names') so the
cascade UI labels detections correctly.

  python tools/ai/export_legacy.py --model config/test-5.pt -o config/test-5.onnx [--size 640]
"""
import argparse
import sys
import types
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]


def shim_pkg_resources() -> None:
    try:
        import pkg_resources  # noqa: F401
        return
    except ImportError:
        pass
    from packaging.version import parse as _pv
    stub = types.ModuleType("pkg_resources")
    stub.parse_version = lambda s: _pv(str(s))
    stub.get_distribution = lambda n: types.SimpleNamespace(version="0.0")
    stub.resource_filename = lambda *a, **k: ""
    sys.modules["pkg_resources"] = stub


def shim_ultralytics_yolo() -> None:
    """Alias the 8.0.x ultralytics.yolo.* namespace to the modern tree."""
    import importlib
    import ultralytics
    if "ultralytics.yolo" in sys.modules:
        return
    pkg = types.ModuleType("ultralytics.yolo")
    pkg.__path__ = []
    sys.modules["ultralytics.yolo"] = pkg
    setattr(ultralytics, "yolo", pkg)
    for old, new in [
        ("ultralytics.yolo.v8", "ultralytics.models.yolo"),
        ("ultralytics.yolo.v8.detect", "ultralytics.models.yolo.detect"),
        ("ultralytics.yolo.engine", "ultralytics.engine"),
        ("ultralytics.yolo.data", "ultralytics.data"),
        ("ultralytics.yolo.utils", "ultralytics.utils"),
        ("ultralytics.yolo.cfg", "ultralytics.cfg"),
        ("ultralytics.yolo.utils.metrics", "ultralytics.utils.metrics"),
        ("ultralytics.yolo.utils.torch_utils", "ultralytics.utils.torch_utils"),
        ("ultralytics.yolo.utils.ops", "ultralytics.utils.ops"),
        ("ultralytics.yolo.utils.checks", "ultralytics.utils.checks"),
        ("ultralytics.yolo.utils.plotting", "ultralytics.utils.plotting"),
    ]:
        try:
            sys.modules[old] = importlib.import_module(new)
            setattr(pkg, old.split(".")[-1], sys.modules[old])
        except Exception:
            pass                                   # alias what exists only


def shim_yolov5_models() -> None:
    """Put the yolov5 pip package dir on sys.path so `import models.*`
    (as pickled) resolves to its vendored source."""
    import site
    for base in site.getsitepackages() + [site.getusersitepackages()]:
        d = Path(base) / "yolov5"
        if d.is_dir() and (d / "models").is_dir():
            if str(d) not in sys.path:
                sys.path.insert(0, str(d))
            return
    raise SystemExit("yolov5 package not found — pip install yolov5")


def _v7_stubs():
    """Stub classes for attrs old pickles reference but modern yolov5 lacks.
    Pickle restores __dict__/_modules without __init__, so only forward must
    be faithful."""
    import torch
    import torch.nn as nn

    class MP(nn.Module):                      # early-yolov5 stride-2 maxpool
        def forward(self, x):
            return self.m(x)

    class SPPCSPC(nn.Module):                 # yolov7 spatial-pyramid CSP
        def forward(self, x):
            x1 = self.cv4(self.cv3(self.cv1(x)))
            y1 = self.cv6(self.cv5(torch.cat(
                [x1] + [m(x1) for m in self.m], 1)))
            y2 = self.cv2(x)
            return self.cv7(torch.cat((y1, y2), 1))

    class Concat(nn.Module):
        def forward(self, x):
            return torch.cat(x, self.d)

    class RepConv(nn.Module):                 # yolov7 reparam conv
        def forward(self, x):
            if getattr(self, "rbr_reparam", None) is not None:
                return self.act(self.rbr_reparam(x))
            out = self.rbr_dense(x) + self.rbr_1x1(x)
            ident = getattr(self, "rbr_identity", None)
            if ident is not None:             # yolov7 stores None when skipped
                out = out + ident(x)
            return self.act(out)

    class ImplicitA(nn.Module):               # yolov7 implicit add
        def forward(self, x):
            return self.implicit + x

    class ImplicitM(nn.Module):               # yolov7 implicit mul
        def forward(self, x):
            return self.implicit * x

    class Detect(nn.Module):                  # yolov5 detect head
        def _make_grid(self, nx=20, ny=20):
            yv, xv = torch.meshgrid(torch.arange(ny), torch.arange(nx),
                                    indexing="ij")
            return torch.stack((xv, yv), 2).view(1, 1, ny, nx, 2).float()

        def forward(self, x):                 # eval path only
            z = []
            for i in range(self.nl):
                xi = self.m[i](x[i])
                bs, _, ny, nx = xi.shape
                xi = xi.view(bs, self.na, self.no, ny, nx)
                xi = xi.permute(0, 1, 3, 4, 2).contiguous()
                a = self.anchors[i]
                if a.dim() < 2 or a.shape[-1] != 2:        # weird pickle
                    a = self.anchor_grid[i].reshape(-1, 2) / \
                        self.stride[i]
                ag = (a.float() * self.stride[i]) \
                    .view(1, self.na, 1, 1, 2)
                y = xi.sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2 - 0.5
                               + self._make_grid(nx, ny).to(xi.device)) \
                    * self.stride[i]
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * ag
                z.append(y.view(bs, -1, self.no))
            return torch.cat(z, 1)

    class IDetect(nn.Module):                 # yolov7 implicit-head detect
        def _make_grid(self, nx=20, ny=20):
            yv, xv = torch.meshgrid(torch.arange(ny), torch.arange(nx),
                                    indexing="ij")
            return torch.stack((xv, yv), 2).view(1, 1, ny, nx, 2).float()

        def forward(self, x):                 # eval path only
            z = []
            for i in range(self.nl):
                # yolov7: ia on input (C_in ch), conv, im on output (na*no ch)
                xi = self.im[i](self.m[i](self.ia[i](x[i])))
                bs, _, ny, nx = xi.shape
                xi = xi.view(bs, self.na, self.no, ny, nx)
                xi = xi.permute(0, 1, 3, 4, 2).contiguous()
                if self.grid[i].shape[2:4] != xi.shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(xi.device)
                y = xi.sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2 - 0.5
                               + self.grid[i]) * self.stride[i]
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 \
                    * self.anchor_grid[i]
                z.append(y.view(bs, -1, self.no))
            return torch.cat(z, 1)

    return {
        "models.common": {"MP": MP, "SPPCSPC": SPPCSPC, "Concat": Concat,
                          "RepConv": RepConv, "ImplicitA": ImplicitA,
                          "ImplicitM": ImplicitM},
        "models.yolo": {"IDetect": IDetect, "Detect": Detect,
                        "ImplicitA": ImplicitA,
                        "ImplicitM": ImplicitM, "SPPCSPC": SPPCSPC,
                        "MP": MP, "Concat": Concat},
    }


def _inject_stub_attrs() -> None:
    """Bulk-install every known stub into the real models.* modules BEFORE
    torch.load — lazy patching can set attrs on a different module object
    than the unpickler imports, looping forever on the same name."""
    import importlib
    for mod_name, attrs in _v7_stubs().items():
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        for name, stub in attrs.items():
            if getattr(mod, name, None) is not stub:
                setattr(mod, name, stub)
                print(f"patched {mod_name}.{name}")


def _patch_v5_attrs(err: BaseException) -> bool:
    """Parse 'Can't get attribute X on module M' and inject the known stub.
    Returns True when something was patched."""
    import importlib
    import re
    mm = re.search(r"Can't get attribute '(\w+)' on <module '([\w.]+)'",
                   str(err))
    if not mm:
        return False
    attr, mod_name = mm.group(1), mm.group(2)
    mod = importlib.import_module(mod_name)
    stub = _v7_stubs().get(mod_name, {}).get(attr)
    if stub is None:
        print(f"no stub for {mod_name}.{attr}")
        return False
    setattr(mod, attr, stub)
    print(f"patched {mod_name}.{attr}")
    return True


def load_model(path: Path):
    """torch.load with all shims; returns (model eval'd float, names dict)."""
    import pathlib
    import torch
    # checkpoints pickled on posix systems can't instantiate PosixPath on
    # Windows — alias it to the pure (instantiable) variant
    pathlib.PosixPath = pathlib.PurePosixPath
    try:
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
    except ModuleNotFoundError as e:
        if "ultralytics.yolo" in str(e):
            shim_ultralytics_yolo()
            ck = torch.load(str(path), map_location="cpu", weights_only=False)
        elif "models" in str(e):
            shim_yolov5_models()
            _inject_stub_attrs()             # all known stubs, up front
            seen = set()
            for _ in range(20):                # chain-patch leftovers
                try:
                    ck = torch.load(str(path), map_location="cpu",
                                    weights_only=False)
                    break
                except AttributeError as e2:
                    key = str(e2)[:200]
                    if key in seen:            # same attr again = stuck
                        raise
                    seen.add(key)
                    if not _patch_v5_attrs(e2):
                        raise
            else:
                raise RuntimeError("too many missing v5/v7 attrs")
        else:
            raise
    m = ck.get("model") or ck.get("ema")
    if m is None:
        raise SystemExit(f"{path.name}: no 'model' in checkpoint")
    m = m.eval().float()
    # Old v5 checkpoints resolve Detect from the INSTALLED yolov5 package,
    # whose forward/_make_grid version-mismatches the pickled buffers
    # (test-9 crash). Reclass every detect head to our canonical stub —
    # the stub table only fires on missing attrs, so do it unconditionally.
    Det = _v7_stubs()["models.yolo"]["Detect"]
    for mod in m.modules():
        if mod.__class__.__name__ == "Detect" and hasattr(mod, "m"):
            mod.__class__ = Det
    names = ck.get("names") or getattr(m, "names", None) or {}
    if isinstance(names, (list, tuple)):
        names = {i: str(v) for i, v in enumerate(names)}
    else:
        names = {int(k): str(v) for k, v in names.items()}
    return m, names


class _Wrap:    # torch.nn.Module forward -> first output tensor
    def __init__(self, m):
        self.m = m

    def __call__(self, x):
        return self.m(x)

    def __getattr__(self, k):
        return getattr(self.m, k)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--size", type=int, default=640)
    args = ap.parse_args()
    shim_pkg_resources()

    import torch
    mp, out = Path(args.model), Path(args.output)
    m, names = load_model(mp)
    print(f"{mp.name}: nc={len(names)} names={names}")

    class F(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m = m

        def forward(self, x):
            y = self.m(x)
            return y[0] if isinstance(y, (list, tuple)) else y

    f = F().eval()
    x = torch.zeros(1, 3, args.size, args.size)
    with torch.no_grad():
        y = f(x)
    print("out shape:", tuple(y.shape))
    torch.onnx.export(
        f, x, str(out), opset_version=17, do_constant_folding=True,
        input_names=["images"], output_names=["output0"],
        dynamic_axes=None)
    # inject class names so the UI can label detections
    try:
        import onnx
        om = onnx.load(str(out))
        p = om.metadata_props.add()
        p.key, p.value = "names", str(names)
        onnx.save(om, str(out))
    except Exception as e:
        print("names metadata skipped:", e)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

