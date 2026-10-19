# Docker

Two images, two different jobs. Docker is **not** installed on the dev machine —
these are provided ready-to-build; install Docker Desktop (Windows containers
enabled) to use the build image.

## What can be dockerized

| Image | Dockerfile | Contents |
|---|---|---|
| `crender-tools` | `tools.Dockerfile` | Linux + Python 3.11: ONNX model generation/verification, detector export/tests (`tools/ai/*`). CPU-only. |
| `crender-build-win` | `windows-build.Dockerfile` | Windows Server Core + VS Build Tools: compiles `d3d11.dll` + `test_cube.exe` for **x86 and x64**. Artifacts are extracted with `docker cp`. |

## What cannot

- **The `d3d11.dll` proxy at runtime** — it must be `LoadLibrary`'d by a real
  D3D11 game (e.g. `Heaven.exe`) on a Windows desktop with a GPU. Containers
  have no display, no game, no swap chain to intercept.
- **`test_cube.exe`** — windowed D3D11 app; builds in the container, must run on the host.
- **`tools/screen_vision.py`** — desktop capture (`mss`/`ImageGrab`) + transparent overlay windows.
- **`tools/CRender_wizard.py`** — PyQt6 GUI that copies files into game directories.

So "everything" is not containerizable: **build + AI tooling yes, run/test no**.

## AI tooling image (Linux)

```powershell
docker build -f docker/tools.Dockerfile -t crender-tools .

# Generate the placeholder + terminator models into ./config (mounted as /out):
docker run --rm -v ${PWD}/config:/out crender-tools tools/ai/make_placeholder.py -o /out/retro_ai.onnx
docker run --rm -v ${PWD}/config:/out crender-tools tools/ai/make_placeholder.py --two-input -o /out/retro_ai_temporal.onnx
docker run --rm -v ${PWD}/config:/out crender-tools tools/ai/make_terminator_onnx.py -o /out/terminator_vision.onnx
docker run --rm -v ${PWD}/config:/out crender-tools tools/ai/fetch_detector.py -o /out/yolov8n.onnx
```

Inside the container ONNX Runtime runs the CPU execution provider — fine for
model generation/verification. The `d3d11.dll` proxy uses ORT-DirectML on the
host, so verify generated models actually run there too (`tools/ai/verify_terminator.py`).

## Windows build image

```powershell
docker build -f docker/windows-build.Dockerfile -t crender-build-win .

docker create --name cbuild crender-build-win
docker cp cbuild:C:/src/build/win32/Release/d3d11.dll ./d3d11-x86.dll
docker cp cbuild:C:/src/build/win32/Release/test_cube.exe ./test_cube-x86.exe
docker cp cbuild:C:/src/build/x64/Release/d3d11.dll ./d3d11-x64.dll
docker rm cbuild
```

Notes:

- **Host OS pairing** — `ltsc2022` requires Windows 10 21H2+ / Windows 11 /
  Server 2022, usually with `docker build --isolation=process` if kernel
  versions don't match. `docker system info` reports the needed values.
- **Toolchain** — the repo presets use the `Visual Studio 18 2026` generator;
  the image installs VS Build Tools 2022 (v14.4x) and drives it with Ninja
  instead. Same compiler tech, but `/W4 /WX` is strict — if newer warnings
  break the build, rebuild with `-e VS_CHANNEL=18/release`-style override via
  `docker build --build-arg VS_CHANNEL=18/release` (once Microsoft publishes
  that channel) or relax `/WX`.
- **Size** — expect a 10-15 GB image; first build downloads the VS installer
  plus the Windows base layers.
- **No deploy step** — `HEAVEN_BIN` doesn't exist in the container, so the
  `POST_BUILD` copy in `CMakeLists.txt` is silently skipped.
- **ORT** — `fetch_ort.ps1` runs inside the image so the Terminator path builds
  with `RETRO_HAVE_ORT=1`. Comment that `RUN` line to save ~200 MB.
