# CRender

D3D11 proxy DLL that intercepts a game's rendering pipeline and re-presents frames in a
retro style: **GPU ASCII shader pass** (primary) and/or **CPU console wireframe** (secondary).

![ASCII mode in Unigine Heaven](docs/images/heaven_ascii_1440p.png)

> Offline / single-player targets only. Graphics-layer interception (ReShade/DXVK category).
> No game-state writes, no anti-cheat interaction.

## Status

**Phase 12+** — performance pass (`RenderScale`, skip rules, `GpuBurn`) landed on top of the
full presentation layer; Terminator stack runs ONNX models on ORT-DML with an HLSL fallback.
Technical report: [`docs/CRender-Technical-Report.pdf`](docs/CRender-Technical-Report.pdf)
(HTML source + numbered ONNX graph SVGs under `docs/images/`).

| Piece | State |
|---|---|
| `d3d11.dll` proxy (x86 + x64, `/MT`, KERNEL32-only) | 51 exports, correct ordinals |
| **Perf (Phase 12)** | `RenderScale` N — real backbuffer created at 1/N (`BufferDesc` shrunk + `STRETCH`; verified on Heaven — believed dims kept so texel-`Load()` post stays consistent) + screen-sized RT/DS textures at 1/N, viewports/scissors rescaled; `SkipShadows`, `SkipPS` hash blocklist, `ShaderStrip` skip rules; `GpuBurn` N — synthetic per-frame GPU load for benchmarking; `PassDump` logs `ps=%016llx` per pass; stats line reports `fps=`/`cpuFrameMs=`/`skipped=` |
| Context hooks | counters + `PSSetShader`/`RSSetState`/`VSSetCBs`/`UpdateSubresource`/`IASetVertexBuffers`/`IASetIndexBuffer` |
| **Modes** | `Off / NullPS / FlatPS / Wireframe / RetroFX / Ascii / ConsoleWire / Terminator` — F9, F10, F11. `"Neural"` in config maps to `Terminator` |
| **Terminator** | Three layers on the presented frame: base = ONNX model on the GPU via ONNX Runtime **DirectML EP** (`LoadLibrary`'d, zero-copy shared-buffer interop) with an HLSL Sobel/luma fallback when ORT or the model is unavailable → HUD overlay (`TermHud*` keys: instanced glyph text/reticle, target brackets) → lens (`TerminatorLens`: CRT barrel + scanlines + red vignette). `TermEdgeGain`/`TermTint` re-grade the model's `R=bg+e, G=B=0.7e` encoding; `HudMaxTargets` caps brackets. `Ai*` keys drive the ONNX base; stats `aiMs=` |
| **Terminator detect** | Up to **two** ORT/DML detector sessions on the raw frame: `DetModel` (session 0, COCO names in `DetClasses`/`DetWeapon`) + optional `DetModel2` (session 1, `DetClasses2`/`DetWeapon2` numeric ids — e.g. `det_weapons.onnx` 0=guns,1=knife). YOLOv8-style `[1,4+K,N]` output, per-class NMS, merged into `term::Tracker` (session-1 ids offset +1000, shown as `GUN`/`KNIFE`/`DET nn`); `DetSight` draws dotted weapon→owner rays anchored on `person` tracks. No model → mesh-AABB scan fallback. `tools/ai/fetch_detector.py` exports `yolov8n.onnx` (`pip install ultralytics onnx`) |
| **Hotkeys** | F9 cycle (Off→RetroFX→Ascii→Terminator) · F10 panic toggle · F11 reload config · **F12 RenderDoc capture** (when injected) |
| **RetroFX** | pixelate + 3-bit palette + scanlines + CRT barrel + vignette (`Crt`/`Vignette`) |
| **Ascii** | cell-grid glyph atlas — `CellPx` (min 4, 8 recommended), 4-tap luminance, **`Palette`: green/amber/mono/cyan/color**. With `RenderScale` > 1, RetroFX/Ascii run at native output res (crisp glyphs) sampling the low-res scene |
| **ConsoleWire** | `world*view*proj` → `WriteConsoleW` grid — **slope-matched `─│╲╱` edges**, depth fade (`─`→`+`→`.`), backface cull (`WireCull`) |
| Transform map | 144B slot-0 CB: proj@0, view@64, params@128. 64B CB0: per-draw world |
| Mesh readback | `DumpMesh`/`WireDraws` budgets; pooled staging buffers (8-slot round-robin) |
| Shutdown | `hooks::Shutdown` releases MinHook + retro objects + staging pool |
| Deploy | `POST_BUILD` copies `d3d11.dll` + `retro.json` to `HEAVEN_BIN` |

Gotchas: `DrawAuto`=vtbl[38]; Heaven swap chain via `IDXGIFactory::CreateSwapChain`; CBs via
`UpdateSubresource`; `SendInput` blocked — `keybd_event` works. `RenderScale` applies only at
launch (RTs + backbuffer are created once): it shrinks the real backbuffer (`BufferDesc/s`,
`STRETCH` scaling) so texel-space `Load()` pipelines like Unigine's stay consistent — RT-only
scaling blacks the screen because the final composite would read shrunken sources at native
coords. If SSAO/deferred passes sample wrong, try `"PatchScreenCB": 1`.

## Target

Primary test target: **Unigine Heaven Benchmark 4.0** (`F:\Heaven Benchmark 4.0\bin\Heaven.exe`).
**Important: it is 32-bit (PE32).** The engine `Unigine_x86.dll` loads `d3d11.dll` via
`LoadLibrary` at runtime (D3D11 selectable in the launcher). Build everything as **Win32**.

## Build

Toolchain: VS Build Tools 2026 at `C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools`
(bundled CMake 4.2.3 at `Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe`;
`dumpbin` at `VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64\dumpbin.exe`).

```powershell
cmake --preset win32-msvc
cmake --build --preset win32-release
# outputs: build/win32/Release/d3d11.dll, build/win32/Release/test_cube.exe
```

x86 output lands in `build/win32`, x64 in `build/x64` (preset `x64-msvc`/`x64-release`; the
forwarding stubs are generated MASM there since MSVC x64 has no `__asm`).

### Terminator ONNX deps

```powershell
powershell -ExecutionPolicy Bypass -File tools/ai/fetch_ort.ps1   # -> third_party/onnxruntime (ORT-DML 1.22.1 x86+x64 + DirectML 1.15.4)
pip install onnx numpy
python tools/ai/make_placeholder.py -o config/retro_ai.onnx
python tools/ai/make_placeholder.py --two-input -o config/retro_ai_temporal.onnx
cmake --preset win32-msvc   # re-configure: detects third_party/onnxruntime -> RETRO_HAVE_ORT=1
```

Without `third_party/onnxruntime` the proxy still builds; the Terminator base layer logs a warning and uses the HLSL fallback.
`POST_BUILD` also deploys `onnxruntime.dll`, `DirectML.dll` and `retro_ai*.onnx` to `HEAVEN_BIN`.
`onnxruntime.dll` needs the matching-arch VC++ 2015+ runtime (`vcruntime140.dll`/`msvcp140.dll`
in `SysWOW64` for x86 games, `System32` for x64).
Models must have fixed shape `[1,3,AiHeight,AiWidth]` fp32 (dynamic shapes are rejected);
`AiWidth` must be a multiple of 64 (D3D12 texture-copy row alignment). Any `ai fail:` log
line latches the ONNX path off until F11 (HLSL fallback takes over).

The bundled placeholder is a real pixel-art pass (blockify → saturate → contrast →
posterize) built from pure DirectML-friendly ops — tune with `--pixels`, `--levels`,
`--saturation`, `--contrast`, or use `--style edge` for the old conv edge look. For actual
style-transfer art, drop in any fp32 NCHW `[1,3,H,W]` ONNX as `retro_ai.onnx`.

### Docker

Two images under `docker/`: `tools.Dockerfile` (Linux, Python AI tooling) and
`windows-build.Dockerfile` (Windows container with VS Build Tools that compiles
`d3d11.dll`/`test_cube.exe` x86+x64 — extract via `docker cp`). The proxy's
runtime — intercepting a real game on a GPU desktop — cannot run in a container.
See [`docker/README.md`](docker/README.md).

## Test

- `test_cube.exe` alone: rotating cube, title shows live FPS, **F1** toggles heavy↔light pixel shader.
- Proxy test: copy `d3d11.dll` next to `test_cube.exe` (once Phase 2 exports exist).
- Heaven: copy `d3d11.dll` to `F:\Heaven Benchmark 4.0\bin\`, select Direct3D11 in the launcher.

## Tools

See [`tools/README.md`](tools/README.md). Quick start:

```powershell
python tools/CRender_wizard.py    # GUI installer — pick a game folder, Apply, Rewind to undo
.\tools\crenderctl.ps1 -Path "<game dir>" -Enable   # CLI equivalent
python tools/screen_vision.py     # desktop preview of the Terminator/ASCII pipeline
python tools/ai/render_model_graph.py --all config docs/images   # Netron-style model graphs (PNG+SVG)
```

## Layout

```
src/proxy/    d3d11.dll export forwarding + proxy entry
src/core/     config + logging
src/d3d11/    hooks, render passes, state tracking
src/ai/       ONNX Runtime / DirectML inference
src/term/     HUD + target tracker
config/       retro.json, ONNX models, per-game configs
docs/         technical report (HTML+PDF), ONNX model graphs (1-…6-*.svg), internals notes, images
tools/        installer GUI/CLI, AI helpers, test_cube
third_party/  MinHook, ONNX Runtime, style models
```

## Safety / legal

For games you own, offline, no anti-cheat. See `docs/agent-prompt.md` §2.1 for the full constraints.
