# Tools

## Install / deploy

| Tool | Purpose |
|---|---|
| `CRender_wizard.py` | GUI installer — pick any game folder, **Apply** deploys the proxy + config + models next to the real `.exe`, **Rewind** restores backups. `pip install PyQt6` |
| `crenderctl.ps1` | CLI equivalent: `-Enable` / `-Disable` / `-Status`, `-Neural` adds the ORT/DirectML runtime + ONNX models. Tracks installs via `.crenderctl.json` in the game dir |

## Preview

| Tool | Purpose |
|---|---|
| `screen_vision.py` | Desktop port of the Terminator pipeline: transparent click-through HUD overlay or styled mirror window over your live screen. `F8` palette, `F9` style, `F10` panic, `F11` reload, `End` quit |

## AI helpers (`ai/`)

| Tool | Purpose |
|---|---|
| `fetch_ort.ps1` | Downloads ONNX Runtime DirectML EP into `third_party/onnxruntime/` (x86 `bin/`, x64 `bin64/`) |
| `fetch_detector.py` | Exports an Ultralytics `.pt` checkpoint to a fixed-shape `config/yolov8n.onnx` detector |
| `make_placeholder.py` | Builds the bundled `retro_ai.onnx` pixel-art style model (pure DML-friendly ops) |
| `make_terminator_onnx.py` | Builds `terminator_vision.onnx` (bg+edge encoding used by the Terminator base layer) |
| `verify_terminator.py` | Sanity-checks the Terminator ONNX model on a still image |
| `sanity_check.py` | Quick ORT/model environment check |
| `test_bestgun.py` | Evaluates `best.onnx` (gun detector) against test images |
| `wrap_style.py` | Wraps a style-transfer ONNX for the expected `[1,3,H,W]` contract |
| `render_model_graph.py` | Renders an ONNX op graph as a Netron-style PNG+SVG (folded initializers, measured text fit) |
| `testimg/` | Sample inputs + outputs for detector/model tests |

`yolov8n.pt` is the default checkpoint `fetch_detector.py` consumes; a copy of the
exported `yolov8n.onnx` lives in `config/`.

## Misc

| Tool | Purpose |
|---|---|
| `test_cube/` | Minimal D3D11 app (`test_cube.exe`) for proxy testing — `F1` toggles a heavy pixel shader |
| `d3d11_export_names.txt`, `d3d11_exports_syswow64.txt` | Reference export lists used to generate the proxy forwarders |
