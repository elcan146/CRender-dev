# AI tooling image — Linux. Covers tools/ai/*.py (model generation, wrapping,
# verification, detector tests). Runs ONNX Runtime CPU EP — no GPU needed.
#
#   docker build -f docker/tools.Dockerfile -t crender-tools .
#   docker run --rm crender-tools tools/ai/make_placeholder.py --width 320 --height 180 -o /out/retro_ai.onnx
#
# Mount a host dir at /out to collect generated files:
#   docker run --rm -v ${PWD}/config:/out crender-tools tools/ai/make_placeholder.py -o /out/retro_ai.onnx
#
# NOT covered (Windows desktop required): screen_vision.py, CRender_wizard.py,
# test_cube.exe, and the d3d11.dll proxy itself.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# CPU-only torch first so ultralytics doesn't pull the multi-GB CUDA wheels.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY tools/requirements.txt tools/requirements.txt
# ultralytics hard-requires opencv-python; reinstall headless last so cv2 works
# without X11 libs in the slim image.
RUN pip install -r tools/requirements.txt \
 && pip install --force-reinstall --no-deps opencv-python-headless

COPY tools/ tools/
COPY config/ config/

WORKDIR /app
ENTRYPOINT ["python"]
