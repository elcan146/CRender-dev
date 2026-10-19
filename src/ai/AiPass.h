// AiPass — RetroMode::Terminator base layer: game backbuffer -> ONNX model on the GPU
// (ORT DirectML EP) -> stylized result written back into the backbuffer, before the HUD,
// lens and UpscaleBlit/Present. When Ensure() fails, HookManager uses retro::Fx::TermBase.
//
// Interop: three linear float buffers (input, output[2] ping-pong) are created as shared
// D3D12 committed resources on the game's adapter and opened on the D3D11 device. The
// D3D11 side writes NCHW planar input via a UAV pixel pass and reads output via SRV;
// ORT binds the same memory through CreateGPUAllocationFromD3DResource. Zero CPU copies.
// Sync (v1): D3D11 event query before Run, D3D12 fence wait after Run.
#pragma once

#include <d3d11.h>
#include <dxgi.h>

namespace ai
{

// Thread-safe (hotkey/config thread). Structural changes (model, dims, temporal mode)
// are applied on the next Ensure() by tearing the session down and rebuilding it.
// model: path relative to the proxy dir (or absolute). temporal: 0 single-input,
// 1 lerp(cur, prevOut, blend), 2 two-input model (prev output bound as input 1).
void SetConfig(const char* model, int width, int height, int temporal, float blend, int stride) noexcept;

// Terminator look applied in the output pass: edge multiplier (TermEdgeGain) and void
// tint (TermTint[3]). Identity at (1, {1,0,0}) — any other model passes through untouched.
void SetLook(float edgeGain, const float* tint) noexcept;

// Lazily builds D3D12/DML/ORT/session/buffers. Returns false (and latches) on failure.
// Requires retro::Ensure(dev) to have succeeded first (shared fullscreen VS).
bool Ensure(ID3D11Device* dev) noexcept;

// Full per-frame pipeline on sc's backbuffer. Call between retro::Ensure and UpscaleBlit.
void Apply(ID3D11DeviceContext* ctx, IDXGISwapChain* sc) noexcept;

// Drops backbuffer references. Must run before ResizeBuffers.
void ReleaseBackbuffer() noexcept;

// Average inference ms (Run + fence wait) and run count since the previous call.
void TakeStats(double* avgMs, long long* runs) noexcept;

// Releases everything (ORT session/env, D3D12/DML objects, buffers) and clears the latch.
void Reset() noexcept;

} // namespace ai
