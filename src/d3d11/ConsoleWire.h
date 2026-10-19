// ConsoleWire — CPU ASCII wireframe output (secondary render path).
//
// Collects budgeted geometry snapshots (VB/IB readback done by HookManager),
// projects verts with the tracked 144-byte projection CB, rasterizes triangle
// edges into a char grid via Bresenham, and blits to an AllocConsole window
// at ConsoleHz. Heavy work is deferred to frame end (called from hkPresent).
#pragma once

#include <d3d11.h>
#include <dxgi.h>

namespace wire
{

// Per-frame collection control (read from retro.json at config load).
extern int  g_maxDrawsPerFrame; // "WireDraws": max DrawIndexed snapshots/frame
extern int  g_consoleW;         // "ConsoleWidth"
extern int  g_consoleH;         // "ConsoleHeight"
extern int  g_consoleHz;        // "ConsoleHz" — blit rate cap
extern int  g_wireCull;         // "WireCull" — 0/1 backface culling

// Called on mode switch / config load: lazily allocates the console.
void EnsureConsole() noexcept;

// Frame boundary: reset the per-frame collection budget.
void BeginFrame() noexcept;

// Feed a captured mesh (verts are float3 positions at 'stride' bytes;
// indices read at ix[startIndex..startIndex+indexCount) + baseVertex).
// 'world' is the per-draw 64B CB0 matrix (row-major v*M) — verts are
// world-transformed at submit time; may be null (identity).
void Submit(const void* verts, UINT vbBytes, UINT stride,
            const void* indices, UINT startIndex, UINT indexCount, INT baseVertex,
            DXGI_FORMAT indexFmt, const float* world) noexcept;

// Frame boundary: rasterize collected geometry + blit to console (rate-capped).
// Reads back projCB (the tracked 144B slot-0 CB) itself; null -> fixed ortho.
void EndFrame(ID3D11DeviceContext* ctx, ID3D11Device* dev,
              ID3D11Buffer* projCB) noexcept;

// Returns true if the per-frame submission budget still has room.
bool WantsMore() noexcept;

} // namespace wire
