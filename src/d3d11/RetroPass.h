// RetroPass — post-process applied inside the Present detour (Phase 6).
// Two effects: RetroFX (pixelate + palette + scanlines) and Ascii (cell grid ->
// luminance -> generated 8x8 glyph atlas). Self-contained, no external assets.
#pragma once

#include <d3d11.h>
#include <dxgi.h>

namespace retro
{

// Effect selector matching the RetroMode entries that reach us.
enum class Fx : int
{
    None = 0,
    RetroFX, // pixelate + palette + scanlines
    Ascii,   // luminance -> glyph atlas
    TermBase,// Terminator base layer (HLSL fallback for terminator_vision.onnx)
    TermLens,// Terminator lens: CRT barrel + scanlines + red vignette, no pixelate/palette
};

// Set pixel-block size (RetroFX) and glyph-cell size (Ascii). Safe before Ensure.
void SetTunables(float pixelPx, float cellPx) noexcept;

// Set CRT barrel-distortion strength and vignette amount (RetroFX). Safe before Ensure.
void SetCrt(float crt, float vignette) noexcept;

// Terminator base look: edge multiplier (TermEdgeGain) and void tint (TermTint, default red).
void SetTermLook(float edgeGain, const float tint[3]) noexcept;

// TerminatorLens preset: barrel strength, vignette amount, scanline depth (0..1).
void SetLens(float crt, float vignette, float scanlines) noexcept;

// Shared 8x8 glyph atlas (valid after Ensure): [0..9] Ascii density ramp, then HUD text.
ID3D11ShaderResourceView* GlyphAtlas() noexcept;
// Atlas index for a HUD character (case-insensitive); -1 = blank/unsupported.
int GlyphIndex(char c) noexcept;

// Select Ascii phosphor palette: 0 green, 1 amber, 2 mono, 3 cyan, 4 color (scene hue).
void SetPalette(int palette) noexcept;
int  GetPalette() noexcept;

// Scene render-scale divisor (from RenderScale). Floors PixelPx/CellPx to N so the
// upscaled pixels stay aligned to the retro grid. 1 = no scaling.
void SetRenderScale(int scale) noexcept;

// Synthetic GPU load: number of heavy fullscreen passes burned per frame in
// BurnGpu. 0 = off. Used to emulate a weaker GPU ("throttle") for benchmarking.
void SetGpuBurn(int passes) noexcept;

// Runs the burn passes into an offscreen scratch RT sized to the swap chain
// backbuffer. Call from the Present hook — the work serializes with the frame.
void BurnGpu(ID3D11DeviceContext* ctx, IDXGISwapChain* sc) noexcept;

// Lazily builds all device objects on first call; safe to call every frame.
void Ensure(ID3D11Device* dev) noexcept;

// Runs the chosen effect onto the swap chain backbuffer. Call BEFORE the real Present.
void Apply(ID3D11DeviceContext* ctx, IDXGISwapChain* sc, Fx fx) noexcept;

// Runs `fx` in place on an arbitrary present target (own copy chain, keyed by texture).
// Returns false if the pass is unavailable.
bool ApplyTo(ID3D11DeviceContext* ctx, ID3D11Texture2D* target, Fx fx) noexcept;

// RTV (+ dims) on the same target ApplyTo uses — for overlay passes (HUD). Non-owning.
ID3D11RenderTargetView* PostRTV(ID3D11Texture2D* target, UINT* w, UINT* h) noexcept;

// Drops every backbuffer reference (copy/RTV/bb, incl. the ApplyTo chain). Must run before ResizeBuffers.
void ReleaseBackbuffer() noexcept;

// Integer-scaled present: point-samples src swap chain's (shrunken) backbuffer
// into dst chain's native-size backbuffer at an integer factor, letterboxed.
// fx != None renders that effect at native res instead of a plain blit, so
// glyphs/scanlines stay crisp. Called INSTEAD of the game's Present when RenderScale > 1.
void UpscaleBlit(ID3D11DeviceContext* ctx, IDXGISwapChain* src, IDXGISwapChain* dst,
                 Fx fx = Fx::None) noexcept;

bool UpscaleTexture(ID3D11DeviceContext* ctx, ID3D11Texture2D* src, IDXGISwapChain* dst,
                    Fx fx = Fx::None) noexcept;

// Drop the cached dst-chain RTV (call when the dst swap chain is released/recreated).
void DropUpscaleChain() noexcept;

// Releases cached objects. Called on shutdown.
void Reset() noexcept;

// Shared with AiPass (valid after Ensure). CompileShader returns a blob the caller Releases.
ID3DBlob*           CompileShader(const char* src, const char* entry, const char* target) noexcept;
ID3D11VertexShader* FullscreenVS() noexcept;
// Binds the fullscreen-triangle pipeline: VS, no IL/GS/HS/DS, opaque blend, no depth, solid RS.
void                BindFullscreenState(ID3D11DeviceContext* ctx) noexcept;

} // namespace retro
