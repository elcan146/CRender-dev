// Hud — Terminator Vision overlay: scrolling diagnostic text columns, center reticle with
// drifting coordinates, and animated target brackets. Rendered as instanced glyph/rect/ring
// quads from the shared retro glyph atlas (retro::GlyphAtlas) in one draw call.
// Shared by the proxy (game mode) and TerminatorStream; depends only on RetroPass.
#pragma once

#include <d3d11.h>

#include "Tracker.h"

namespace hud
{

// Normalized 0..1 screen-space segment (sight lines / tethers), drawn dotted.
struct Line
{
    float x0, y0, x1, y1;
};

struct Config
{
    bool  enabled    = true;  // TermHud
    bool  text       = true;  // TermHudText — scrolling columns
    bool  brackets   = true;  // TermBrackets
    bool  reticle    = true;  // TermReticle
    float speed      = 1.0f;  // TermHudSpeed — scroll/animation rate multiplier
};

void   SetConfig(const Config& c) noexcept;
Config GetConfig() noexcept;

// Draws the HUD onto rtv (w x h px) with alpha blending. tracks are normalized 0..1.
// labels: optional parallel array of per-track class names (nullptr entry -> "TGT nn").
// lines: optional normalized sight/tether segments, drawn as dotted lines under brackets.
// status: optional fixed lines shown at the top of the right column (real stats).
// Captures and restores all pipeline state it touches. Requires retro::Ensure first.
void Draw(ID3D11DeviceContext* ctx, ID3D11RenderTargetView* rtv, UINT w, UINT h,
          const term::Track* tracks, int nTracks,
          const char* const* status, int nStatus,
          const char* const* labels = nullptr,
          const Line* lines = nullptr, int nLines = 0) noexcept;

// Releases device objects (shutdown / device change).
void Reset() noexcept;

} // namespace hud
