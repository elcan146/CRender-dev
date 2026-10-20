// Targets — game-mode bracket sources for Terminator Vision.
//
// Local AABBs come from ID3D11Device::CreateBuffer initial data (position f32x3 @0,
// computed for every candidate stride 12..64 since the stride is only known at bind time).
// The world matrix (64B VS slot-0 CB) and proj/view (144B CB: proj @0, view @64) are
// captured CPU-side from UpdateSubresource. Per scene draw the 8 corners are projected
// (row-vector convention, same as ConsoleWire) into a normalized screen box; Present
// feeds the frame's best boxes to term::Tracker. No GPU readback anywhere.
// Dynamic / late-filled vertex buffers never get a bracket.
#pragma once

#include <d3d11.h>

#include "../term/Tracker.h"

namespace targets
{

// Any thread. Records (or clears) the AABB set for a freshly created buffer.
void OnCreateBuffer(const D3D11_BUFFER_DESC* desc, const D3D11_SUBRESOURCE_DATA* init,
                    ID3D11Buffer* buf) noexcept;

// Render thread. Full-buffer CB write (UpdateSubresource, box == nullptr).
void OnUpdateCB(ID3D11Buffer* buf, UINT byteWidth, const void* data) noexcept;

// Render thread, scene draws only (screen-sized RT bound, not our own passes).
void OnDraw(ID3D11Buffer* vb, UINT stride, UINT vbOffset, ID3D11Buffer* vsCB0) noexcept;

// Present: turns this frame's candidates into tracks (top maxTargets) and resets.
// Returns the live track list (valid until the next EndFrame).
const term::Track* EndFrame(float dt, int maxTargets, int* count) noexcept;

// Scene gate (render thread): while muted the tracker is fed an empty list, so
// stale/false-positive tracks coast out instead of stacking hits behind the gate.
void SetMuted(bool muted) noexcept;

void Reset() noexcept;

} // namespace targets
