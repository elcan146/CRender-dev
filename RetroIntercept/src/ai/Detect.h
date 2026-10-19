// Detect — Terminator Vision object detector: second ONNX session (YOLOv8-style output)
// on the game's adapter via ORT DirectML. Samples the raw backbuffer (before the Terminator
// base restyles it) into its own shared NCHW texture, decodes boxes CPU-side, filters by
// configured class names/ids, feeds term::Tracker, and derives sight lines (weapon->owner
// heuristic: the person box containing the weapon, else nearest person).
//
// Self-contained: own D3D12 device/queue/env — independent of ai::Ensure so it works with
// the HLSL base too. When the model/config/DLLs are missing it stays inert and the mesh-AABB
// scanner remains the bracket source.
#pragma once

#include <d3d11.h>
#include <dxgi.h>

#include "../term/Hud.h"
#include "../term/Tracker.h"

namespace det
{

struct Config
{
    char  model[260]  = "yolov8n.onnx";     // session 0 — "" disables detection
    char  model2[260] = "";                 // session 1 (optional, e.g. weapon-trained)
    float conf      = 0.35f;                // DetConf
    int   stride    = 2;                    // DetStride — infer every Nth present
    int   maxDets   = 32;
    bool  sight     = true;                 // DetSight — weapon sight lines
    float sightLen  = 0.55f;                // normalized line length beyond the weapon
    int   classes[16]   = { 0 };            // session-0 ids
    int   nClasses      = 1;
    int   classes2[16]  = { 0, 1 };         // session-1 ids (default: guns+knife)
    int   nClasses2     = 2;
    bool  weapons[256]  = {};               // ids flagged as weapons (session-0 space)
    bool  weapons2[256] = {};               // session-1 weapon ids
    int   personClass   = 0;                // anchor class for weapon ownership (sess 0)
    int   dump        = 0;                  // DetDump — write input PPMs at probe runs
};

// Thread-safe (config/hotkey thread). classes/weapons are CSV names (COCO table,
// session 0 only) or numeric ids (either session).
void SetConfig(const char* model, const char* model2, float conf, int stride,
               bool sight, float sightLen,
               const char* classesCsv, const char* weaponsCsv,
               const char* classes2Csv, const char* weapons2Csv, int dump) noexcept;

// Lazily builds D3D12/DML/ORT/buffers. Returns false (and latches) on failure.
bool Ensure(ID3D11Device* dev) noexcept;

// Runs the detector on sc's CURRENT backbuffer (call before ai::Apply restyles it).
// Cheap on off-stride frames.
void Run(ID3D11DeviceContext* ctx, IDXGISwapChain* sc) noexcept;

// Present: ticks the tracker with the latest decode + recomputes sight lines.
// Returns the live track list (valid until the next Update).
const term::Track* Update(float dt, int maxTargets, int* count) noexcept;

// Per-track display labels parallel to Update()'s track list (nullptr entries allowed).
const char* const* Labels() noexcept;

// Sight-line segments for the current tracks (normalized). n up to 32.
const hud::Line* SightLines(int* n) noexcept;

const char* Label(int cls) noexcept;   // COCO name or "OBJ nn"
bool        Active() noexcept;         // True while the session is live (Active() can gate HUD labels).

// Drops cached backbuffer refs (call before game ResizeBuffers).
void ReleaseBackbuffer() noexcept;

void        Reset() noexcept;

} // namespace det
