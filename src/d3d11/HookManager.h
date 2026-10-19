// HookManager — MinHook vtable hooks on the captured D3D11 objects (Phase 3).
// Counters prove interception; retro rendering work lands on top of these in Phase 4+.
#pragma once

#include <d3d11.h>
#include <dxgi.h>

namespace hooks
{

// Retro render modes, cycled by F9. Order matches config "Mode" values.
enum class RetroMode : int
{
    Off = 0,
    NullPS,     // PSSetShader(nullptr) — depth/geom only
    FlatPS,     // substitute a constant-color pixel shader
    Wireframe,  // RSSetState(wireframe rasterizer)
    RetroFX,    // Present post-process: pixelate + palette + scanlines
    Ascii,      // Present post-process: glyph-atlas ASCII render
    ConsoleWire,// CPU wireframe into an AllocConsole ASCII grid
    Terminator, // Present: ONNX (ORT DirectML) or HLSL base -> HUD -> lens. Config alias "Neural".
};

RetroMode GetMode() noexcept;
void      SetMode(RetroMode mode) noexcept;
const char* ModeName(RetroMode mode) noexcept;

// MinHook global init/teardown. InitOnce is lazy-safe — call from any thread.
void InitOnce() noexcept;
void Shutdown() noexcept;

// Called by the wrapped D3D11CreateDevice* after a successful create.
// Hooks the immediate-context vtable and reaches the DXGI factory through the
// device (device -> IDXGIDevice -> IDXGIAdapter -> IDXGIFactory) so we can catch
// swap chains created via CreateSwapChain / CreateSwapChainForHwnd.
void OnDeviceCreated(ID3D11Device* dev, ID3D11DeviceContext* ctx, IDXGISwapChain* sc) noexcept;

// Hooks Present on a swap chain captured by a factory detour (or AndSwapChain path).
void InstallSwapChainHooks(IDXGISwapChain* sc) noexcept;

// Periodic stats emit — self-throttles to ~2 s; safe to call at 50 ms cadence.
void StatsTick() noexcept;

// Polls F9 (cycle mode), F10 (panic: disable/enable all hooks), F11 (reload retro.json).
// Call every ~50 ms from the proxy init thread.
void HotkeyTick() noexcept;

// Reads "Mode" from retro.json next to the proxy DLL. Called at init and on F11.
void LoadConfigFile() noexcept;

// Loads retro.json exactly once (thread-safe); later callers wait for it.
void EnsureConfigLoaded() noexcept;

} // namespace hooks
