// HookManager — MinHook vtable hooks on the captured D3D11 objects (Phase 3).
//
// Technique: read *object -> vtbl -> entry[index], MH_CreateHook the function. Hooking a
// vtable slot covers every instance of that COM class (all contexts/swap chains of the
// same implementation share the vtable).
//
// Phase 3 hooks are pure counters — they prove interception while staying pass-through.
#include "HookManager.h"
#include "RetroPass.h"
#include "ConsoleWire.h"
#include "Targets.h"
#include "../ai/AiPass.h"
#include "../ai/Detect.h"
#include "../core/Config.h"
#include "../core/Log.h"
#include "../term/Hud.h"

#include <Windows.h>
#include <dxgi1_4.h>
#include <d3dcompiler.h> // types only — resolved via GetProcAddress, no import dep
#include <MinHook.h>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace
{

volatile LONGLONG g_draws = 0;
volatile LONGLONG g_maps = 0;
volatile LONGLONG g_presents = 0;

volatile LONG g_mhState = 0; // 0 none, 1 init, 2 ready, 3 failed
volatile LONG g_hooksActive = 0; // F10 panic toggle
volatile LONG g_mode = 0;        // RetroMode as int

ID3D11Device*         g_device = nullptr;   // captured, OWNING (AddRef'd)
ID3D11DeviceContext*  g_ctx    = nullptr;   // captured, OWNING (AddRef'd)

// Deferred COM release queue. Releasing an interface inside OnDeviceCreated runs
// its destructor on the game's D3D11CreateDevice call stack — Yakuza 0 faulted
// inside dxgi this way when the second (BGRA/D2D) device displaced the first.
// Queue old captures; DrainReleases() runs them from hkPresent/Shutdown instead.
IUnknown*       g_pendRel[16] = {};
volatile LONG   g_pendRelN = 0;

void DeferRelease(IUnknown* p) noexcept
{
    if (p == nullptr)
        return;
    const LONG i = InterlockedIncrement(&g_pendRelN) - 1;
    if (i >= 0 && i < static_cast<LONG>(sizeof(g_pendRel) / sizeof(g_pendRel[0])))
        g_pendRel[i] = p;
    else
        InterlockedDecrement(&g_pendRelN); // full — leak rather than release in-place
}

void DrainReleases() noexcept
{
    for (LONG i = 0; i < static_cast<LONG>(sizeof(g_pendRel) / sizeof(g_pendRel[0])); ++i)
    {
        IUnknown* p = static_cast<IUnknown*>(
            InterlockedExchangePointer(reinterpret_cast<PVOID*>(&g_pendRel[i]), nullptr));
        if (p != nullptr)
            p->Release();
    }
    InterlockedExchange(&g_pendRelN, 0);
}
ID3D11PixelShader*    g_flatPS = nullptr;   // created lazily on first FlatPS mode
ID3D11RasterizerState* g_wireRS = nullptr;  // created lazily on first Wireframe mode
volatile LONG g_flatPSTried = 0;
volatile LONG g_wireRSTried = 0;

// ---- Phase 5 recon: CB dump machinery -----------------------------------------
volatile LONG g_dumpCBs = 0; // retro.json "DumpCBs": N — dump N small CB writes, then stop

struct MapRec  { ID3D11Resource* res; void* ptr; };
struct SlotRec { ID3D11Buffer* buf; int slot; };
struct DumpedRec { ID3D11Buffer* buf; int times; };
MapRec    g_mapRecs[32];    // hashed by resource pointer; recon-grade, overwrite ok
SlotRec   g_cbSlots[64];    // hashed by buffer pointer
DumpedRec g_dumped[128];    // per-buffer dedup so one hot buffer can't eat the budget

// ---- Phase 7: geometry readback -------------------------------------------------
volatile LONG g_dumpMesh = 0; // retro.json "DumpMesh": N — capture N DrawIndexed calls

struct VBBind { ID3D11Buffer* buf; UINT stride; UINT offset; };
struct IBBind { ID3D11Buffer* buf; DXGI_FORMAT fmt; UINT offset; };
VBBind g_vb0 = {};
IBBind g_ib  = {};
ID3D11Buffer* g_vsCB0 = nullptr; // last buffer bound at VS slot 0
ID3D11Buffer* g_projCB = nullptr; // last 144B CB bound at VS slot 0 (perspective proj)

// ---- Phase 12: low-res render injection + pass skipping -------------------------
//
// The retro modes above only restyle the frame — the game still renders the whole
// scene at native res, so there is no perf win. This section makes the game render
// its scene at 1/N resolution (RenderScale) and drops passes the retro look does
// not need (shadow maps, blocklisted post-FX shaders, ortho HUD quads).
volatile LONG g_renderScale   = 1;   // retro.json "RenderScale": 1=off, N=scene res divisor
volatile LONG g_gpuBurn       = 0;   // "GpuBurn": heavy fullscreen passes per frame (GPU throttle)
volatile LONG g_patchScreenCB = 0;   // "PatchScreenCB": rewrite screen-dim floats in CB writes
volatile LONG g_skipShadows   = 0;   // "SkipShadows": drop depth-only offscreen passes
volatile LONG g_skipOrtho     = 0;   // ShaderStrip "SkipOrthographic"
volatile LONG g_skipMaxIndex  = 0;   // ShaderStrip "SkipMaxIndexCount"
volatile LONG g_passDump      = 0;   // "PassDump": N frames of per-PS pass discovery logging
volatile LONG  g_inRetroPass   = 0;   // guard: our fullscreen pass is never skipped/scaled
volatile LONG  g_inSCCreate    = 0;   // guard: swap-chain-internal ops (bb create / our own ResizeBuffers) must not be rescaled again
IDXGISwapChain* g_sc2 = nullptr;        // our native-size present chain (integer upscale target)
ID3D11Texture2D* g_proxyBB = nullptr;
IDXGISwapChain*  g_proxySC = nullptr;
DXGI_FORMAT      g_proxyFmt = DXGI_FORMAT_UNKNOWN;
volatile LONGLONG g_skips     = 0;

UINT g_scW = 0, g_scH = 0;           // swap-chain dims = the game's "screen size"
volatile LONG g_scIsFlip = 0;        // flip-model chain (FLIP_*): real bb stays native -
                                     // the game renders into a 1/N proxy returned by GetBuffer

// Resources we shrank at creation: ptr -> original (game-believed) dims.
struct ScaledRec { ID3D11Resource* res; UINT w, h; };
ScaledRec g_scaledRes[1024] = {};

// Last viewport/scissor set by the game (unscaled, as the game believes).
D3D11_VIEWPORT g_vp[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE] = {};
UINT           g_vpN = 0;
D3D11_RECT     g_sr[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE] = {};
UINT           g_srN = 0;
volatile LONG  g_scaledBound = 0;    // a scaled RT/DS is currently bound

// Bound-target info for skip decisions (original dims, not the shrunk ones).
UINT          g_boundRTw  = 0, g_boundRTh  = 0;
UINT          g_boundDSVw = 0, g_boundDSVh = 0; // 0,0 = no depth bound
volatile LONG g_boundHasRTV = 0;

// Pixel-shader bytecode hashes (pass identification) + skip blocklist.
ID3D11PixelShader* g_curPS = nullptr; // last shader the game asked to bind
struct PsRec { ID3D11PixelShader* ps; ULONGLONG hash; };
PsRec     g_psMap[1024] = {};
ULONGLONG g_skipPs[64] = {};
UINT      g_skipPsN = 0;
ULONGLONG g_psLogged[256] = {};      // PassDump dedup set

// Constant buffers whose last full write looked orthographic (UI projection).
struct CbOrthoRec { ID3D11Buffer* buf; bool ortho; };
CbOrthoRec g_cbOrtho[64] = {};

struct PatchRec { ID3D11Resource* res; void* ptr; UINT bytes; };
PatchRec g_patchMaps[32] = {};

volatile LONGLONG g_frameTickSum = 0; // QPC ticks between presents (CPU frame-time stats)

// ---- Terminator Vision ----------------------------------------------------------
volatile LONG g_termBrackets  = 1; // TermHud && TermBrackets: collect mesh bounds
volatile LONG g_termLens      = 1; // TermLens
volatile LONG g_hudMaxTargets = 6; // HudMaxTargets
volatile LONG g_termOnnx      = 0; // last base layer came from ONNX (HUD status line)
volatile LONG g_termBaseOnnx  = 1; // TermBaseOnnx: 0 = full-res HLSL base (crisp), 1 = ONNX model
volatile LONG g_termBase      = 1; // TermBase: 0 = skip the red base restyle (normal frame)
bool          g_termFallbackLogged = false;

// Read back a buffer's contents via a staging copy on the immediate context.
// Returns a HeapFreeable buffer + fills 'bytes' — caller HeapFree's the return.
// Staging buffers are pooled (8 slots, round-robin, grown on demand) instead of
// create+release per call — ConsoleWire does 3 readbacks per submitted draw.
struct StagingSlot { ID3D11Buffer* buf; UINT size; };
StagingSlot g_staging[8] = {};
int         g_stagingNext = 0;

ID3D11Buffer* GetStaging(UINT bytes) noexcept
{
    for (auto& s : g_staging)
        if (s.buf != nullptr && s.size >= bytes)
            return s.buf;
    // grow-or-fill round-robin slot
    StagingSlot& slot = g_staging[g_stagingNext];
    g_stagingNext = (g_stagingNext + 1) & 7;
    if (slot.buf != nullptr && slot.size >= bytes)
        return slot.buf;
    if (slot.buf != nullptr)
        slot.buf->Release();
    slot.buf = nullptr;
    D3D11_BUFFER_DESC sd = {};
    sd.ByteWidth      = bytes;
    sd.Usage          = D3D11_USAGE_STAGING;
    sd.BindFlags      = 0;
    sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    if (g_device == nullptr || FAILED(g_device->CreateBuffer(&sd, nullptr, &slot.buf)) || slot.buf == nullptr)
        return nullptr;
    slot.size = bytes;
    return slot.buf;
}

void ReleaseStagingPool() noexcept
{
    for (auto& s : g_staging)
        if (s.buf != nullptr) { s.buf->Release(); s.buf = nullptr; s.size = 0; }
}

void* ReadbackBuffer(ID3D11DeviceContext* ctx, ID3D11Buffer* buf, UINT* bytes) noexcept
{
    if (ctx == nullptr || buf == nullptr || g_device == nullptr)
        return nullptr;
    D3D11_BUFFER_DESC bd = {};
    buf->GetDesc(&bd);
    ID3D11Buffer* staging = GetStaging(bd.ByteWidth);
    if (staging == nullptr)
        return nullptr;
    ctx->CopyResource(staging, buf);
    D3D11_MAPPED_SUBRESOURCE m = {};
    void* out = nullptr;
    if (SUCCEEDED(ctx->Map(staging, 0, D3D11_MAP_READ, 0, &m)) && m.pData != nullptr)
    {
        out = HeapAlloc(GetProcessHeap(), 0, bd.ByteWidth);
        if (out != nullptr)
        {
            memcpy(out, m.pData, bd.ByteWidth);
            *bytes = bd.ByteWidth;
        }
        ctx->Unmap(staging, 0);
    }
    return out;
}

// Returns remaining-dump allowance for this buffer (max 2 dumps each).
bool TakeDumpAllowance(ID3D11Buffer* buf) noexcept
{
    const size_t h = reinterpret_cast<UINT_PTR>(buf) & 127;
    if (g_dumped[h].buf == buf)
    {
        if (g_dumped[h].times >= 2)
            return false;
        ++g_dumped[h].times;
        return true;
    }
    g_dumped[h] = { buf, 1 };
    return true;
}

int KnownVSSlot(ID3D11Buffer* buf) noexcept
{
    const size_t h = reinterpret_cast<UINT_PTR>(buf) & 63;
    return (g_cbSlots[h].buf == buf) ? g_cbSlots[h].slot : -1;
}

// Check whether 16 floats starting at f[base] look like a rigid transform:
// first 3 basis vectors ~unit length and pairwise ~orthogonal (row or column
// layout — scan both). Allows w-column junk typical of affine matrices.
bool LooksLikeRotation(const float* f, int base) noexcept
{
    const float* m = f + base;
    // row-major: rows r0..r2 = m[0..3], m[4..7], m[8..11] using cols 0..2
    // col-major: cols c0..c2 = m[0],m[4],m[8] | m[1],m[5],m[9] | m[2],m[6],m[10]
    for (int layout = 0; layout < 2; ++layout)
    {
        float v[3][3];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                v[i][j] = (layout == 0) ? m[i * 4 + j] : m[j * 4 + i];
        bool unitOk = true;
        for (int i = 0; i < 3; ++i)
        {
            const float l = v[i][0]*v[i][0] + v[i][1]*v[i][1] + v[i][2]*v[i][2];
            if (l < 0.81f || l > 1.21f) { unitOk = false; break; } // |len|≈1 ±10%
        }
        if (!unitOk) continue;
        bool ortho = true;
        for (int i = 0; i < 3 && ortho; ++i)
            for (int j = i + 1; j < 3; ++j)
            {
                const float d = v[i][0]*v[j][0] + v[i][1]*v[j][1] + v[i][2]*v[j][2];
                if (d < -0.15f || d > 0.15f) { ortho = false; break; }
            }
        if (ortho)
            return true;
    }
    return false;
}

void DumpBufferFloats(ID3D11Buffer* buf, const void* data, UINT bytes, const char* how) noexcept
{
    const float* f = static_cast<const float*>(data);
    const int n = static_cast<int>(bytes / 4);
    const int slot = KnownVSSlot(buf);
    Log::Info("recon", "%s buf=%p bytes=%u vsSlot=%d n=%d", how, buf, bytes, slot, n);
    for (int i = 0; i < n && i < 40; i += 4)
        Log::Info("recon", "   [%3d] %9.4f %9.4f %9.4f %9.4f", i, f[i], f[i + 1], f[i + 2], f[i + 3]);
    // scan every 16-float block for rigid-transform candidates (view matrices)
    for (int b = 0; b + 16 <= n; b += 16)
        if (LooksLikeRotation(f, b))
            Log::Info("recon", "   ^ rot-matrix candidate at float offset %d (byte %d)", b, b * 4);
}

// vtable indices — fixed by the COM layout in d3d11.h/dxgi.h.
// ID3D11DeviceContext: IUnknown(0-2) + ID3D11DeviceChild(3-6) then methods.
// Order verified against SDK d3d11.h MIDL interface.
constexpr int kIdx_VSSetConstantBuffers        = 7;
constexpr int kIdx_PSSetShaderResources        = 8;
constexpr int kIdx_PSSetShader                 = 9;
constexpr int kIdx_DrawIndexed                 = 12;
constexpr int kIdx_Draw                        = 13;
constexpr int kIdx_Map                         = 14;
constexpr int kIdx_Unmap                       = 15;
constexpr int kIdx_IASetVertexBuffers          = 18;
constexpr int kIdx_IASetIndexBuffer            = 19;
constexpr int kIdx_DrawIndexedInstanced        = 20;
constexpr int kIdx_DrawInstanced               = 21;
constexpr int kIdx_DrawAuto                    = 38; // vtbl[24] is IASetPrimitiveTopology — verified against SDK d3d11.h
constexpr int kIdx_OMSetRenderTargets          = 33;
constexpr int kIdx_OMSetRTsAndUAVs             = 34;
constexpr int kIdx_RSSetState                  = 43;
constexpr int kIdx_RSSetViewports              = 44;
constexpr int kIdx_RSSetScissorRects           = 45;
constexpr int kIdx_CopySubresourceRegion       = 46;
constexpr int kIdx_UpdateSubresource           = 48;
// ID3D11Device: IUnknown(0-2) then CreateBuffer(3)... CreateInputLayout(11)
constexpr int kIdx_DevCreateBuffer             = 3;
constexpr int kIdx_DevCreateTexture2D          = 5;
constexpr int kIdx_DevCreateRenderTargetView   = 9;
constexpr int kIdx_DevCreateDepthStencilView   = 10;
constexpr int kIdx_DevCreateInputLayout        = 11;
constexpr int kIdx_DevCreatePixelShader        = 15;
// IDXGIFactory: IUnknown(0-2) + IDXGIObject(3-6) then EnumAdapters(7)...
constexpr int kIdx_FactoryCreateSwapChain      = 10;
// IDXGIFactory2 adds: IsWindowedStereoEnabled(14), CreateSwapChainForHwnd(15)
constexpr int kIdx_Factory2CreateSCForHwnd     = 15;
// IDXGISwapChain: IUnknown(0-2) + IDXGIObject(3-6) + IDXGIDeviceSubObject(7)
constexpr int kIdx_SwapChainPresent            = 8;
constexpr int kIdx_SwapChainGetBuffer          = 9;
constexpr int kIdx_SwapChainResizeBuffers      = 13;
constexpr int kIdx_SwapChain1Present1          = 22;
constexpr int kIdx_SwapChain3ResizeBuffers1    = 39;

// ---- original-call typedefs (all COM methods are __stdcall on x86) -------------
using FnDrawIndexed     = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, UINT, INT);
using FnDraw            = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, UINT);
using FnDrawIndexedInst = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, UINT, UINT, INT, UINT);
using FnDrawInstanced   = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, UINT, UINT, UINT);
using FnDrawAuto        = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*);
using FnMap             = HRESULT(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, ID3D11Resource*, UINT, D3D11_MAP, UINT, D3D11_MAPPED_SUBRESOURCE*);
using FnUnmap           = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, ID3D11Resource*, UINT);
using FnPresent         = HRESULT(STDMETHODCALLTYPE*)(IDXGISwapChain*, UINT, UINT);
using FnResizeBuffers   = HRESULT(STDMETHODCALLTYPE*)(IDXGISwapChain*, UINT, UINT, UINT, DXGI_FORMAT, UINT);
using FnGetBuffer       = HRESULT(STDMETHODCALLTYPE*)(IDXGISwapChain*, UINT, REFIID, void**);
using FnCreateSwapChain = HRESULT(STDMETHODCALLTYPE*)(IDXGIFactory*, IUnknown*, DXGI_SWAP_CHAIN_DESC*, IDXGISwapChain**);
using FnCreateSCForHwnd = HRESULT(STDMETHODCALLTYPE*)(IDXGIFactory2*, IUnknown*, HWND, const DXGI_SWAP_CHAIN_DESC1*, const DXGI_SWAP_CHAIN_FULLSCREEN_DESC*, IDXGIOutput*, IDXGISwapChain1**);
using FnPSSetShader     = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, ID3D11PixelShader*, ID3D11ClassInstance* const*, UINT);
using FnRSSetState      = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, ID3D11RasterizerState*);
using FnVSSetCBs        = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, UINT, ID3D11Buffer* const*);
using FnUpdateSubres    = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, ID3D11Resource*, UINT, const D3D11_BOX*, const void*, UINT, UINT);
using FnIASetVBs        = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, UINT, ID3D11Buffer* const*, const UINT*, const UINT*);
using FnIASetIB         = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, ID3D11Buffer*, DXGI_FORMAT, UINT);
using FnCreateInputLayout = HRESULT(STDMETHODCALLTYPE*)(ID3D11Device*, const D3D11_INPUT_ELEMENT_DESC*, UINT, const void*, SIZE_T, ID3D11InputLayout**);
using FnCreateTexture2D = HRESULT(STDMETHODCALLTYPE*)(ID3D11Device*, const D3D11_TEXTURE2D_DESC*, const D3D11_SUBRESOURCE_DATA*, ID3D11Texture2D**);
using FnCreateBuffer    = HRESULT(STDMETHODCALLTYPE*)(ID3D11Device*, const D3D11_BUFFER_DESC*, const D3D11_SUBRESOURCE_DATA*, ID3D11Buffer**);
using FnCreateRTV       = HRESULT(STDMETHODCALLTYPE*)(ID3D11Device*, ID3D11Resource*, const D3D11_RENDER_TARGET_VIEW_DESC*, ID3D11RenderTargetView**);
using FnCreateDSV       = HRESULT(STDMETHODCALLTYPE*)(ID3D11Device*, ID3D11Resource*, const D3D11_DEPTH_STENCIL_VIEW_DESC*, ID3D11DepthStencilView**);
using FnCreatePS        = HRESULT(STDMETHODCALLTYPE*)(ID3D11Device*, const void*, SIZE_T, ID3D11ClassLinkage*, ID3D11PixelShader**);
using FnOMSetRTs        = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, ID3D11RenderTargetView* const*, ID3D11DepthStencilView*);
using FnOMSetRTsUAVs    = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, ID3D11RenderTargetView* const*, ID3D11DepthStencilView*, UINT, UINT, ID3D11UnorderedAccessView* const*, const UINT*);
using FnRSSetViewports  = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, const D3D11_VIEWPORT*);
using FnRSSetScissor    = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, const D3D11_RECT*);
using FnCopySubres      = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, ID3D11Resource*, UINT, UINT, UINT, UINT, ID3D11Resource*, UINT, const D3D11_BOX*);
using FnPresent1        = HRESULT(STDMETHODCALLTYPE*)(IDXGISwapChain1*, UINT, UINT, const DXGI_PRESENT_PARAMETERS*);
using FnResizeBuffers1  = HRESULT(STDMETHODCALLTYPE*)(IDXGISwapChain3*, UINT, UINT, UINT, DXGI_FORMAT, UINT, const UINT*, IUnknown* const*);

FnDrawIndexed     o_DrawIndexed     = nullptr;
FnDraw            o_Draw            = nullptr;
FnDrawIndexedInst o_DrawIndexedInst = nullptr;
FnDrawInstanced   o_DrawInstanced   = nullptr;
FnDrawAuto        o_DrawAuto        = nullptr;
FnMap             o_Map             = nullptr;
FnUnmap           o_Unmap           = nullptr;
FnPresent         o_Present         = nullptr;
FnResizeBuffers   o_ResizeBuffers   = nullptr;
FnGetBuffer       o_GetBuffer       = nullptr;
FnCreateSwapChain o_CreateSwapChain = nullptr;
FnCreateSCForHwnd o_CreateSCForHwnd = nullptr;
FnPSSetShader     o_PSSetShader     = nullptr;
FnRSSetState      o_RSSetState      = nullptr;
FnVSSetCBs        o_VSSetCBs        = nullptr;
FnUpdateSubres    o_UpdateSubres    = nullptr;
FnIASetVBs        o_IASetVBs        = nullptr;
FnIASetIB         o_IASetIB         = nullptr;
FnCreateInputLayout o_CreateInputLayout = nullptr;
FnCreateTexture2D o_CreateTexture2D = nullptr;
FnCreateBuffer    o_CreateBuffer    = nullptr;
FnCreateRTV       o_CreateRTV       = nullptr;
FnCreateDSV       o_CreateDSV       = nullptr;
FnCreatePS        o_CreatePS        = nullptr;
FnOMSetRTs        o_OMSetRTs        = nullptr;
FnOMSetRTsUAVs    o_OMSetRTsUAVs    = nullptr;
FnRSSetViewports  o_RSSetViewports  = nullptr;
FnRSSetScissor    o_RSSetScissor    = nullptr;
FnCopySubres      o_CopySubres      = nullptr;
FnPresent1        o_Present1        = nullptr;
FnResizeBuffers1  o_ResizeBuffers1  = nullptr;

// Per-implementation originals. A process can hold several COM classes behind
// the same interface (debug-layer / 11on12 / differently-flagged devices, DXGI
// debug swap chains). Each distinct vtbl[idx] target gets its own trampoline;
// a single o_X global would be overwritten by the second class and route the
// first class's objects into the wrong implementation (AV inside d3d11/dxgi).
// o_X stays as the fallback (first hooked implementation).
struct TrampRec { const void* target; void* tramp; };
TrampRec      g_tramps[256] = {};
volatile LONG g_trampN = 0;
SRWLOCK       g_hookLock = SRWLOCK_INIT;

template <class Fn>
Fn Orig(Fn first, const void* obj, int idx) noexcept
{
    const void* target = (*reinterpret_cast<void* const* const*>(obj))[idx];
    const LONG n = g_trampN;
    for (LONG i = 0; i < n; ++i)
        if (g_tramps[i].target == target)
            return reinterpret_cast<Fn>(g_tramps[i].tramp);
    return first;
}

// Swap-chain methods are hooked by vtable-slot patching; the original lives in
// the patch record of the object's own vtable.
struct VtblSlot { void** slot; void* orig; void* detour; const char* name; };
VtblSlot      g_vtblPatches[48] = {};
volatile LONG g_vtblPatchN = 0;

template <class Fn>
Fn OrigV(Fn first, const void* obj, int idx) noexcept
{
    void* const* slot = &(*reinterpret_cast<void* const* const*>(obj))[idx];
    const LONG n = g_vtblPatchN;
    for (LONG i = 0; i < n; ++i)
        if (g_vtblPatches[i].slot == slot)
            return reinterpret_cast<Fn>(g_vtblPatches[i].orig);
    return first;
}

volatile LONG g_scaleLatched = 0; // RenderScale is fixed once a device exists
volatile LONG g_sc2Failed    = 0; // our present chain failed — stop retrying until resize/recreate

// ---- Phase 12 helpers ----------------------------------------------------------

int RenderScale() noexcept
{
    const LONG s = g_renderScale;
    return s > 1 ? static_cast<int>(s) : 1;
}

UINT ScreenDivisor(UINT w, UINT h) noexcept
{
    if (w == 0 || h == 0 || g_scW == 0 || g_scH == 0 || g_scW % w != 0)
        return 0;
    const UINT f = g_scW / w;
    if (f < 1 || f > 4)
        return 0;
    const UINT eh = h * f, tol = g_scH / 16;
    return (eh + tol < g_scH || eh > g_scH + tol) ? 0 : f;
}

bool IsScreenTarget(UINT w, UINT h) noexcept
{
    return ScreenDivisor(w, h) == 1;
}

constexpr size_t kScaledMask  = 1023;
constexpr size_t kScaledProbe = 64;
ID3D11Resource* const kScaledTomb = reinterpret_cast<ID3D11Resource*>(static_cast<UINT_PTR>(1));

size_t ScaledHome(const void* p) noexcept
{
    return (reinterpret_cast<UINT_PTR>(p) >> 4) & kScaledMask;
}

ScaledRec* FindScaledSlot(const ID3D11Resource* res) noexcept
{
    if (res == nullptr)
        return nullptr;
    const size_t h = ScaledHome(res);
    for (size_t i = 0; i < kScaledProbe; ++i)
    {
        ScaledRec& r = g_scaledRes[(h + i) & kScaledMask];
        if (r.res == res)
            return &r;
        if (r.res == nullptr)
            return nullptr;
    }
    return nullptr;
}

const ScaledRec* FindScaled(const ID3D11Resource* res) noexcept
{
    return FindScaledSlot(res);
}

void SetScaled(ID3D11Resource* res, UINT w, UINT h) noexcept
{
    if (res == nullptr)
        return;
    ScaledRec* r = FindScaledSlot(res);
    if (r == nullptr)
    {
        const size_t home = ScaledHome(res);
        for (size_t i = 0; i < kScaledProbe && r == nullptr; ++i)
        {
            ScaledRec& c = g_scaledRes[(home + i) & kScaledMask];
            if (c.res == nullptr || c.res == kScaledTomb)
                r = &c;
        }
        if (r == nullptr)
            r = &g_scaledRes[home];
    }
    *r = { res, w, h };
}

void UnsetScaled(const ID3D11Resource* res) noexcept
{
    ScaledRec* r = FindScaledSlot(res);
    if (r != nullptr)
        *r = { kScaledTomb, 0, 0 };
}

// Get a texture's real (allocated) dims — the scaled value if we shrank it.
void ResourceDims(ID3D11Resource* res, UINT* w, UINT* h, bool* scaled) noexcept
{
    *w = *h = 0; *scaled = false;
    const ScaledRec* s = FindScaled(res);
    if (s != nullptr)
    {
        *w = s->w; *h = s->h; *scaled = true;
        return;
    }
    D3D11_RESOURCE_DIMENSION dim = D3D11_RESOURCE_DIMENSION_UNKNOWN;
    res->GetType(&dim);
    if (dim == D3D11_RESOURCE_DIMENSION_TEXTURE2D)
    {
        D3D11_TEXTURE2D_DESC td = {};
        static_cast<ID3D11Texture2D*>(res)->GetDesc(&td);
        *w = td.Width; *h = td.Height;
    }
}

bool ViewDims(ID3D11View* v, UINT* w, UINT* h, bool* scaled) noexcept
{
    *w = *h = 0; *scaled = false;
    if (v == nullptr)
        return false;
    ID3D11Resource* r = nullptr;
    v->GetResource(&r);
    if (r == nullptr)
        return false;
    ResourceDims(r, w, h, scaled);
    r->Release();
    return true;
}

void ClearScalingTables() noexcept
{
    memset(g_scaledRes, 0, sizeof(g_scaledRes));
    g_vpN = g_srN = 0;
    g_scaledBound = 0;
    g_boundRTw = g_boundRTh = g_boundDSVw = g_boundDSVh = 0;
    g_boundHasRTV = 0;
}

// Tag the swap-chain backbuffer as a scaled resource: it is REALLY allocated at
// g_scW/s while the game believes it is g_scW. Views created on it then report
// believed dims + scaled=1, so the final composite pass (and its texel-space
// Load()s into shrunken sources) stays consistent — this is what makes scaled
// rendering work on engines whose post chain uses texel fetch (Unigine/Heaven).
void RegisterScaledBackbuffer(IDXGISwapChain* sc) noexcept
{
    const int s = RenderScale();
    if (sc == nullptr || g_scW == 0 || g_scH == 0 || s <= 1 || g_scIsFlip != 0)
        return;
    ID3D11Texture2D* bb = nullptr;
    if (FAILED(sc->GetBuffer(0, __uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&bb))) || bb == nullptr)
        return;
    D3D11_TEXTURE2D_DESC td = {};
    bb->GetDesc(&td);
    if (td.Width == g_scW / static_cast<UINT>(s) && td.Height == g_scH / static_cast<UINT>(s))
    {
        SetScaled(bb, g_scW, g_scH);
        Log::Info("scale", "backbuffer real %ux%u, believed %ux%u", td.Width, td.Height, g_scW, g_scH);
    }
    bb->Release();
}

void ReleaseProxyBackbuffer() noexcept
{
    if (g_proxyBB == nullptr)
        return;
    UnsetScaled(g_proxyBB);
    g_proxyBB->Release();
    g_proxyBB = nullptr;
    g_proxySC = nullptr;
}

ID3D11Texture2D* EnsureProxyBackbuffer(IDXGISwapChain* sc) noexcept
{
    if (g_proxyBB != nullptr)
        return g_proxySC == sc ? g_proxyBB : nullptr;
    const int s = RenderScale();
    if (sc == nullptr || s <= 1 || g_scIsFlip == 0 || g_device == nullptr || o_GetBuffer == nullptr)
        return nullptr;
    ID3D11Texture2D* bb = nullptr;
    if (FAILED(OrigV(o_GetBuffer, sc, kIdx_SwapChainGetBuffer)(sc, 0, __uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&bb))) || bb == nullptr)
        return nullptr;
    D3D11_TEXTURE2D_DESC td = {};
    bb->GetDesc(&td);
    bb->Release();
    D3D11_TEXTURE2D_DESC pd = td;
    pd.Width  = td.Width  / static_cast<UINT>(s); if (pd.Width == 0)  pd.Width = 1;
    pd.Height = td.Height / static_cast<UINT>(s); if (pd.Height == 0) pd.Height = 1;
    pd.MipLevels = 1;
    pd.ArraySize = 1;
    pd.SampleDesc.Count = 1;
    pd.SampleDesc.Quality = 0;
    pd.Usage = D3D11_USAGE_DEFAULT;
    pd.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    pd.CPUAccessFlags = 0;
    pd.MiscFlags = 0;
    InterlockedExchange(&g_inSCCreate, 1);
    const HRESULT hr = g_device->CreateTexture2D(&pd, nullptr, &g_proxyBB);
    InterlockedExchange(&g_inSCCreate, 0);
    if (FAILED(hr) || g_proxyBB == nullptr)
    {
        g_proxyBB = nullptr;
        Log::Warn("scale", "proxy backbuffer %ux%u fmt=%u create failed 0x%08lX",
                  pd.Width, pd.Height, static_cast<UINT>(pd.Format), static_cast<unsigned long>(hr));
        return nullptr;
    }
    g_proxySC  = sc;
    g_proxyFmt = pd.Format;
    if (g_scW != td.Width || g_scH != td.Height)
    {
        g_scW = td.Width;
        g_scH = td.Height;
    }
    SetScaled(g_proxyBB, td.Width, td.Height);
    Log::Info("scale", "flip chain: game renders into proxy %ux%u (native %ux%u, RenderScale=%d)",
              pd.Width, pd.Height, td.Width, td.Height, s);
    return g_proxyBB;
}

void PatchScreenFloats(void* data, UINT bytes) noexcept
{
    const int s = RenderScale();
    if (data == nullptr || s <= 1 || g_scW == 0 || g_scH == 0)
        return;
    float* f = static_cast<float*>(data);
    const UINT n = bytes / 4;
    const float w = static_cast<float>(g_scW), h = static_cast<float>(g_scH);
    const float fs = static_cast<float>(s);
    for (UINT i = 0; i + 1 < n; ++i)
    {
        if (f[i] == w && f[i + 1] == h)
        {
            f[i] = w / fs; f[i + 1] = h / fs; ++i;
        }
        else if (f[i] > 0.0f && f[i] < 0.01f &&
                 fabsf(f[i] * w - 1.0f) < 1e-3f && fabsf(f[i + 1] * h - 1.0f) < 1e-3f)
        {
            f[i] *= fs; f[i + 1] *= fs; ++i;
        }
    }
}

UINT PatchableCBBytes(ID3D11Resource* res) noexcept
{
    if (res == nullptr)
        return 0;
    D3D11_RESOURCE_DIMENSION dim = D3D11_RESOURCE_DIMENSION_UNKNOWN;
    res->GetType(&dim);
    if (dim != D3D11_RESOURCE_DIMENSION_BUFFER)
        return 0;
    D3D11_BUFFER_DESC bd = {};
    static_cast<ID3D11Buffer*>(res)->GetDesc(&bd);
    if ((bd.BindFlags & D3D11_BIND_CONSTANT_BUFFER) == 0 || bd.ByteWidth > 4096)
        return 0;
    return bd.ByteWidth;
}

ULONGLONG HashBytecode(const void* p, SIZE_T n) noexcept
{
    const unsigned char* b = static_cast<const unsigned char*>(p);
    ULONGLONG h = 1469598103934665603ull; // FNV-1a 64
    for (SIZE_T i = 0; i < n; ++i)
        h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

ULONGLONG PsHashOf(ID3D11PixelShader* ps) noexcept
{
    const size_t h = (reinterpret_cast<UINT_PTR>(ps) >> 4) & 1023;
    return (g_psMap[h].ps == ps) ? g_psMap[h].hash : 0;
}

// Row-major D3D ortho projection: m00=2/w and m11=±2/h are tiny non-zero,
// off-diagonal 3x3 terms ~0, m33/m44 ~= 1. Good enough to mark HUD/UI CBs.
bool LooksOrtho(const void* data, UINT bytes) noexcept
{
    if (bytes < 64)
        return false;
    const float* m = static_cast<const float*>(data);
    const float m00 = fabsf(m[0]), m11 = fabsf(m[5]);
    return m00 > 1e-6f && m00 < 0.05f && m11 > 1e-6f && m11 < 0.05f &&
           fabsf(m[1]) < 1e-4f && fabsf(m[2]) < 1e-4f &&
           fabsf(m[4]) < 1e-4f && fabsf(m[6]) < 1e-4f &&
           fabsf(m[15] - 1.0f) < 0.01f;
}

// Decide whether this draw can be dropped entirely. Only active in retro modes
// (mode != Off) and never on our own fullscreen pass (g_inRetroPass).
bool ShouldSkipDraw(UINT indexCount) noexcept
{
    if (g_inRetroPass != 0 || g_mode == static_cast<LONG>(hooks::RetroMode::Off))
        return false;
    // Never skip draws into a screen-sized target (bound dims track the game's
    // belief, so this covers the unscaled backbuffer AND scaled screen-sized
    // RTs). The final composite/HDR-to-LDR blit is a fullscreen triangle (≤6
    // indices) — skipping it leaves the backbuffer black.
    if (g_boundHasRTV != 0 && IsScreenTarget(g_boundRTw, g_boundRTh))
        return false;
    if (g_skipMaxIndex > 0 && indexCount > 0 && indexCount <= static_cast<UINT>(g_skipMaxIndex))
        return true;
    // depth-only pass onto a non-screen-sized target = shadow/cascade map
    if (g_skipShadows && g_boundHasRTV == 0 &&
        g_boundDSVw != 0 && !IsScreenTarget(g_boundDSVw, g_boundDSVh))
        return true;
    if (g_skipOrtho && g_vsCB0 != nullptr)
    {
        const size_t h = reinterpret_cast<UINT_PTR>(g_vsCB0) & 63;
        if (g_cbOrtho[h].buf == g_vsCB0 && g_cbOrtho[h].ortho)
            return true;
    }
    if (g_skipPsN > 0 && g_curPS != nullptr)
    {
        const ULONGLONG hh = PsHashOf(g_curPS);
        if (hh != 0)
            for (UINT i = 0; i < g_skipPsN; ++i)
                if (g_skipPs[i] == hh)
                    return true;
    }
    return false;
}

// First-encounter log of "which PS draws into which target" — for building SkipPS.
void PassDumpNoteDraw() noexcept
{
    if (g_passDump <= 0 || g_curPS == nullptr)
        return;
    const ULONGLONG h = PsHashOf(g_curPS);
    if (h == 0)
        return;
    const size_t s = static_cast<size_t>(h >> 8) & 255;
    if (g_psLogged[s] == h)
        return;
    g_psLogged[s] = h;
    Log::Info("pass", "ps=%016llx rt=%ux%u dsv=%ux%u scaled=%d",
              h, g_boundRTw, g_boundRTh, g_boundDSVw, g_boundDSVh,
              static_cast<int>(g_scaledBound));
}

// Re-push the game's stored viewport/scissor at the scale matching the bound RTs.
void PushViewports(ID3D11DeviceContext* ctx) noexcept
{
    if (o_RSSetViewports == nullptr || g_vpN == 0)
        return;
    const int s = RenderScale();
    if (g_scaledBound && s > 1)
    {
        D3D11_VIEWPORT out[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE];
        for (UINT i = 0; i < g_vpN; ++i)
        {
            out[i] = g_vp[i];
            out[i].TopLeftX /= s; out[i].TopLeftY /= s;
            out[i].Width    /= s; out[i].Height    /= s;
        }
        Orig(o_RSSetViewports, ctx, kIdx_RSSetViewports)(ctx, g_vpN, out);
        return;
    }
    Orig(o_RSSetViewports, ctx, kIdx_RSSetViewports)(ctx, g_vpN, g_vp);
}

void PushScissors(ID3D11DeviceContext* ctx) noexcept
{
    if (o_RSSetScissor == nullptr || g_srN == 0)
        return;
    const int s = RenderScale();
    if (g_scaledBound && s > 1)
    {
        D3D11_RECT out[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE];
        for (UINT i = 0; i < g_srN; ++i)
        {
            out[i].left   = g_sr[i].left   / s;
            out[i].top    = g_sr[i].top    / s;
            out[i].right  = g_sr[i].right  / s;
            out[i].bottom = g_sr[i].bottom / s;
        }
        Orig(o_RSSetScissor, ctx, kIdx_RSSetScissorRects)(ctx, g_srN, out);
        return;
    }
    Orig(o_RSSetScissor, ctx, kIdx_RSSetScissorRects)(ctx, g_srN, g_sr);
}

// Inspect an RTV/DSV set and update the bound/scaled tracking.
void NoteBoundTargets(ID3D11DeviceContext* ctx, UINT n,
                      ID3D11RenderTargetView* const* rtvs, ID3D11DepthStencilView* dsv) noexcept
{
    LONG scaled = 0, hasRTV = 0;
    g_boundRTw = g_boundRTh = g_boundDSVw = g_boundDSVh = 0;
    if (rtvs != nullptr)
        for (UINT i = 0; i < n && i < D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT; ++i)
        {
            if (rtvs[i] == nullptr)
                continue;
            hasRTV = 1;
            UINT vw = 0, vh = 0; bool vs = false;
            if (ViewDims(rtvs[i], &vw, &vh, &vs))
            {
                if (i == 0) { g_boundRTw = vw; g_boundRTh = vh; }
                if (vs) scaled = 1;
            }
        }
    if (dsv != nullptr)
    {
        UINT vw = 0, vh = 0; bool vs = false;
        if (ViewDims(dsv, &vw, &vh, &vs))
        {
            g_boundDSVw = vw; g_boundDSVh = vh;
            if (vs) scaled = 1;
        }
    }
    g_boundHasRTV = hasRTV;
    InterlockedExchange(&g_scaledBound, scaled);
    // The game may have set its viewport before binding these targets; re-apply at
    // the correct scale. Without RenderScale there is nothing to correct, and
    // re-pushing globally-tracked state would clobber viewports on other
    // (deferred) contexts recorded by worker threads.
    if (RenderScale() > 1)
    {
        PushViewports(ctx);
        PushScissors(ctx);
    }
}

// Terminator brackets: project this scene draw's mesh bounds (immediate context only).
void NoteTargetDraw(ID3D11DeviceContext* ctx) noexcept
{
    if (g_mode != static_cast<LONG>(hooks::RetroMode::Terminator) || g_termBrackets == 0 ||
        g_inRetroPass != 0 || ctx != g_ctx || g_boundHasRTV == 0 ||
        !IsScreenTarget(g_boundRTw, g_boundRTh))
        return;
    targets::OnDraw(g_vb0.buf, g_vb0.stride, g_vb0.offset, g_vsCB0);
}

// ---- detours ------------------------------------------------------------------
void STDMETHODCALLTYPE hkDrawIndexed(ID3D11DeviceContext* ctx, UINT idx, UINT start, INT base)
{
    InterlockedIncrement64(&g_draws);
    if (g_dumpMesh > 0 && idx > 12 && g_vb0.buf != nullptr && g_ib.buf != nullptr &&
        InterlockedDecrement(&g_dumpMesh) >= 0)
    {
        // snapshot geometry + slot-0 CB so the VP matrix can be verified offline
        UINT vbBytes = 0, ibBytes = 0, cbBytes = 0;
        void* vb = ReadbackBuffer(ctx, g_vb0.buf, &vbBytes);
        void* ib = ReadbackBuffer(ctx, g_ib.buf, &ibBytes);
        void* cb = (g_vsCB0 != nullptr) ? ReadbackBuffer(ctx, g_vsCB0, &cbBytes) : nullptr;
        Log::Info("mesh", "draw idx=%u start=%u base=%d vb=%p(%uB stride=%u) ib=%p(%uB fmt=%u) cb0=%p(%uB)",
                  idx, start, base, g_vb0.buf, vbBytes, g_vb0.stride,
                  g_ib.buf, ibBytes, g_ib.fmt, g_vsCB0, cbBytes);
        if (vb != nullptr)
        {
            const float* v = static_cast<const float*>(vb);
            const int floatsPerVert = static_cast<int>(g_vb0.stride / 4);
            for (int i = 0; i < 4 && i * floatsPerVert + 2 < static_cast<int>(vbBytes / 4); ++i)
                Log::Info("mesh", "   v%d: %9.3f %9.3f %9.3f", i,
                          v[i * floatsPerVert], v[i * floatsPerVert + 1], v[i * floatsPerVert + 2]);
        }
        if (ib != nullptr)
        {
            if (g_ib.fmt == DXGI_FORMAT_R16_UINT)
            {
                const unsigned short* ix = static_cast<const unsigned short*>(ib);
                Log::Info("mesh", "   idx16: %u %u %u %u %u %u", ix[0], ix[1], ix[2], ix[3], ix[4], ix[5]);
            }
            else
            {
                const UINT* ix = static_cast<const UINT*>(ib);
                Log::Info("mesh", "   idx32: %u %u %u %u %u %u", ix[0], ix[1], ix[2], ix[3], ix[4], ix[5]);
            }
        }
        if (cb != nullptr)
        {
            const float* c = static_cast<const float*>(cb);
            for (int i = 0; i < 32 && i + 3 < static_cast<int>(cbBytes / 4); i += 4)
                Log::Info("mesh", "   cb[%2d] %9.4f %9.4f %9.4f %9.4f", i, c[i], c[i+1], c[i+2], c[i+3]);
        }
        if (vb != nullptr) HeapFree(GetProcessHeap(), 0, vb);
        if (ib != nullptr) HeapFree(GetProcessHeap(), 0, ib);
        if (cb != nullptr) HeapFree(GetProcessHeap(), 0, cb);
    }
    if (g_mode == static_cast<int>(hooks::RetroMode::ConsoleWire) &&
        idx > 12 && wire::WantsMore() &&
        g_vb0.buf != nullptr && g_ib.buf != nullptr && g_vb0.stride >= 12)
    {
        UINT vbBytes = 0, ibBytes = 0, wbBytes = 0;
        void* vb = ReadbackBuffer(ctx, g_vb0.buf, &vbBytes);
        void* ib = ReadbackBuffer(ctx, g_ib.buf, &ibBytes);
        // per-draw world matrix lives in the 64B CB at VS slot 0
        void* wb = nullptr;
        if (g_vsCB0 != nullptr)
        {
            D3D11_BUFFER_DESC wd = {};
            g_vsCB0->GetDesc(&wd);
            if (wd.ByteWidth == 64)
                wb = ReadbackBuffer(ctx, g_vsCB0, &wbBytes);
        }
        if (vb != nullptr && ib != nullptr)
            wire::Submit(vb, vbBytes, g_vb0.stride, ib, start, idx, base, g_ib.fmt,
                         (wb != nullptr && wbBytes >= 64) ? static_cast<const float*>(wb) : nullptr);
        if (vb != nullptr) HeapFree(GetProcessHeap(), 0, vb);
        if (ib != nullptr) HeapFree(GetProcessHeap(), 0, ib);
        if (wb != nullptr) HeapFree(GetProcessHeap(), 0, wb);
    }
    if (ShouldSkipDraw(idx))
    {
        InterlockedIncrement64(&g_skips);
        return;
    }
    PassDumpNoteDraw();
    NoteTargetDraw(ctx);
    Orig(o_DrawIndexed, ctx, kIdx_DrawIndexed)(ctx, idx, start, base);
}

void STDMETHODCALLTYPE hkDraw(ID3D11DeviceContext* ctx, UINT count, UINT start)
{
    InterlockedIncrement64(&g_draws);
    if (ShouldSkipDraw(0))
    {
        InterlockedIncrement64(&g_skips);
        return;
    }
    PassDumpNoteDraw();
    Orig(o_Draw, ctx, kIdx_Draw)(ctx, count, start);
}

void STDMETHODCALLTYPE hkDrawIndexedInstanced(ID3D11DeviceContext* ctx, UINT iCount, UINT inst, UINT start, INT base, UINT startInst)
{
    InterlockedIncrement64(&g_draws);
    if (ShouldSkipDraw(iCount))
    {
        InterlockedIncrement64(&g_skips);
        return;
    }
    PassDumpNoteDraw();
    NoteTargetDraw(ctx);
    Orig(o_DrawIndexedInst, ctx, kIdx_DrawIndexedInstanced)(ctx, iCount, inst, start, base, startInst);
}

void STDMETHODCALLTYPE hkDrawInstanced(ID3D11DeviceContext* ctx, UINT vCount, UINT inst, UINT startV, UINT startI)
{
    InterlockedIncrement64(&g_draws);
    if (ShouldSkipDraw(0))
    {
        InterlockedIncrement64(&g_skips);
        return;
    }
    PassDumpNoteDraw();
    Orig(o_DrawInstanced, ctx, kIdx_DrawInstanced)(ctx, vCount, inst, startV, startI);
}

void STDMETHODCALLTYPE hkDrawAuto(ID3D11DeviceContext* ctx)
{
    InterlockedIncrement64(&g_draws);
    if (ShouldSkipDraw(0))
    {
        InterlockedIncrement64(&g_skips);
        return;
    }
    PassDumpNoteDraw();
    Orig(o_DrawAuto, ctx, kIdx_DrawAuto)(ctx);
}

void STDMETHODCALLTYPE hkVSSetConstantBuffers(ID3D11DeviceContext* ctx, UINT start,
                                              UINT num, ID3D11Buffer* const* cbs)
{
    if (cbs != nullptr)
        for (UINT i = 0; i < num && i < 14; ++i)
            if (cbs[i] != nullptr)
            {
                const size_t h = reinterpret_cast<UINT_PTR>(cbs[i]) & 63;
                g_cbSlots[h] = { cbs[i], static_cast<int>(start + i) };
                if (start + i == 0)
                {
                    g_vsCB0 = cbs[i];
                    // track the 144-byte proj CB for console-wireframe projection
                    if (cbs[i] != g_projCB)
                    {
                        D3D11_BUFFER_DESC pd = {};
                        cbs[i]->GetDesc(&pd);
                        if (pd.ByteWidth == 144)
                            g_projCB = cbs[i];
                    }
                }
            }
    Orig(o_VSSetCBs, ctx, kIdx_VSSetConstantBuffers)(ctx, start, num, cbs);
}

void STDMETHODCALLTYPE hkIASetVertexBuffers(ID3D11DeviceContext* ctx, UINT start, UINT num,
                                            ID3D11Buffer* const* vbs, const UINT* strides,
                                            const UINT* offsets)
{
    if (vbs != nullptr && start == 0 && num > 0 && vbs[0] != nullptr)
        g_vb0 = { vbs[0], strides != nullptr ? strides[0] : 0,
                  offsets != nullptr ? offsets[0] : 0 };
    Orig(o_IASetVBs, ctx, kIdx_IASetVertexBuffers)(ctx, start, num, vbs, strides, offsets);
}

void STDMETHODCALLTYPE hkIASetIndexBuffer(ID3D11DeviceContext* ctx, ID3D11Buffer* ib,
                                          DXGI_FORMAT fmt, UINT offset)
{
    g_ib = { ib, fmt, offset };
    Orig(o_IASetIB, ctx, kIdx_IASetIndexBuffer)(ctx, ib, fmt, offset);
}

HRESULT STDMETHODCALLTYPE hkMap(ID3D11DeviceContext* ctx, ID3D11Resource* res, UINT sub, D3D11_MAP type, UINT flags, D3D11_MAPPED_SUBRESOURCE* out)
{
    InterlockedIncrement64(&g_maps);
    const HRESULT hr = Orig(o_Map, ctx, kIdx_Map)(ctx, res, sub, type, flags, out);
    if (g_dumpCBs > 0 && SUCCEEDED(hr) && out != nullptr && res != nullptr)
    {
        const size_t h = reinterpret_cast<UINT_PTR>(res) & 31;
        g_mapRecs[h] = { res, out->pData };
    }
    if (g_patchScreenCB != 0 && type == D3D11_MAP_WRITE_DISCARD && SUCCEEDED(hr) &&
        out != nullptr && out->pData != nullptr && RenderScale() > 1)
    {
        const UINT bytes = PatchableCBBytes(res);
        if (bytes > 0)
            g_patchMaps[(reinterpret_cast<UINT_PTR>(res) >> 4) & 31] = { res, out->pData, bytes };
    }
    return hr;
}

void STDMETHODCALLTYPE hkUnmap(ID3D11DeviceContext* ctx, ID3D11Resource* res, UINT sub)
{
    if (g_dumpCBs > 0 && res != nullptr)
    {
        const size_t h = reinterpret_cast<UINT_PTR>(res) & 31;
        if (g_mapRecs[h].res == res && g_mapRecs[h].ptr != nullptr)
        {
            ID3D11Buffer* buf = nullptr;
            if (SUCCEEDED(res->QueryInterface(__uuidof(ID3D11Buffer), reinterpret_cast<void**>(&buf))) && buf != nullptr)
            {
                D3D11_BUFFER_DESC bd = {};
                buf->GetDesc(&bd);
                if ((bd.BindFlags & D3D11_BIND_CONSTANT_BUFFER) != 0 &&
                    bd.ByteWidth >= 64 && bd.ByteWidth <= 2048 &&
                    InterlockedDecrement(&g_dumpCBs) >= 0 && TakeDumpAllowance(buf))
                {
                    DumpBufferFloats(buf, g_mapRecs[h].ptr, bd.ByteWidth, "unmap");
                }
                buf->Release();
            }
            g_mapRecs[h].ptr = nullptr;
        }
    }
    if (g_patchScreenCB != 0 && res != nullptr)
    {
        PatchRec& pr = g_patchMaps[(reinterpret_cast<UINT_PTR>(res) >> 4) & 31];
        if (pr.res == res && pr.ptr != nullptr)
        {
            PatchScreenFloats(pr.ptr, pr.bytes);
            pr = {};
        }
    }
    Orig(o_Unmap, ctx, kIdx_Unmap)(ctx, res, sub);
}

void STDMETHODCALLTYPE hkUpdateSubresource(ID3D11DeviceContext* ctx, ID3D11Resource* res,
                                           UINT sub, const D3D11_BOX* box,
                                           const void* src, UINT rowPitch, UINT depthPitch)
{
    if (g_dumpCBs > 0 && res != nullptr && src != nullptr)
    {
        ID3D11Buffer* buf = nullptr;
        if (SUCCEEDED(res->QueryInterface(__uuidof(ID3D11Buffer), reinterpret_cast<void**>(&buf))) && buf != nullptr)
        {
            D3D11_BUFFER_DESC bd = {};
            buf->GetDesc(&bd);
            const bool cb = (bd.BindFlags & D3D11_BIND_CONSTANT_BUFFER) != 0;
            if (cb && bd.ByteWidth >= 64 && bd.ByteWidth <= 2048 &&
                InterlockedDecrement(&g_dumpCBs) >= 0 && TakeDumpAllowance(buf))
            {
                DumpBufferFloats(buf, src, bd.ByteWidth, "updsubres");
            }
            buf->Release();
        }
    }
    if (g_mode == static_cast<LONG>(hooks::RetroMode::Terminator) && g_termBrackets != 0 &&
        g_inRetroPass == 0 && ctx == g_ctx && box == nullptr && src != nullptr)
    {
        const UINT bytes = PatchableCBBytes(res);
        if (bytes == 64 || bytes == 144)
            targets::OnUpdateCB(static_cast<ID3D11Buffer*>(res), bytes, src);
    }
    // never rewrite our own pass constants (their dims can equal the screen size)
    if (g_patchScreenCB != 0 && g_inRetroPass == 0 && box == nullptr && src != nullptr && RenderScale() > 1)
    {
        const UINT bytes = PatchableCBBytes(res);
        if (bytes > 0)
        {
            float tmp[1024];
            memcpy(tmp, src, bytes);
            PatchScreenFloats(tmp, bytes);
            Orig(o_UpdateSubres, ctx, kIdx_UpdateSubresource)(ctx, res, sub, box, tmp, rowPitch, depthPitch);
            return;
        }
    }
    Orig(o_UpdateSubres, ctx, kIdx_UpdateSubresource)(ctx, res, sub, box, src, rowPitch, depthPitch);
}

HRESULT STDMETHODCALLTYPE hkResizeBuffers(IDXGISwapChain* sc, UINT count, UINT w, UINT h,
                                          DXGI_FORMAT fmt, UINT flags)
{
    retro::ReleaseBackbuffer(); // outstanding bb refs make ResizeBuffers fail
    ai::ReleaseBackbuffer();
    det::ReleaseBackbuffer();
    const FnResizeBuffers orig = OrigV(o_ResizeBuffers, sc, kIdx_SwapChainResizeBuffers);
    const int s = RenderScale();
    if (g_inSCCreate != 0)                    // our own shrink call — pass through as-is
    {
        return orig(sc, count, w, h, fmt, flags);
    }
    retro::DropUpscaleChain();
    ReleaseProxyBackbuffer();
    InterlockedExchange(&g_sc2Failed, 0);     // window/mode changed — our chain may work now
    if (w > 0 && h > 0 && (w != g_scW || h != g_scH))
    {
        const bool minor = w == g_scW && h + g_scH / 16 >= g_scH && h <= g_scH + g_scH / 16;
        g_scW = w; g_scH = h;
        if (!minor)
            ClearScalingTables(); // stale scaled-res/view entries are invalid now
        if (g_sc2 != nullptr) { g_sc2->Release(); g_sc2 = nullptr; }
        retro::DropUpscaleChain();
        Log::Info("scale", "screen dims -> %ux%u", w, h);
    }
    UINT rw = w, rh = h;
    if (s > 1 && g_scIsFlip == 0 && w > 0 && h > 0)  // keep the real backbuffer at believed/s
    {
        rw = w / static_cast<UINT>(s);
        rh = h / static_cast<UINT>(s);
        if (rw == 0) rw = 1;
        if (rh == 0) rh = 1;
    }
    InterlockedExchange(&g_inSCCreate, 1);    // DXGI-internal bb create must not reshrink
    HRESULT hr = orig(sc, count, rw, rh, fmt, flags);
    if (FAILED(hr) && (rw != w || rh != h))
    {
        // Flip-model chains (CreateSwapChainForHwnd) can reject the shrunken size;
        // retry at the game's requested size rather than fail the game outright.
        Log::Warn("hooks", "ResizeBuffers(%ux%u) failed 0x%08lX - retrying native",
                  rw, rh, static_cast<unsigned long>(hr));
        hr = orig(sc, count, w, h, fmt, flags);
    }
    InterlockedExchange(&g_inSCCreate, 0);
    if (SUCCEEDED(hr) && g_scIsFlip != 0 && (w == 0 || h == 0))
    {
        DXGI_SWAP_CHAIN_DESC rd = {};
        if (SUCCEEDED(sc->GetDesc(&rd)) && rd.BufferDesc.Width > 0 &&
            (rd.BufferDesc.Width != g_scW || rd.BufferDesc.Height != g_scH))
        {
            const UINT nh = rd.BufferDesc.Height;
            const bool minor = rd.BufferDesc.Width == g_scW && nh + g_scH / 16 >= g_scH &&
                               nh <= g_scH + g_scH / 16;
            g_scW = rd.BufferDesc.Width; g_scH = nh;
            if (!minor)
                ClearScalingTables();
            Log::Info("scale", "screen dims -> %ux%u (window size)", g_scW, g_scH);
        }
    }
    if (SUCCEEDED(hr) && s > 1)
        RegisterScaledBackbuffer(sc);
    Log::Info("hooks", "ResizeBuffers(%u, %ux%u real %ux%u, fmt=%u) -> 0x%08lX", count, w, h,
              rw, rh, static_cast<UINT>(fmt), static_cast<unsigned long>(hr));
    return hr;
}

bool EnsurePresentChain(IDXGISwapChain* sc) noexcept;  // fwd — defined below

// Terminator layer 1: ONNX (ORT DirectML) on sc's backbuffer, HLSL fallback otherwise.
void TerminatorBase(ID3D11DeviceContext* ctx, ID3D11Device* dev, IDXGISwapChain* sc) noexcept
{
    InterlockedExchange(&g_inRetroPass, 1);
    retro::Ensure(dev);
    // Detector samples the RAW frame — must run before any restyle touches the backbuffer.
    if (det::Ensure(dev))
        det::Run(ctx, sc);
    if (g_termBase == 0)
    {
        InterlockedExchange(&g_termOnnx, 0);
        InterlockedExchange(&g_inRetroPass, 0);
        return;
    }
    if (g_termBaseOnnx != 0 && ai::Ensure(dev))
    {
        ai::Apply(ctx, sc);
        InterlockedExchange(&g_termOnnx, 1);
    }
    else
    {
        if (!g_termFallbackLogged)
        {
            g_termFallbackLogged = true;
            Log::Warn("term", "ONNX path unavailable — Terminator base layer using HLSL fallback");
        }
        InterlockedExchange(&g_termOnnx, 0);
        retro::Apply(ctx, sc, retro::Fx::TermBase);
    }
    InterlockedExchange(&g_inRetroPass, 0);
}

// Terminator layers 2+3: HUD then lens, on the chain that actually presents (native res).
void TerminatorOverlay(ID3D11DeviceContext* ctx, IDXGISwapChain* target) noexcept
{
    static LARGE_INTEGER freq = {}, prev = {};
    LARGE_INTEGER now = {};
    if (freq.QuadPart == 0)
        QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&now);
    const float dt = prev.QuadPart != 0
                   ? static_cast<float>(static_cast<double>(now.QuadPart - prev.QuadPart) / static_cast<double>(freq.QuadPart))
                   : 0.0f;
    prev = now;
    int nTracks = 0;
    const term::Track* tracks = nullptr;
    const char* const* labels = nullptr;
    const hud::Line* lines = nullptr;
    int nLines = 0;
    const bool detOn = det::Active();
    if (detOn)
    {
        tracks = det::Update(dt, static_cast<int>(g_hudMaxTargets), &nTracks);
        labels = det::Labels();
        lines = det::SightLines(&nLines);
        int drop = 0;
        targets::EndFrame(dt, 0, &drop); // drain the AABB scan so it can't go stale
    }
    else
        tracks = targets::EndFrame(dt, static_cast<int>(g_hudMaxTargets), &nTracks);

    InterlockedExchange(&g_inRetroPass, 1); // real bb from GetBuffer; our draws never skipped/scaled
    ID3D11Texture2D* bb = nullptr;
    if (SUCCEEDED(target->GetBuffer(0, __uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&bb))) && bb != nullptr)
    {
        UINT w = 0, h = 0;
        ID3D11RenderTargetView* rtv = retro::PostRTV(bb, &w, &h);
        if (rtv != nullptr)
        {
            char tgt[24];
            _snprintf_s(tgt, _TRUNCATE, "TGT %02d", nTracks);
            const char* status[4] = { "TERMINATOR VISION",
                                      g_termOnnx != 0 ? "AI ONNX DML" : "AI HLSL",
                                      detOn ? "DET ONNX" : "SCAN AABB", tgt };
            hud::Draw(ctx, rtv, w, h, g_termBrackets != 0 ? tracks : nullptr, nTracks,
                      status, 4, labels, lines, nLines);
        }
        if (g_termLens != 0)
            retro::ApplyTo(ctx, bb, retro::Fx::TermLens);
        bb->Release();
    }
    InterlockedExchange(&g_inRetroPass, 0);
}

thread_local int t_inPresent = 0; // DXGI/overlays may route Present <-> Present1 internally

HRESULT PresentCommon(IDXGISwapChain* sc, UINT sync, UINT flags,
                      const DXGI_PRESENT_PARAMETERS* pp, bool one) noexcept
{
    auto presentGame = [&]() -> HRESULT
    {
        if (one)
            return OrigV(o_Present1, sc, kIdx_SwapChain1Present1)(
                static_cast<IDXGISwapChain1*>(sc), sync, flags, pp);
        return OrigV(o_Present, sc, kIdx_SwapChainPresent)(sc, sync, flags);
    };
    DrainReleases(); // flush deferred captures — safe spot, not inside a create call
    // DXGI_PRESENT_TEST presents nothing (occlusion probe while minimized/alt-tabbed);
    // nested calls are the same frame. Neither may run our passes.
    if ((flags & DXGI_PRESENT_TEST) != 0 || t_inPresent != 0)
        return presentGame();
    struct Guard { Guard() { ++t_inPresent; } ~Guard() { --t_inPresent; } } guard;
    // CPU frame time: QPC delta between presents, summed for StatsTick.
    static LARGE_INTEGER prevPresent = {};
    LARGE_INTEGER now = {};
    QueryPerformanceCounter(&now);
    if (prevPresent.QuadPart != 0)
        InterlockedExchangeAdd64(&g_frameTickSum, now.QuadPart - prevPresent.QuadPart);
    prevPresent = now;
    if (g_passDump > 0)
        InterlockedDecrement(&g_passDump);

    if (InterlockedIncrement64(&g_presents) == 1)
    {
        DXGI_SWAP_CHAIN_DESC d = {};
        if (SUCCEEDED(sc->GetDesc(&d)))
            Log::Info("hooks", "swapchain desc: %ux%u fmt=%u buffers=%u effect=%u flags=0x%X windowed=%d",
                      d.BufferDesc.Width, d.BufferDesc.Height, static_cast<UINT>(d.BufferDesc.Format),
                      d.BufferCount, static_cast<UINT>(d.SwapEffect), d.Flags, d.Windowed);
    }
    const auto mode = static_cast<hooks::RetroMode>(g_mode);
    const retro::Fx postFx = mode == hooks::RetroMode::Ascii   ? retro::Fx::Ascii
                           : mode == hooks::RetroMode::RetroFX ? retro::Fx::RetroFX
                                                               : retro::Fx::None;
    // Resolve the presenting chain's OWN device + immediate context — multi-device
    // engines (Watch Dogs 2 spawns probe/loading devices) can leave the captured
    // globals bound to a different device than the chain that actually presents,
    // which silently breaks retro passes that create resources or draw via the
    // wrong device's context.
    ID3D11Device*        scDev = nullptr;
    ID3D11DeviceContext* scCtx = nullptr;
    const bool mayUseRetro = postFx != retro::Fx::None ||
                             mode == hooks::RetroMode::Terminator ||
                             mode == hooks::RetroMode::ConsoleWire ||
                             g_gpuBurn > 0 || g_proxyBB != nullptr ||
                             (g_scIsFlip == 0 && RenderScale() > 1);
    if (mayUseRetro &&
        SUCCEEDED(sc->GetDevice(__uuidof(ID3D11Device), reinterpret_cast<void**>(&scDev))) &&
        scDev != nullptr)
        scDev->GetImmediateContext(&scCtx);

    const bool proxyPresent = g_proxyBB != nullptr && g_proxySC == sc &&
                              scDev != nullptr && scCtx != nullptr;
    // Our extra windowed chain must never present while the game's chain owns the
    // output in exclusive fullscreen — DXGI would fight over the window/mode.
    BOOL exclusiveFs = FALSE;
    if (!proxyPresent && g_scIsFlip == 0 && RenderScale() > 1 && scCtx != nullptr)
        sc->GetFullscreenState(&exclusiveFs, nullptr);
    const bool scaledPresent = !proxyPresent && g_scIsFlip == 0 && RenderScale() > 1 &&
                               scCtx != nullptr && !exclusiveFs && EnsurePresentChain(sc);
    if (proxyPresent)
    {
        InterlockedExchange(&g_inRetroPass, 1);
        retro::Ensure(scDev);
        retro::UpscaleTexture(scCtx, g_proxyBB, sc, postFx);
        InterlockedExchange(&g_inRetroPass, 0);
    }
    if (postFx != retro::Fx::None && !scaledPresent && !proxyPresent &&
        scDev != nullptr && scCtx != nullptr)
    {
        retro::Ensure(scDev);
        InterlockedExchange(&g_inRetroPass, 1); // never skip/scale our own fullscreen pass
        retro::Apply(scCtx, sc, postFx);
        InterlockedExchange(&g_inRetroPass, 0);
    }
    // Terminator: base on the game's bb (shrunken in the scaled path), then HUD + lens on
    // whichever chain presents — the game's, or ours after the integer upscale below.
    const bool terminator = mode == hooks::RetroMode::Terminator && scDev != nullptr && scCtx != nullptr;
    if (terminator)
    {
        TerminatorBase(scCtx, scDev, sc);
        if (!scaledPresent)
            TerminatorOverlay(scCtx, sc);
    }

    // Synthetic GPU load for benchmarking ("GpuBurn" config). Runs even in Off
    // mode so a throttled baseline can be compared 1:1 with a throttled retro run.
    if (g_gpuBurn > 0 && scCtx != nullptr && scDev != nullptr)
    {
        InterlockedExchange(&g_inRetroPass, 1); // our own draws must not be skipped
        retro::Ensure(scDev);
        retro::BurnGpu(scCtx, sc);
        InterlockedExchange(&g_inRetroPass, 0);
    }
    if (mode == hooks::RetroMode::ConsoleWire)
    {
        wire::EndFrame(scCtx, scDev, g_projCB); // rasterize+blit what this frame collected
        wire::BeginFrame();                     // reset for the next frame
    }
    // Scaled path: point-sample the shrunken frame into our native-size chain and
    // present THAT — true integer scaling, no bilinear stretch from the game chain.
    HRESULT hr;
    if (scaledPresent)
    {
        InterlockedExchange(&g_inRetroPass, 1);
        retro::Ensure(scDev);
        retro::UpscaleBlit(scCtx, sc, g_sc2, postFx);
        InterlockedExchange(&g_inRetroPass, 0);
        if (terminator)
            TerminatorOverlay(scCtx, g_sc2);
        hr = OrigV(o_Present, g_sc2, kIdx_SwapChainPresent)(g_sc2, sync, flags);
        if (FAILED(hr) && hr != DXGI_ERROR_DEVICE_REMOVED && hr != DXGI_ERROR_DEVICE_RESET)
        {
            // Never surface a failure of OUR chain to the game (engines treat a
            // failed Present as device loss and bail). Drop it and present theirs.
            Log::Warn("scale", "present chain Present failed 0x%08lX - disabled, presenting game chain",
                      static_cast<unsigned long>(hr));
            g_sc2->Release();
            g_sc2 = nullptr;
            retro::DropUpscaleChain();
            InterlockedExchange(&g_sc2Failed, 1);
            hr = presentGame();
        }
    }
    else
        hr = presentGame();
    if (scCtx != nullptr) scCtx->Release();
    if (scDev != nullptr) scDev->Release();
    return hr;
}

HRESULT STDMETHODCALLTYPE hkPresent(IDXGISwapChain* sc, UINT sync, UINT flags)
{
    return PresentCommon(sc, sync, flags, nullptr, false);
}

HRESULT STDMETHODCALLTYPE hkPresent1(IDXGISwapChain1* sc, UINT sync, UINT flags,
                                     const DXGI_PRESENT_PARAMETERS* pp)
{
    return PresentCommon(sc, sync, flags, pp, true);
}

// ResizeBuffers1 (DXGI 1.4) has the same "no outstanding backbuffer refs" rule;
// without this detour our cached refs make it fail and the game aborts.
HRESULT STDMETHODCALLTYPE hkResizeBuffers1(IDXGISwapChain3* sc, UINT count, UINT w, UINT h,
                                           DXGI_FORMAT fmt, UINT flags, const UINT* nodeMask,
                                           IUnknown* const* queues)
{
    retro::ReleaseBackbuffer();
    ai::ReleaseBackbuffer();
    retro::DropUpscaleChain();
    ReleaseProxyBackbuffer();
    InterlockedExchange(&g_sc2Failed, 0);
    const HRESULT hr = OrigV(o_ResizeBuffers1, sc, kIdx_SwapChain3ResizeBuffers1)(
        sc, count, w, h, fmt, flags, nodeMask, queues);
    if (SUCCEEDED(hr) && w > 0 && h > 0 && (w != g_scW || h != g_scH))
    {
        g_scW = w; g_scH = h;
        ClearScalingTables();
        if (g_sc2 != nullptr) { g_sc2->Release(); g_sc2 = nullptr; }
    }
    Log::Info("hooks", "ResizeBuffers1(%u, %ux%u, fmt=%u) -> 0x%08lX", count, w, h,
              static_cast<UINT>(fmt), static_cast<unsigned long>(hr));
    return hr;
}

const char* DxgiFormatName(UINT f) noexcept
{
    switch (f)
    {
    case DXGI_FORMAT_R32G32B32A32_FLOAT: return "f32x4";
    case DXGI_FORMAT_R32G32B32_FLOAT:    return "f32x3";
    case DXGI_FORMAT_R32G32_FLOAT:       return "f32x2";
    case DXGI_FORMAT_R32_FLOAT:          return "f32";
    case DXGI_FORMAT_R8G8B8A8_UNORM:     return "u8x4n";
    case DXGI_FORMAT_R8G8B8A8_UINT:      return "u8x4";
    case DXGI_FORMAT_R16G16_FLOAT:       return "f16x2";
    case DXGI_FORMAT_R16G16B16A16_FLOAT: return "f16x4";
    case DXGI_FORMAT_R32G32_UINT:        return "u32x2";
    case DXGI_FORMAT_R32G32B32_UINT:     return "u32x3";
    case DXGI_FORMAT_R32G32B32A32_UINT:  return "u32x4";
    case DXGI_FORMAT_R32_UINT:           return "u32";
    case DXGI_FORMAT_R16G16_SINT:        return "s16x2";
    case DXGI_FORMAT_R16G16B16A16_SNORM: return "s16x4n";
    case DXGI_FORMAT_R16G16B16A16_SINT:  return "s16x4";
    case DXGI_FORMAT_R10G10B10A2_UNORM:  return "rgb10a2n";
    case DXGI_FORMAT_R11G11B10_FLOAT:    return "r11g11b10";
    }
    return "?";
}

HRESULT STDMETHODCALLTYPE hkCreateInputLayout(ID3D11Device* dev, const D3D11_INPUT_ELEMENT_DESC* descs,
                                              UINT num, const void* bytecode, SIZE_T bcLen,
                                              ID3D11InputLayout** ppOut)
{
    const HRESULT hr = Orig(o_CreateInputLayout, dev, kIdx_DevCreateInputLayout)(dev, descs, num, bytecode, bcLen, ppOut);
    if (SUCCEEDED(hr) && descs != nullptr)
    {
        Log::Info("recon", "CreateInputLayout num=%u il=%p", num, ppOut != nullptr ? *ppOut : nullptr);
        for (UINT i = 0; i < num && i < 16; ++i)
        {
            const D3D11_INPUT_ELEMENT_DESC& e = descs[i];
            Log::Info("recon", "   [%2u] %-12s%u fmt=%s(%u) slot=%u off=%u cls=%u step=%u",
                      i, e.SemanticName ? e.SemanticName : "?", e.SemanticIndex,
                      DxgiFormatName(e.Format), e.Format, e.InputSlot,
                      e.AlignedByteOffset, e.InputSlotClass, e.InstanceDataStepRate);
        }
    }
    return hr;
}

// ---- mode plumbing ------------------------------------------------------------

using FnD3DCompile = HRESULT(WINAPI*)(LPCVOID, SIZE_T, LPCSTR, const D3D_SHADER_MACRO*,
                                     ID3DInclude*, LPCSTR, LPCSTR, UINT, UINT,
                                     ID3DBlob**, ID3DBlob**);

const char kFlatPsSrc[] =
    "float4 main(float4 pos : SV_POSITION) : SV_TARGET { return float4(0.15f, 0.95f, 0.35f, 1.0f); }";

ID3D11PixelShader* EnsureFlatPS() noexcept
{
    if (g_flatPS != nullptr)
        return g_flatPS;
    if (InterlockedCompareExchange(&g_flatPSTried, 1, 0) != 0)
        return nullptr; // already tried and failed
    if (g_device == nullptr)
        return nullptr;

    HMODULE d3dc = LoadLibraryW(L"d3dcompiler_47.dll");
    if (d3dc == nullptr)
        d3dc = LoadLibraryW(L"d3dcompiler_43.dll");
    if (d3dc == nullptr)
    {
        Log::Warn("hooks", "FlatPS: no d3dcompiler_4x.dll available");
        return nullptr;
    }
    const auto compile = reinterpret_cast<FnD3DCompile>(GetProcAddress(d3dc, "D3DCompile"));
    if (compile == nullptr)
        return nullptr;

    ID3DBlob* blob = nullptr;
    ID3DBlob* errors = nullptr;
    const HRESULT hr = compile(kFlatPsSrc, sizeof(kFlatPsSrc) - 1, nullptr, nullptr, nullptr,
                               "main", "ps_4_0", 0, 0, &blob, &errors);
    if (FAILED(hr) || blob == nullptr)
    {
        if (errors != nullptr)
        {
            Log::Warn("hooks", "FlatPS compile: %s", static_cast<const char*>(errors->GetBufferPointer()));
            errors->Release();
        }
        return nullptr;
    }
    const HRESULT cr = g_device->CreatePixelShader(blob->GetBufferPointer(),
                                                   blob->GetBufferSize(), nullptr, &g_flatPS);
    blob->Release();
    if (FAILED(cr))
    {
        Log::Warn("hooks", "CreatePixelShader(flat) failed hr=0x%08lX", static_cast<unsigned long>(cr));
        return nullptr;
    }
    Log::Info("hooks", "FlatPS shader created %p", g_flatPS);
    return g_flatPS;
}

ID3D11RasterizerState* EnsureWireRS() noexcept
{
    if (g_wireRS != nullptr)
        return g_wireRS;
    if (InterlockedCompareExchange(&g_wireRSTried, 1, 0) != 0)
        return nullptr;
    if (g_device == nullptr)
        return nullptr;

    D3D11_RASTERIZER_DESC rd = {};
    rd.FillMode              = D3D11_FILL_WIREFRAME;
    rd.CullMode              = D3D11_CULL_NONE;
    rd.DepthClipEnable       = TRUE;
    rd.AntialiasedLineEnable = FALSE;
    const HRESULT hr = g_device->CreateRasterizerState(&rd, &g_wireRS);
    if (FAILED(hr))
    {
        Log::Warn("hooks", "CreateRasterizerState(wire) failed hr=0x%08lX", static_cast<unsigned long>(hr));
        return nullptr;
    }
    Log::Info("hooks", "wireframe rasterizer state created %p", g_wireRS);
    return g_wireRS;
}

void STDMETHODCALLTYPE hkPSSetShader(ID3D11DeviceContext* ctx, ID3D11PixelShader* ps,
                                     ID3D11ClassInstance* const* inst, UINT numInst)
{
    g_curPS = ps; // the shader the game intends, before any substitution
    switch (static_cast<hooks::RetroMode>(g_mode))
    {
    case hooks::RetroMode::NullPS:
        Orig(o_PSSetShader, ctx, kIdx_PSSetShader)(ctx, nullptr, inst, numInst);
        return;
    case hooks::RetroMode::FlatPS:
    {
        ID3D11PixelShader* flat = EnsureFlatPS();
        Orig(o_PSSetShader, ctx, kIdx_PSSetShader)(ctx, flat, inst, numInst); // flat==nullptr degrades to NullPS
        return;
    }
    default:
        Orig(o_PSSetShader, ctx, kIdx_PSSetShader)(ctx, ps, inst, numInst);
        return;
    }
}

void STDMETHODCALLTYPE hkRSSetState(ID3D11DeviceContext* ctx, ID3D11RasterizerState* rs)
{
    if (static_cast<hooks::RetroMode>(g_mode) == hooks::RetroMode::Wireframe)
    {
        ID3D11RasterizerState* wire = EnsureWireRS();
        if (wire != nullptr)
        {
            Orig(o_RSSetState, ctx, kIdx_RSSetState)(ctx, wire);
            return;
        }
    }
    Orig(o_RSSetState, ctx, kIdx_RSSetState)(ctx, rs);
}

// ---- Phase 12 detours ----------------------------------------------------------

// Shrink screen-sized render targets / depth buffers at creation. A texture is a
// scaling candidate when it is RT- or DS-bindable and either exactly screen-sized
// or a proportional divisor of it (1/2, 1/4 res effects like bloom) — shadow maps
// and thumbnails don't match that shape and stay untouched.
HRESULT STDMETHODCALLTYPE hkCreateTexture2D(ID3D11Device* dev, const D3D11_TEXTURE2D_DESC* desc,
                                            const D3D11_SUBRESOURCE_DATA* init,
                                            ID3D11Texture2D** out)
{
    D3D11_TEXTURE2D_DESC d2 = {};
    const D3D11_TEXTURE2D_DESC* gameDesc = desc;
    const int s = RenderScale();
    UINT ow = 0, oh = 0;
    bool scaled = false;
    // Never resize textures whose size is a contract with someone else: shared
    // (interop/overlays/video), GDI, cube maps, or ones uploaded with initial data.
    constexpr UINT kNoScaleMisc = D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX |
                                  D3D11_RESOURCE_MISC_SHARED_NTHANDLE | D3D11_RESOURCE_MISC_GDI_COMPATIBLE |
                                  D3D11_RESOURCE_MISC_TEXTURECUBE;
    if (s > 1 && g_inSCCreate == 0 && desc != nullptr && init == nullptr && g_scW > 0 && g_scH > 0 &&
        (desc->BindFlags & (D3D11_BIND_RENDER_TARGET | D3D11_BIND_DEPTH_STENCIL)) != 0 &&
        (desc->MiscFlags & kNoScaleMisc) == 0 && desc->Usage == D3D11_USAGE_DEFAULT)
    {
        if (ScreenDivisor(desc->Width, desc->Height) != 0)
        {
            d2 = *desc;
            d2.Width  = desc->Width  / s;  if (d2.Width == 0)  d2.Width = 1;
            d2.Height = desc->Height / s;  if (d2.Height == 0) d2.Height = 1;
            // An explicit mip count valid at full size can exceed the chain of the
            // shrunken size -> E_INVALIDARG -> game null-derefs its texture.
            UINT maxMips = 1;
            for (UINT m = d2.Width > d2.Height ? d2.Width : d2.Height; m > 1; m >>= 1)
                ++maxMips;
            if (d2.MipLevels > maxMips)
                d2.MipLevels = maxMips;
            ow = desc->Width; oh = desc->Height;
            desc = &d2;
            scaled = true;
        }
    }
    const FnCreateTexture2D orig = Orig(o_CreateTexture2D, dev, kIdx_DevCreateTexture2D);
    HRESULT hr = orig(dev, desc, init, out);
    if (scaled && FAILED(hr))
    {
        // Whatever the driver disliked about the shrunken desc, the game's own
        // desc must keep working.
        Log::Warn("scale", "tex2d %ux%u shrink rejected 0x%08lX - creating native",
                  ow, oh, static_cast<unsigned long>(hr));
        desc = gameDesc;
        scaled = false;
        hr = orig(dev, desc, init, out);
    }
    if (scaled && SUCCEEDED(hr) && out != nullptr && *out != nullptr)
    {
        SetScaled(*out, ow, oh);
        Log::Info("scale", "tex2d %ux%u -> %ux%u", ow, oh, d2.Width, d2.Height);
    }
    else if (SUCCEEDED(hr) && out != nullptr && *out != nullptr)
        UnsetScaled(*out);
    if (!scaled && s > 1 && g_inSCCreate == 0 && desc != nullptr && desc->Width >= 256 &&
        (desc->BindFlags & (D3D11_BIND_RENDER_TARGET | D3D11_BIND_DEPTH_STENCIL)) != 0)
    {
        static volatile LONG keptLogged = 0;
        if (InterlockedIncrement(&keptLogged) <= 48)
            Log::Info("scale", "tex2d kept %ux%u fmt=%u bind=0x%X (screen %ux%u)", desc->Width,
                      desc->Height, static_cast<UINT>(desc->Format), desc->BindFlags, g_scW, g_scH);
    }
    return hr;
}

// Terminator brackets: local AABBs of vertex buffers from their initial data.
HRESULT STDMETHODCALLTYPE hkCreateBuffer(ID3D11Device* dev, const D3D11_BUFFER_DESC* desc,
                                         const D3D11_SUBRESOURCE_DATA* init, ID3D11Buffer** out)
{
    const HRESULT hr = Orig(o_CreateBuffer, dev, kIdx_DevCreateBuffer)(dev, desc, init, out);
    if (SUCCEEDED(hr) && g_termBrackets != 0 && out != nullptr && *out != nullptr)
        targets::OnCreateBuffer(desc, init, *out);
    return hr;
}

HRESULT STDMETHODCALLTYPE hkCreateRTV(ID3D11Device* dev, ID3D11Resource* res,
                                      const D3D11_RENDER_TARGET_VIEW_DESC* d,
                                      ID3D11RenderTargetView** out)
{
    const FnCreateRTV orig = Orig(o_CreateRTV, dev, kIdx_DevCreateRenderTargetView);
    HRESULT hr = orig(dev, res, d, out);
    if (FAILED(hr) && res != nullptr && res == g_proxyBB && d != nullptr && d->Format != g_proxyFmt)
    {
        D3D11_RENDER_TARGET_VIEW_DESC d2 = *d;
        d2.Format = g_proxyFmt;
        hr = orig(dev, res, &d2, out);
        Log::Warn("scale", "proxy RTV fmt %u -> %u (0x%08lX)", static_cast<UINT>(d->Format),
                  static_cast<UINT>(g_proxyFmt), static_cast<unsigned long>(hr));
    }
    return hr;
}

HRESULT STDMETHODCALLTYPE hkCreateDSV(ID3D11Device* dev, ID3D11Resource* res,
                                      const D3D11_DEPTH_STENCIL_VIEW_DESC* d,
                                      ID3D11DepthStencilView** out)
{
    return Orig(o_CreateDSV, dev, kIdx_DevCreateDepthStencilView)(dev, res, d, out);
}

HRESULT STDMETHODCALLTYPE hkCreatePixelShader(ID3D11Device* dev, const void* bc, SIZE_T len,
                                              ID3D11ClassLinkage* cl, ID3D11PixelShader** out)
{
    const HRESULT hr = Orig(o_CreatePS, dev, kIdx_DevCreatePixelShader)(dev, bc, len, cl, out);
    if (SUCCEEDED(hr) && out != nullptr && *out != nullptr && bc != nullptr)
    {
        const size_t s = (reinterpret_cast<UINT_PTR>(*out) >> 4) & 1023;
        g_psMap[s] = { *out, HashBytecode(bc, len) };
    }
    return hr;
}

void STDMETHODCALLTYPE hkOMSetRenderTargets(ID3D11DeviceContext* ctx, UINT n,
                                            ID3D11RenderTargetView* const* rtvs,
                                            ID3D11DepthStencilView* dsv)
{
    Orig(o_OMSetRTs, ctx, kIdx_OMSetRenderTargets)(ctx, n, rtvs, dsv);
    if (g_inRetroPass == 0)            // bound-state tracking belongs to the game
        NoteBoundTargets(ctx, n, rtvs, dsv);
}

void STDMETHODCALLTYPE hkOMSetRTsAndUAVs(ID3D11DeviceContext* ctx, UINT n,
                                         ID3D11RenderTargetView* const* rtvs,
                                         ID3D11DepthStencilView* dsv,
                                         UINT uavStart, UINT uavN,
                                         ID3D11UnorderedAccessView* const* uavs,
                                         const UINT* uavCounts)
{
    Orig(o_OMSetRTsUAVs, ctx, kIdx_OMSetRTsAndUAVs)(ctx, n, rtvs, dsv, uavStart, uavN, uavs, uavCounts);
    if (g_inRetroPass == 0)
        NoteBoundTargets(ctx, n, rtvs, dsv);
}

void STDMETHODCALLTYPE hkRSSetViewports(ID3D11DeviceContext* ctx, UINT n,
                                        const D3D11_VIEWPORT* vps)
{
    const FnRSSetViewports orig = Orig(o_RSSetViewports, ctx, kIdx_RSSetViewports);
    if (g_inRetroPass != 0)            // our own Present-time passes use raw state
    {
        orig(ctx, n, vps);
        return;
    }
    if (vps != nullptr && n > 0)
    {
        g_vpN = n > D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE
              ? D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE : n;
        memcpy(g_vp, vps, g_vpN * sizeof(D3D11_VIEWPORT));
    }
    const int s = RenderScale();
    if (g_scaledBound && s > 1 && vps != nullptr && n > 0)
    {
        D3D11_VIEWPORT out[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE];
        const UINT m = n > D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE
                     ? D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE : n;
        for (UINT i = 0; i < m; ++i)
        {
            out[i] = vps[i];
            out[i].TopLeftX /= s; out[i].TopLeftY /= s;
            out[i].Width    /= s; out[i].Height    /= s;
        }
        orig(ctx, m, out);
        return;
    }
    orig(ctx, n, vps);
}

void STDMETHODCALLTYPE hkRSSetScissorRects(ID3D11DeviceContext* ctx, UINT n,
                                           const D3D11_RECT* rects)
{
    const FnRSSetScissor orig = Orig(o_RSSetScissor, ctx, kIdx_RSSetScissorRects);
    if (g_inRetroPass != 0)
    {
        orig(ctx, n, rects);
        return;
    }
    if (rects != nullptr && n > 0)
    {
        g_srN = n > D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE
              ? D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE : n;
        memcpy(g_sr, rects, g_srN * sizeof(D3D11_RECT));
    }
    const int s = RenderScale();
    if (g_scaledBound && s > 1 && rects != nullptr && n > 0)
    {
        D3D11_RECT out[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE];
        const UINT m = n > D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE
                     ? D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE : n;
        for (UINT i = 0; i < m; ++i)
        {
            out[i].left   = rects[i].left   / s;
            out[i].top    = rects[i].top    / s;
            out[i].right  = rects[i].right  / s;
            out[i].bottom = rects[i].bottom / s;
        }
        orig(ctx, m, out);
        return;
    }
    orig(ctx, n, rects);
}

// Box coords live in source space, dst offsets in dest space — divide by N for
// whichever side is a scaled resource.
void STDMETHODCALLTYPE hkCopySubresourceRegion(ID3D11DeviceContext* ctx,
    ID3D11Resource* dst, UINT dstSub, UINT dx, UINT dy, UINT dz,
    ID3D11Resource* src, UINT srcSub, const D3D11_BOX* box)
{
    const int s = RenderScale();
    if (s > 1 && (FindScaled(dst) != nullptr || FindScaled(src) != nullptr))
    {
        D3D11_BOX b2 = {};
        if (FindScaled(dst) != nullptr) { dx /= s; dy /= s; }
        if (box != nullptr && FindScaled(src) != nullptr)
        {
            b2 = *box;
            b2.left   /= s; b2.right  /= s;
            b2.top    /= s; b2.bottom /= s;
            box = &b2;
        }
        Orig(o_CopySubres, ctx, kIdx_CopySubresourceRegion)(ctx, dst, dstSub, dx, dy, dz, src, srcSub, box);
        return;
    }
    Orig(o_CopySubres, ctx, kIdx_CopySubresourceRegion)(ctx, dst, dstSub, dx, dy, dz, src, srcSub, box);
}

HRESULT STDMETHODCALLTYPE hkGetBuffer(IDXGISwapChain* sc, UINT idx, REFIID riid, void** out)
{
    if (g_inRetroPass == 0 && g_inSCCreate == 0 && out != nullptr &&
        g_scIsFlip != 0 && RenderScale() > 1)
    {
        ID3D11Texture2D* p = EnsureProxyBackbuffer(sc);
        if (p != nullptr)
            return p->QueryInterface(riid, out);
    }
    return OrigV(o_GetBuffer, sc, kIdx_SwapChainGetBuffer)(sc, idx, riid, out);
}

// Our own native-size windowed swap chain — presents the integer-scaled image
// so DXGI's bilinear backbuffer stretch is bypassed entirely.
bool EnsurePresentChain(IDXGISwapChain* sc) noexcept
{
    if (g_sc2 != nullptr)
        return true;
    if (g_sc2Failed != 0)
        return false; // don't retry (and log) every frame — reset on resize/recreate
    DXGI_SWAP_CHAIN_DESC gd = {};
    ID3D11Device* dev = nullptr;
    IDXGIDevice* dxgiDev = nullptr;
    IDXGIAdapter* adapter = nullptr;
    IDXGIFactory* fac = nullptr;
    bool ok = false;
    if (SUCCEEDED(sc->GetDesc(&gd)) && gd.OutputWindow != nullptr && gd.Windowed &&
        SUCCEEDED(sc->GetDevice(__uuidof(ID3D11Device), reinterpret_cast<void**>(&dev))) &&
        SUCCEEDED(dev->QueryInterface(__uuidof(IDXGIDevice), reinterpret_cast<void**>(&dxgiDev))) &&
        SUCCEEDED(dxgiDev->GetAdapter(&adapter)) &&
        SUCCEEDED(adapter->GetParent(__uuidof(IDXGIFactory), reinterpret_cast<void**>(&fac))))
    {
        DXGI_SWAP_CHAIN_DESC d = {};
        d.BufferDesc.Format = gd.BufferDesc.Format;  // BufferDesc 0x0 = window client size
        d.SampleDesc.Count  = 1;
        d.BufferUsage       = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        d.BufferCount       = 1;
        d.OutputWindow      = gd.OutputWindow;
        d.Windowed          = TRUE;
        d.SwapEffect        = DXGI_SWAP_EFFECT_DISCARD;
        InterlockedExchange(&g_inSCCreate, 1);       // our chain: never shrink, never hook
        ok = SUCCEEDED(fac->CreateSwapChain(dev, &d, &g_sc2));
        InterlockedExchange(&g_inSCCreate, 0);
    }
    if (fac != nullptr) fac->Release();
    if (adapter != nullptr) adapter->Release();
    if (dxgiDev != nullptr) dxgiDev->Release();
    if (dev != nullptr) dev->Release();
    if (ok) Log::Info("scale", "present chain created — integer upscale to window");
    else    Log::Warn("scale", "present chain creation failed — DXGI stretch fallback");
    if (!ok)
        InterlockedExchange(&g_sc2Failed, 1);
    return ok;
}

// Resolve the game's believed screen dims: BufferDesc if nonzero, else the
// output window's client rect (windowed BufferDesc 0x0 = "use window size").
bool BelievedSwapChainDims(const DXGI_SWAP_CHAIN_DESC* desc, UINT* w, UINT* h) noexcept
{
    if (desc != nullptr && desc->BufferDesc.Width > 0 && desc->BufferDesc.Height > 0)
    {
        *w = desc->BufferDesc.Width;
        *h = desc->BufferDesc.Height;
        return true;
    }
    if (desc != nullptr && desc->OutputWindow != nullptr)
    {
        RECT rc = {};
        if (GetClientRect(desc->OutputWindow, &rc) && rc.right > 0 && rc.bottom > 0)
        {
            *w = static_cast<UINT>(rc.right - rc.left);
            *h = static_cast<UINT>(rc.bottom - rc.top);
            return true;
        }
    }
    return false;
}

// Games recreate their swap chain on resolution/fullscreen/vsync changes. Any
// backbuffer reference we still hold keeps the OLD chain alive, and a second
// chain on the same HWND is rejected (flip model: E_ACCESSDENIED) — the game then
// crashes on a null swap chain. Drop everything tied to the old chain first.
void ReleaseAllChainRefs() noexcept
{
    retro::ReleaseBackbuffer();
    ai::ReleaseBackbuffer();
    retro::DropUpscaleChain();
    ReleaseProxyBackbuffer();
    if (g_sc2 != nullptr) { g_sc2->Release(); g_sc2 = nullptr; }
    InterlockedExchange(&g_sc2Failed, 0);
}

HRESULT STDMETHODCALLTYPE hkCreateSwapChain(IDXGIFactory* fac, IUnknown* dev, DXGI_SWAP_CHAIN_DESC* desc, IDXGISwapChain** ppSC)
{
    const FnCreateSwapChain orig = Orig(o_CreateSwapChain, fac, kIdx_FactoryCreateSwapChain);
    if (g_inSCCreate != 0)            // our own present chain — pass through, no shrink/hook
        return orig(fac, dev, desc, ppSC);
    ReleaseAllChainRefs();
    const bool flip = desc != nullptr && desc->SwapEffect >= DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
    InterlockedExchange(&g_scIsFlip, flip ? 1 : 0);
    if (flip)
        Log::Info("scale", "flip-model swap chain - RenderScale via proxy backbuffer");
    const int s = flip ? 1 : RenderScale();
    DXGI_SWAP_CHAIN_DESC d2 = {};
    const DXGI_SWAP_CHAIN_DESC* useDesc = desc;
    if (s > 1 && desc != nullptr)
    {
        UINT bw = 0, bh = 0;
        if (BelievedSwapChainDims(desc, &bw, &bh))
        {
            g_scW = bw; g_scH = bh;                       // game's believed size
            d2 = *desc;
            d2.BufferDesc.Width  = bw / static_cast<UINT>(s);   // REAL buffer is small —
            d2.BufferDesc.Height = bh / static_cast<UINT>(s);   // blit-model present
            if (d2.BufferDesc.Width == 0)  d2.BufferDesc.Width  = 1;  // stretches it
            if (d2.BufferDesc.Height == 0) d2.BufferDesc.Height = 1;  // to the window
            d2.BufferDesc.Scaling = DXGI_MODE_SCALING_STRETCHED;
            useDesc = &d2;
            Log::Info("scale", "swapchain believed %ux%u -> real %ux%u",
                      bw, bh, d2.BufferDesc.Width, d2.BufferDesc.Height);
        }
    }
    InterlockedExchange(&g_inSCCreate, 1);  // DXGI-internal bb create must not reshrink
    HRESULT hr = orig(fac, dev, const_cast<DXGI_SWAP_CHAIN_DESC*>(useDesc), ppSC);
    if (FAILED(hr) && useDesc != desc)
    {
        Log::Warn("scale", "shrunken CreateSwapChain rejected 0x%08lX - creating native",
                  static_cast<unsigned long>(hr));
        hr = orig(fac, dev, desc, ppSC);
    }
    InterlockedExchange(&g_inSCCreate, 0);
    if (SUCCEEDED(hr) && ppSC != nullptr && *ppSC != nullptr)
    {
        Log::Info("hooks", "IDXGIFactory::CreateSwapChain -> sc=%p", *ppSC);
        hooks::InstallSwapChainHooks(*ppSC);
        if ((g_scW == 0 || flip) && RenderScale() > 1)
        {
            DXGI_SWAP_CHAIN_DESC rd = {};
            if (SUCCEEDED((*ppSC)->GetDesc(&rd)) && rd.BufferDesc.Width > 0)
            {
                g_scW = rd.BufferDesc.Width;
                g_scH = rd.BufferDesc.Height;
                Log::Info("scale", "screen dims %ux%u (post-creation)", g_scW, g_scH);
            }
        }
        RegisterScaledBackbuffer(*ppSC);
    }
    return hr;
}

HRESULT STDMETHODCALLTYPE hkCreateSCForHwnd(IDXGIFactory2* fac, IUnknown* dev, HWND hwnd,
                                            const DXGI_SWAP_CHAIN_DESC1* desc,
                                            const DXGI_SWAP_CHAIN_FULLSCREEN_DESC* fs,
                                            IDXGIOutput* out, IDXGISwapChain1** ppSC)
{
    const FnCreateSCForHwnd orig = Orig(o_CreateSCForHwnd, fac, kIdx_Factory2CreateSCForHwnd);
    if (g_inSCCreate != 0)
        return orig(fac, dev, hwnd, desc, fs, out, ppSC);
    ReleaseAllChainRefs();
    const bool flip = desc != nullptr && desc->SwapEffect >= DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
    InterlockedExchange(&g_scIsFlip, flip ? 1 : 0);
    if (flip)
        Log::Info("scale", "flip-model swap chain - RenderScale via proxy backbuffer");
    const int s = flip ? 1 : RenderScale();
    DXGI_SWAP_CHAIN_DESC1 d2 = {};
    const DXGI_SWAP_CHAIN_DESC1* useDesc = desc;
    if (s > 1 && desc != nullptr && desc->Width > 0 && desc->Height > 0)
    {
        g_scW = desc->Width;
        g_scH = desc->Height;
        d2 = *desc;
        d2.Width  = desc->Width  / static_cast<UINT>(s);
        d2.Height = desc->Height / static_cast<UINT>(s);
        if (d2.Width == 0)  d2.Width = 1;
        if (d2.Height == 0) d2.Height = 1;
        d2.Scaling = DXGI_SCALING_STRETCH;
        useDesc = &d2;
        Log::Info("scale", "swapchain believed %ux%u -> real %ux%u (RenderScale=%d)",
                  g_scW, g_scH, d2.Width, d2.Height, s);
    }
    InterlockedExchange(&g_inSCCreate, 1);
    HRESULT hr = orig(fac, dev, hwnd, useDesc, fs, out, ppSC);
    if (FAILED(hr) && useDesc != desc)
    {
        Log::Warn("scale", "shrunken CreateSwapChainForHwnd rejected 0x%08lX - creating native",
                  static_cast<unsigned long>(hr));
        hr = orig(fac, dev, hwnd, desc, fs, out, ppSC);
    }
    InterlockedExchange(&g_inSCCreate, 0);
    if (SUCCEEDED(hr) && ppSC != nullptr && *ppSC != nullptr)
    {
        Log::Info("hooks", "IDXGIFactory2::CreateSwapChainForHwnd -> sc=%p", *ppSC);
        hooks::InstallSwapChainHooks(*ppSC);
        if ((g_scW == 0 || flip) && RenderScale() > 1)
        {
            DXGI_SWAP_CHAIN_DESC1 rd = {};
            if (SUCCEEDED((*ppSC)->GetDesc1(&rd)) && rd.Width > 0)
            {
                g_scW = rd.Width;
                g_scH = rd.Height;
                Log::Info("scale", "screen dims %ux%u (post-creation)", g_scW, g_scH);
            }
        }
        RegisterScaledBackbuffer(*ppSC);
    }
    return hr;
}

// ---- helpers ------------------------------------------------------------------
bool EnsureMinHook() noexcept
{
    const LONG prev = InterlockedCompareExchange(&g_mhState, 1, 0);
    if (prev == 0)
    {
        const MH_STATUS st = MH_Initialize();
        InterlockedExchange(&g_mhState, (st == MH_OK) ? 2 : 3);
        if (st != MH_OK)
            Log::Error("hooks", "MH_Initialize failed: %d", static_cast<int>(st));
    }
    else
    {
        while (InterlockedCompareExchange(&g_mhState, 0, 0) == 1)
            SwitchToThread();
    }
    return InterlockedCompareExchange(&g_mhState, 0, 0) == 2;
}

// Direct vtable-slot patching for swapchain objects. MinHook trampolines for some
// dxgi methods (e.g. IDXGISwapChain::GetBuffer) relocate RIP-relative operands
// wrongly on this build — first call executed a garbage pointer (Yakuza 0 AV at
// execute of a truncated dxgi-range address). Writing the slot needs no
// trampoline: the previous value IS the original function. Slots are restored in
// Shutdown. Note the class vtable is shared by all instances of the class — one
// patch covers every swapchain of that class.
bool ReadVtblSlot(const void* obj, int index, void*** slotOut, void** curOut) noexcept
{
    __try
    {
        void** vtbl = *reinterpret_cast<void** const*>(obj);
        *slotOut = &vtbl[index];
        *curOut  = vtbl[index];
        return *curOut != nullptr;
    }
    __except (EXCEPTION_EXECUTE_HANDLER)
    {
        return false;
    }
}

// Vtables of wrapper DLLs can live in a section merged with code — dropping the
// execute bit (plain PAGE_READWRITE) would fault every thread running there.
bool WriteVtblSlot(void** slot, void* value) noexcept
{
    __try
    {
        MEMORY_BASIC_INFORMATION mbi = {};
        if (VirtualQuery(slot, &mbi, sizeof(mbi)) == 0)
            return false;
        constexpr DWORD kExec = PAGE_EXECUTE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE |
                                PAGE_EXECUTE_WRITECOPY;
        const DWORD prot = (mbi.Protect & kExec) != 0 ? PAGE_EXECUTE_READWRITE : PAGE_READWRITE;
        DWORD old = 0;
        if (VirtualProtect(slot, sizeof(void*), prot, &old) == 0)
            return false;
        InterlockedExchangePointer(slot, value);
        VirtualProtect(slot, sizeof(void*), old, &old);
        return true;
    }
    __except (EXCEPTION_EXECUTE_HANDLER)
    {
        return false;
    }
}

void PatchVtblEntry(void* obj, int index, void* detour, void** origOut, const char* name) noexcept
{
    if (obj == nullptr || detour == nullptr)
        return;
    void** slot = nullptr;
    void* cur = nullptr;
    if (!ReadVtblSlot(obj, index, &slot, &cur))
    {
        Log::Error("hooks", "vtbl access fault reading %s", name);
        return;
    }
    if (cur == detour)
        return; // already patched (shared vtable)
    AcquireSRWLockExclusive(&g_hookLock);
    const LONG n = g_vtblPatchN;
    bool known = false;
    for (LONG i = 0; i < n && !known; ++i)
        known = g_vtblPatches[i].slot == slot; // someone layered over our patch — leave it
    if (!known && n < static_cast<LONG>(sizeof(g_vtblPatches) / sizeof(g_vtblPatches[0])))
    {
        // Publish the original BEFORE the slot flips: another thread may enter
        // the detour the instant the write lands.
        g_vtblPatches[n] = { slot, cur, detour, name };
        if (origOut != nullptr && *origOut == nullptr)
            *origOut = cur;
        InterlockedExchange(&g_vtblPatchN, n + 1);
        if (WriteVtblSlot(slot, detour))
            Log::Info("hooks", "patched %s vtbl[%d] orig=%p", name, index, cur);
        else
        {
            InterlockedExchange(&g_vtblPatchN, n);
            Log::Error("hooks", "vtbl write failed patching %s (err=%lu)", name, GetLastError());
        }
    }
    else if (!known)
        Log::Warn("hooks", "vtbl patch table full — %s left unhooked", name);
    ReleaseSRWLockExclusive(&g_hookLock);
}

void RestoreVtblPatches() noexcept
{
    AcquireSRWLockExclusive(&g_hookLock);
    const LONG n = g_vtblPatchN;
    for (LONG i = 0; i < n; ++i)
    {
        VtblSlot& p = g_vtblPatches[i];
        void* cur = nullptr;
        __try { cur = *p.slot; }
        __except (EXCEPTION_EXECUTE_HANDLER) { cur = nullptr; }
        if (cur == p.detour) // only restore if nobody layered over us
            WriteVtblSlot(p.slot, p.orig);
    }
    ReleaseSRWLockExclusive(&g_hookLock);
}

// Hooks vtbl[idx] of `obj`. Each distinct implementation (target) gets its own
// trampoline recorded in g_tramps; `orig` keeps the first one as fallback.
void HookEntry(void* obj, int index, void* detour, void** orig, const char* name) noexcept
{
    void** slot = nullptr;
    void* target = nullptr;
    if (!ReadVtblSlot(obj, index, &slot, &target))
    {
        Log::Error("hooks", "vtbl access fault reading %s", name);
        return;
    }
    if (target == detour)
        return;

    void* tramp = nullptr;
    MH_STATUS st = MH_CreateHook(target, detour, &tramp);
    if (st == MH_ERROR_ALREADY_CREATED)
    {
        Log::Info("hooks", "%s already hooked (vtbl[%d]=%p)", name, index, target);
        return;
    }
    if (st != MH_OK || tramp == nullptr)
    {
        Log::Error("hooks", "MH_CreateHook(%s vtbl[%d]=%p) failed: %d",
                   name, index, target, static_cast<int>(st));
        return;
    }
    // Publish target->trampoline before the hook goes live.
    AcquireSRWLockExclusive(&g_hookLock);
    const LONG n = g_trampN;
    const bool room = n < static_cast<LONG>(sizeof(g_tramps) / sizeof(g_tramps[0]));
    if (room)
    {
        g_tramps[n] = { target, tramp };
        if (*orig == nullptr)
            *orig = tramp;
        InterlockedExchange(&g_trampN, n + 1);
    }
    ReleaseSRWLockExclusive(&g_hookLock);
    if (!room)
    {
        MH_RemoveHook(target);
        Log::Warn("hooks", "trampoline table full — %s left unhooked", name);
        return;
    }
    st = MH_EnableHook(target);
    if (st != MH_OK && st != MH_ERROR_ENABLED)
    {
        Log::Error("hooks", "MH_EnableHook(%s) failed: %d", name, static_cast<int>(st));
        return;
    }
    Log::Info("hooks", "hooked %s vtbl[%d] target=%p tramp=%p", name, index, target, tramp);
}

void InstallContextHooks(ID3D11DeviceContext* ctx) noexcept
{
    if (ctx == nullptr)
        return;
    HookEntry(ctx, kIdx_DrawIndexed,          reinterpret_cast<void*>(&hkDrawIndexed),          reinterpret_cast<void**>(&o_DrawIndexed),     "ctx::DrawIndexed");
    HookEntry(ctx, kIdx_Draw,                 reinterpret_cast<void*>(&hkDraw),                 reinterpret_cast<void**>(&o_Draw),            "ctx::Draw");
    HookEntry(ctx, kIdx_Map,                  reinterpret_cast<void*>(&hkMap),                  reinterpret_cast<void**>(&o_Map),             "ctx::Map");
    HookEntry(ctx, kIdx_Unmap,                reinterpret_cast<void*>(&hkUnmap),                reinterpret_cast<void**>(&o_Unmap),           "ctx::Unmap");
    HookEntry(ctx, kIdx_DrawIndexedInstanced, reinterpret_cast<void*>(&hkDrawIndexedInstanced), reinterpret_cast<void**>(&o_DrawIndexedInst), "ctx::DrawIndexedInstanced");
    HookEntry(ctx, kIdx_DrawInstanced,        reinterpret_cast<void*>(&hkDrawInstanced),        reinterpret_cast<void**>(&o_DrawInstanced),   "ctx::DrawInstanced");
    HookEntry(ctx, kIdx_DrawAuto,             reinterpret_cast<void*>(&hkDrawAuto),             reinterpret_cast<void**>(&o_DrawAuto),        "ctx::DrawAuto");
    HookEntry(ctx, kIdx_IASetVertexBuffers,   reinterpret_cast<void*>(&hkIASetVertexBuffers), reinterpret_cast<void**>(&o_IASetVBs),        "ctx::IASetVertexBuffers");
    HookEntry(ctx, kIdx_IASetIndexBuffer,     reinterpret_cast<void*>(&hkIASetIndexBuffer),   reinterpret_cast<void**>(&o_IASetIB),         "ctx::IASetIndexBuffer");
    HookEntry(ctx, kIdx_VSSetConstantBuffers,   reinterpret_cast<void*>(&hkVSSetConstantBuffers), reinterpret_cast<void**>(&o_VSSetCBs),        "ctx::VSSetConstantBuffers");
    HookEntry(ctx, kIdx_UpdateSubresource,    reinterpret_cast<void*>(&hkUpdateSubresource),    reinterpret_cast<void**>(&o_UpdateSubres),    "ctx::UpdateSubresource");
    HookEntry(ctx, kIdx_PSSetShader,          reinterpret_cast<void*>(&hkPSSetShader),          reinterpret_cast<void**>(&o_PSSetShader),     "ctx::PSSetShader");
    HookEntry(ctx, kIdx_RSSetState,           reinterpret_cast<void*>(&hkRSSetState),           reinterpret_cast<void**>(&o_RSSetState),      "ctx::RSSetState");
    HookEntry(ctx, kIdx_OMSetRenderTargets,   reinterpret_cast<void*>(&hkOMSetRenderTargets),   reinterpret_cast<void**>(&o_OMSetRTs),        "ctx::OMSetRenderTargets");
    HookEntry(ctx, kIdx_OMSetRTsAndUAVs,      reinterpret_cast<void*>(&hkOMSetRTsAndUAVs),      reinterpret_cast<void**>(&o_OMSetRTsUAVs),    "ctx::OMSetRTsAndUAVs");
    HookEntry(ctx, kIdx_RSSetViewports,       reinterpret_cast<void*>(&hkRSSetViewports),       reinterpret_cast<void**>(&o_RSSetViewports),  "ctx::RSSetViewports");
    HookEntry(ctx, kIdx_RSSetScissorRects,    reinterpret_cast<void*>(&hkRSSetScissorRects),    reinterpret_cast<void**>(&o_RSSetScissor),    "ctx::RSSetScissorRects");
    HookEntry(ctx, kIdx_CopySubresourceRegion,reinterpret_cast<void*>(&hkCopySubresourceRegion),reinterpret_cast<void**>(&o_CopySubres),      "ctx::CopySubresourceRegion");
}

void InstallFactoryHooks(ID3D11Device* dev) noexcept
{
    IDXGIDevice* dxgiDev = nullptr;
    if (FAILED(dev->QueryInterface(__uuidof(IDXGIDevice), reinterpret_cast<void**>(&dxgiDev))) || dxgiDev == nullptr)
    {
        Log::Warn("hooks", "QI(IDXGIDevice) failed — cannot reach factory");
        return;
    }
    IDXGIAdapter* adapter = nullptr;
    if (FAILED(dxgiDev->GetAdapter(&adapter)) || adapter == nullptr)
    {
        dxgiDev->Release();
        Log::Warn("hooks", "IDXGIDevice::GetAdapter failed");
        return;
    }
    IDXGIFactory* factory = nullptr;
    if (SUCCEEDED(adapter->GetParent(__uuidof(IDXGIFactory), reinterpret_cast<void**>(&factory))) && factory != nullptr)
    {
        Log::Info("hooks", "reached factory %p via device->adapter->parent", factory);
        HookEntry(factory, kIdx_FactoryCreateSwapChain, reinterpret_cast<void*>(&hkCreateSwapChain),
                  reinterpret_cast<void**>(&o_CreateSwapChain), "IDXGIFactory::CreateSwapChain");

        IDXGIFactory2* fac2 = nullptr;
        if (SUCCEEDED(factory->QueryInterface(__uuidof(IDXGIFactory2), reinterpret_cast<void**>(&fac2))) && fac2 != nullptr)
        {
            HookEntry(fac2, kIdx_Factory2CreateSCForHwnd, reinterpret_cast<void*>(&hkCreateSCForHwnd),
                      reinterpret_cast<void**>(&o_CreateSCForHwnd), "IDXGIFactory2::CreateSwapChainForHwnd");
            fac2->Release();
        }
        factory->Release();
    }
    adapter->Release();
    dxgiDev->Release();
}

} // namespace

void hooks::InitOnce() noexcept
{
    EnsureMinHook();
}

void hooks::Shutdown() noexcept
{
    if (InterlockedCompareExchange(&g_mhState, 0, 0) == 2)
    {
        MH_DisableHook(MH_ALL_HOOKS);
        RestoreVtblPatches(); // swap-chain slots before MinHook teardown
        MH_Uninitialize();
        InterlockedExchange(&g_mhState, 0);
        ai::Reset();
        det::Reset();
        hud::Reset();
        targets::Reset();
        retro::Reset();
        retro::DropUpscaleChain();
        if (g_sc2 != nullptr) { g_sc2->Release(); g_sc2 = nullptr; }
        ReleaseProxyBackbuffer();
        ReleaseStagingPool();
        if (g_flatPS != nullptr)  { g_flatPS->Release();  g_flatPS = nullptr; }
        if (g_wireRS != nullptr)  { g_wireRS->Release();  g_wireRS = nullptr; }
        // Owning captures — drop our refs, plus anything still queued.
        if (g_ctx != nullptr)    { g_ctx->Release();    g_ctx = nullptr; }
        if (g_device != nullptr) { g_device->Release(); g_device = nullptr; }
        DrainReleases();
        Log::Info("hooks", "MinHook uninitialized; retro objects released");
    }
}

void hooks::EnsureConfigLoaded() noexcept
{
    static volatile LONG state = 0; // 0 none, 1 loading, 2 done
    if (InterlockedCompareExchange(&state, 1, 0) == 0)
    {
        Log::Init(); // idempotent; resolves ModuleDir for the config path
        LoadConfigFile();
        InterlockedExchange(&state, 2);
        return;
    }
    while (InterlockedCompareExchange(&state, 0, 0) == 1)
        SwitchToThread();
}

void hooks::OnDeviceCreated(ID3D11Device* dev, ID3D11DeviceContext* ctx, IDXGISwapChain* sc) noexcept
{
    // The init thread races the game's first D3D11CreateDevice. Settings such as
    // RenderScale must be known BEFORE any swap chain/texture exists, otherwise
    // half the pipeline is native and half shrunk.
    EnsureConfigLoaded();
    if (!EnsureMinHook())
        return;
    if (dev != nullptr)
        InterlockedExchange(&g_scaleLatched, 1);

    Log::Info("hooks", "installing vtable hooks dev=%p ctx=%p sc=%p", dev, ctx, sc);
    // Capture the device/context pair as OWNING references. Games may create a
    // transient probe device (ctx==nullptr) and release it after caps queries —
    // a non-owning capture then dangles and faults at the next retro::Ensure
    // (observed in Watch Dogs 2: AV reading dev->lpVtbl in Ensure). A probe
    // device never displaces a live context-bearing capture.
    if (dev != nullptr && (g_ctx == nullptr || ctx != nullptr))
    {
        if (g_device != nullptr)
            DeferRelease(g_device); // destructor must not run on this call stack
        g_device = dev;
        dev->AddRef();
        if (ctx != nullptr)
        {
            if (g_ctx != nullptr)
                DeferRelease(g_ctx);
            g_ctx = ctx;
            ctx->AddRef();
        }
    }
    // D3D11CreateDeviceAndSwapChain hands us an existing swap chain — grab the
    // screen dims here since the factory detour never fired for it.
    if (sc != nullptr && g_scW == 0)
    {
        DXGI_SWAP_CHAIN_DESC sd = {};
        if (SUCCEEDED(sc->GetDesc(&sd)))
        {
            InterlockedExchange(&g_scIsFlip,
                                sd.SwapEffect >= DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL ? 1 : 0);
            if (sd.BufferDesc.Width > 0)
            {
                g_scW = sd.BufferDesc.Width;
                g_scH = sd.BufferDesc.Height;
                Log::Info("scale", "screen dims %ux%u (AndSwapChain path)", g_scW, g_scH);
            }
        }
    }
    // Chain already exists here — shrink the backbuffer via our own guarded
    // ResizeBuffers so the whole pipeline renders at 1/N like the factory path.
    const int s0 = RenderScale();
    if (sc != nullptr && g_scW > 0 && s0 > 1 && g_scIsFlip == 0)
    {
        retro::ReleaseBackbuffer();
        ai::ReleaseBackbuffer();
        InterlockedExchange(&g_inSCCreate, 1);
        const HRESULT shr = sc->ResizeBuffers(0, g_scW / static_cast<UINT>(s0),
                                              g_scH / static_cast<UINT>(s0),
                                              DXGI_FORMAT_UNKNOWN, 0);
        InterlockedExchange(&g_inSCCreate, 0);
        if (SUCCEEDED(shr))
            RegisterScaledBackbuffer(sc);
        else
            Log::Warn("hooks", "internal shrink ResizeBuffers failed 0x%08lX",
                      static_cast<unsigned long>(shr));
    }
    InterlockedExchange(&g_hooksActive, 1);
    InstallContextHooks(ctx);
    if (dev != nullptr)
    {
        InstallFactoryHooks(dev);
        HookEntry(dev, kIdx_DevCreateInputLayout, reinterpret_cast<void*>(&hkCreateInputLayout),
                  reinterpret_cast<void**>(&o_CreateInputLayout), "dev::CreateInputLayout");
        HookEntry(dev, kIdx_DevCreateTexture2D, reinterpret_cast<void*>(&hkCreateTexture2D),
                  reinterpret_cast<void**>(&o_CreateTexture2D), "dev::CreateTexture2D");
        HookEntry(dev, kIdx_DevCreateRenderTargetView, reinterpret_cast<void*>(&hkCreateRTV),
                  reinterpret_cast<void**>(&o_CreateRTV), "dev::CreateRenderTargetView");
        HookEntry(dev, kIdx_DevCreateDepthStencilView, reinterpret_cast<void*>(&hkCreateDSV),
                  reinterpret_cast<void**>(&o_CreateDSV), "dev::CreateDepthStencilView");
        HookEntry(dev, kIdx_DevCreatePixelShader, reinterpret_cast<void*>(&hkCreatePixelShader),
                  reinterpret_cast<void**>(&o_CreatePS), "dev::CreatePixelShader");
        HookEntry(dev, kIdx_DevCreateBuffer, reinterpret_cast<void*>(&hkCreateBuffer),
                  reinterpret_cast<void**>(&o_CreateBuffer), "dev::CreateBuffer");
    }
    if (sc != nullptr)
        InstallSwapChainHooks(sc);
}

void hooks::InstallSwapChainHooks(IDXGISwapChain* sc) noexcept
{
    if (sc == nullptr)
        return;
    // Direct slot patching — MinHook trampolines on these dxgi entries executed a
    // garbage relocated call on Yakuza 0 (see PatchVtblEntry comment).
    PatchVtblEntry(sc, kIdx_SwapChainPresent, reinterpret_cast<void*>(&hkPresent),
                   reinterpret_cast<void**>(&o_Present), "IDXGISwapChain::Present");
    PatchVtblEntry(sc, kIdx_SwapChainResizeBuffers, reinterpret_cast<void*>(&hkResizeBuffers),
                   reinterpret_cast<void**>(&o_ResizeBuffers), "IDXGISwapChain::ResizeBuffers");
    PatchVtblEntry(sc, kIdx_SwapChainGetBuffer, reinterpret_cast<void*>(&hkGetBuffer),
                   reinterpret_cast<void**>(&o_GetBuffer), "IDXGISwapChain::GetBuffer");
    // Present1 / ResizeBuffers1 bypass the slots above; only patch them when the
    // object actually implements the interface (the slots exist in its vtable).
    IDXGISwapChain1* sc1 = nullptr;
    if (SUCCEEDED(sc->QueryInterface(__uuidof(IDXGISwapChain1), reinterpret_cast<void**>(&sc1))) && sc1 != nullptr)
    {
        PatchVtblEntry(sc1, kIdx_SwapChain1Present1, reinterpret_cast<void*>(&hkPresent1),
                       reinterpret_cast<void**>(&o_Present1), "IDXGISwapChain1::Present1");
        sc1->Release();
    }
    IDXGISwapChain3* sc3 = nullptr;
    if (SUCCEEDED(sc->QueryInterface(__uuidof(IDXGISwapChain3), reinterpret_cast<void**>(&sc3))) && sc3 != nullptr)
    {
        PatchVtblEntry(sc3, kIdx_SwapChain3ResizeBuffers1, reinterpret_cast<void*>(&hkResizeBuffers1),
                       reinterpret_cast<void**>(&o_ResizeBuffers1), "IDXGISwapChain3::ResizeBuffers1");
        sc3->Release();
    }
}

void hooks::StatsTick() noexcept
{
    // Self-throttle: called at 50 ms cadence alongside HotkeyTick.
    static DWORD lastTick = 0;
    const DWORD now = GetTickCount();
    if (now - lastTick < 2000)
        return;
    lastTick = now;

    static LONGLONG lastPresents = -1, lastDraws = -1, lastMaps = -1, lastSkips = -1;
    static LARGE_INTEGER freq = {};
    if (freq.QuadPart == 0)
        QueryPerformanceFrequency(&freq);
    const LONGLONG p = InterlockedCompareExchange64(&g_presents, 0, 0);
    const LONGLONG d = InterlockedCompareExchange64(&g_draws, 0, 0);
    const LONGLONG m = InterlockedCompareExchange64(&g_maps, 0, 0);
    const LONGLONG sk = InterlockedCompareExchange64(&g_skips, 0, 0);
    // Fetch-and-reset the CPU frame-time accumulator for this window.
    const LONGLONG tickSum = InterlockedExchange64(&g_frameTickSum, 0);
    if (p == lastPresents && d == lastDraws && m == lastMaps)
        return;
    if (p > 0 || d > 0 || m > 0)
    {
        const LONGLONG frames = (lastPresents >= 0) ? p - lastPresents : 0;
        const double fps = frames / 2.0;
        const double cpuMs = (frames > 0 && freq.QuadPart > 0)
                           ? static_cast<double>(tickSum) * 1000.0 /
                             static_cast<double>(freq.QuadPart) / static_cast<double>(frames)
                           : 0.0;
        double aiMs = 0.0;
        long long aiRuns = 0;
        ai::TakeStats(&aiMs, &aiRuns);
        Log::Info("stats", "mode=%s fps=%.1f cpuFrameMs=%.2f presents=%lld draws=%lld maps=%lld skipped=%lld scale=%d aiMs=%.2f aiRuns=%lld",
                  hooks::ModeName(hooks::GetMode()), fps, cpuMs, p, d, m, sk,
                  static_cast<int>(g_renderScale), aiMs, aiRuns);
    }
    lastPresents = p; lastDraws = d; lastMaps = m; lastSkips = sk;
}

// ---- modes / hotkeys / config --------------------------------------------------

hooks::RetroMode hooks::GetMode() noexcept
{
    return static_cast<RetroMode>(g_mode);
}

void hooks::SetMode(RetroMode mode) noexcept
{
    if (mode == RetroMode::ConsoleWire)
        wire::EnsureConsole();
    const LONG m = static_cast<LONG>(mode);
    if (InterlockedExchange(&g_mode, m) != m)
        Log::Info("hooks", "mode -> %s", ModeName(mode));
}

const char* hooks::ModeName(RetroMode mode) noexcept
{
    switch (mode)
    {
    case RetroMode::Off:       return "Off";
    case RetroMode::NullPS:    return "NullPS";
    case RetroMode::FlatPS:    return "FlatPS";
    case RetroMode::Wireframe: return "Wireframe";
    case RetroMode::RetroFX:   return "RetroFX";
    case RetroMode::Ascii:     return "Ascii";
    case RetroMode::ConsoleWire: return "ConsoleWire";
    case RetroMode::Terminator: return "Terminator";
    }
    return "?";
}

void hooks::HotkeyTick() noexcept
{
    // Edge-triggered on real global key state (works fullscreen + windowed).
    static bool prevF8 = false, prevF9 = false, prevF10 = false, prevF11 = false, prevF12 = false;

    const bool f8  = (GetAsyncKeyState(VK_F8)  & 0x8000) != 0;
    const bool f9  = (GetAsyncKeyState(VK_F9)  & 0x8000) != 0;
    const bool f10 = (GetAsyncKeyState(VK_F10) & 0x8000) != 0;
    const bool f11 = (GetAsyncKeyState(VK_F11) & 0x8000) != 0;
    const bool f12 = (GetAsyncKeyState(VK_F12) & 0x8000) != 0;

    // Global key state: ignore presses aimed at other applications.
    DWORD fgPid = 0;
    const HWND fg = GetForegroundWindow();
    if (fg != nullptr)
        GetWindowThreadProcessId(fg, &fgPid);
    if (fgPid != GetCurrentProcessId())
    {
        prevF8 = f8; prevF9 = f9; prevF10 = f10; prevF11 = f11; prevF12 = f12;
        return;
    }

    if (f8 && !prevF8)
    {
        static const char* kPal[] = { "green", "amber", "mono", "cyan", "color" };
        const int pal = (retro::GetPalette() + 1) % 5;
        retro::SetPalette(pal);
        Log::Info("retro", "palette -> %s", kPal[pal]);
    }

    if (f9 && !prevF9)
    {
        // User-facing cycle: Off -> RetroFX -> Ascii -> Terminator. Debug modes (NullPS, FlatPS,
        // Wireframe, ConsoleWire) are still selectable via config "Mode". Terminator is always
        // offered: without ORT (or a model) its base layer runs the HLSL fallback.
        static const int kCycle[] = { 0, 4, 5, 7 };
        constexpr int kCycleN = static_cast<int>(sizeof(kCycle) / sizeof(kCycle[0]));
        int i = 0;
        for (; i < kCycleN; ++i)
            if (kCycle[i] == g_mode)
                break;
        SetMode(static_cast<RetroMode>(i < kCycleN ? kCycle[(i + 1) % kCycleN] : 0));
    }
    if (f10 && !prevF10)
    {
        if (g_hooksActive != 0)
        {
            MH_DisableHook(MH_ALL_HOOKS);
            InterlockedExchange(&g_hooksActive, 0);
            Log::Warn("hooks", "PANIC: all hooks disabled");
        }
        else
        {
            MH_EnableHook(MH_ALL_HOOKS);
            InterlockedExchange(&g_hooksActive, 1);
            Log::Info("hooks", "hooks re-enabled");
        }
    }
    if (f11 && !prevF11)
    {
        Log::Info("hooks", "reloading retro.json");
        LoadConfigFile();
    }
    if (f12 && !prevF12)
    {
        // RenderDoc on-demand capture: the API lives in renderdoc.dll when the
        // process is launched under RenderDoc — absent otherwise.
        struct RdocAPI
        {
            void* getVersion; void* setPath; void* getPath;
            void* getNum;     void* getCap;
            void (WINAPI* trigger)(void);
        };
        const HMODULE rd = GetModuleHandleA("renderdoc.dll");
        if (rd != nullptr)
        {
            typedef int (WINAPI* GetApiFn)(int, void**);
            const GetApiFn getApi = reinterpret_cast<GetApiFn>(GetProcAddress(rd, "RENDERDOC_GetAPI"));
            RdocAPI* api = nullptr;
            if (getApi != nullptr && getApi(10100, reinterpret_cast<void**>(&api)) == 1 && api != nullptr)
            {
                api->trigger();
                Log::Info("hooks", "RenderDoc capture triggered");
            }
        }
        else
            Log::Info("hooks", "F12 ignored — renderdoc.dll not loaded");
    }
    prevF8 = f8; prevF9 = f9; prevF10 = f10; prevF11 = f11; prevF12 = f12;
}

void hooks::LoadConfigFile() noexcept
{
    char path[MAX_PATH] = {};
    _snprintf_s(path, _TRUNCATE, "%s\\retro.json", Log::ModuleDir());

    FILE* f = nullptr;
    if (fopen_s(&f, path, "rb") != 0 || f == nullptr)
    {
        Log::Info("hooks", "no retro.json (%s) — mode stays %s", path, ModeName(GetMode()));
        return;
    }
    constexpr size_t kCfgMax = 32768;
    struct HeapBuf { char* p; ~HeapBuf() { if (p != nullptr) HeapFree(GetProcessHeap(), 0, p); } } heap{
        static_cast<char*>(HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, kCfgMax)) };
    char* const buf = heap.p;
    if (buf == nullptr)
    {
        fclose(f);
        return;
    }
    const size_t n = fread(buf, 1, kCfgMax - 1, f);
    fclose(f);
    if (n == 0)
        return;
    // Preset blocks reuse top-level key names ("Crt", "Vignette"): lift them out first so
    // the flat strstr scan below never sees them.
    char lensBlock[1024] = {};
    cfg::ExtractObject(buf, "TerminatorLens", lensBlock, sizeof(lensBlock));

    // Minimal scan: find "Mode" then the quoted value. Good enough for our own file.
    const char* m = strstr(buf, "\"Mode\"");
    if (m == nullptr)
        return;
    m = strchr(m + 6, ':');
    if (m == nullptr)
        return;
    m = strchr(m, '"');
    if (m == nullptr)
        return;
    const char* end = strchr(m + 1, '"');
    if (end == nullptr || end - m > 32)
        return;
    char name[32] = {};
    memcpy(name, m + 1, static_cast<size_t>(end - m - 1));

    static const char* kNames[] = { "Off", "NullPS", "FlatPS", "Wireframe", "RetroFX", "Ascii", "ConsoleWire", "Terminator" };
    if (_stricmp(name, "Neural") == 0) // pre-Terminator configs
        memcpy(name, "Terminator", sizeof("Terminator"));
    for (int i = 0; i < 8; ++i)
    {
        if (_stricmp(name, kNames[i]) == 0)
        {
            SetMode(static_cast<RetroMode>(i));
            Log::Info("hooks", "config mode = %s", kNames[i]);
            break;
        }
    }

    // "DumpCBs": N — recon budget for Phase 5 matrix-slot hunting.
    const char* d = strstr(buf, "\"DumpCBs\"");
    if (d != nullptr)
    {
        d = strchr(d + 9, ':');
        if (d != nullptr)
        {
            const int budget = atoi(d + 1);
            if (budget > 0)
            {
                InterlockedExchange(&g_dumpCBs, budget);
                Log::Info("recon", "DumpCBs budget = %d", budget);
            }
        }
    }

    // "DumpMesh": N — Phase 7 geometry readback budget.
    const char* dm = strstr(buf, "\"DumpMesh\"");
    if (dm != nullptr)
    {
        dm = strchr(dm + 10, ':');
        if (dm != nullptr)
        {
            const int budget = atoi(dm + 1);
            if (budget > 0)
            {
                InterlockedExchange(&g_dumpMesh, budget);
                Log::Info("mesh", "DumpMesh budget = %d", budget);
            }
        }
    }

    // Console wireframe params (nested under "Output" in retro.json — strstr finds them).
    const char* cw = strstr(buf, "\"ConsoleWidth\"");
    const char* ch = strstr(buf, "\"ConsoleHeight\"");
    const char* hz = strstr(buf, "\"ConsoleHz\"");
    const char* wd = strstr(buf, "\"WireDraws\"");
    if (cw != nullptr) { const char* c = strchr(cw + 14, ':'); if (c != nullptr) { const int v = atoi(c + 1); if (v >= 40 && v <= 400) wire::g_consoleW = v; } }
    if (ch != nullptr) { const char* c = strchr(ch + 15, ':'); if (c != nullptr) { const int v = atoi(c + 1); if (v >= 10 && v <= 200) wire::g_consoleH = v; } }
    if (hz != nullptr) { const char* c = strchr(hz + 11, ':'); if (c != nullptr) { const int v = atoi(c + 1); if (v >= 1 && v <= 120) wire::g_consoleHz = v; } }
    if (wd != nullptr) { const char* c = strchr(wd + 10, ':'); if (c != nullptr) { const int v = atoi(c + 1); if (v >= 1 && v <= 64) wire::g_maxDrawsPerFrame = v; } }
    const char* wc = strstr(buf, "\"WireCull\"");
    if (wc != nullptr) { const char* c = strchr(wc + 10, ':'); if (c != nullptr) { const int v = atoi(c + 1); wire::g_wireCull = (v != 0) ? 1 : 0; } }

    // "PixelPx" / "CellPx" — retro pass tunables.
    const char* px = strstr(buf, "\"PixelPx\"");
    const char* cx = strstr(buf, "\"CellPx\"");
    if (px != nullptr || cx != nullptr)
    {
        float pixelPx = 4.0f, cellPx = 8.0f;
        if (px != nullptr) { const char* c = strchr(px + 9, ':');  if (c != nullptr) pixelPx = static_cast<float>(atof(c + 1)); }
        if (cx != nullptr) { const char* c = strchr(cx + 8, ':');  if (c != nullptr) cellPx  = static_cast<float>(atof(c + 1)); }
        if (pixelPx >= 1.0f && pixelPx <= 64.0f && cellPx >= 1.0f && cellPx <= 32.0f)
        {
            retro::SetTunables(pixelPx, cellPx);
            Log::Info("retro", "tunables: pixelPx=%.0f cellPx=%.0f", pixelPx, cellPx);
        }
    }

    // "Crt" / "Vignette" — RetroFX strength tunables.
    const char* cr = strstr(buf, "\"Crt\"");
    const char* vg = strstr(buf, "\"Vignette\"");
    if (cr != nullptr || vg != nullptr)
    {
        float crt = 0.12f, vignette = 0.35f;
        if (cr != nullptr) { const char* c = strchr(cr + 5, ':');  if (c != nullptr) crt      = static_cast<float>(atof(c + 1)); }
        if (vg != nullptr) { const char* c = strchr(vg + 10, ':'); if (c != nullptr) vignette = static_cast<float>(atof(c + 1)); }
        if (crt >= 0.0f && crt <= 0.6f && vignette >= 0.0f && vignette <= 1.0f)
        {
            retro::SetCrt(crt, vignette);
            Log::Info("retro", "crt=%.2f vignette=%.2f", crt, vignette);
        }
    }

    // "Palette" — Ascii phosphor theme by name.
    const char* pl = strstr(buf, "\"Palette\"");
    if (pl != nullptr)
    {
        const char* c = strchr(pl + 9, ':');
        if (c != nullptr)
        {
            const char* v = c + 1;
            int pal = 0;
            if      (strstr(v, "\"amber\"") != nullptr) pal = 1;
            else if (strstr(v, "\"mono\"")  != nullptr) pal = 2;
            else if (strstr(v, "\"cyan\"")  != nullptr) pal = 3;
            else if (strstr(v, "\"color\"") != nullptr) pal = 4;
            retro::SetPalette(pal);
            Log::Info("retro", "palette=%d", pal);
        }
    }

    // ---- Phase 12: perf options ------------------------------------------------
    // "RenderScale": N — scene renders at 1/N resolution. Only effective when set
    // before the device/swap chain exists (i.e. at launch, not on F11 reload).
    const char* rs = strstr(buf, "\"RenderScale\"");
    if (rs != nullptr)
    {
        const char* c = strchr(rs + 13, ':');
        if (c != nullptr)
        {
            const int v = atoi(c + 1);
            if (v >= 1 && v <= 8 && g_scaleLatched != 0 && v != g_renderScale)
                Log::Warn("scale", "RenderScale %d -> %d ignored: takes effect on next launch",
                          static_cast<int>(g_renderScale), v);
            else if (v >= 1 && v <= 8)
            {
                InterlockedExchange(&g_renderScale, v);
                retro::SetRenderScale(v);
                Log::Info("scale", "RenderScale=%d", v);
            }
        }
    }

    // "GpuBurn": N — N heavy fullscreen passes per frame; emulates a weaker GPU.
    const char* gb = strstr(buf, "\"GpuBurn\"");
    if (gb != nullptr)
    {
        const char* c = strchr(gb + 9, ':');
        if (c != nullptr)
        {
            const int v = atoi(c + 1);
            InterlockedExchange(&g_gpuBurn, v < 0 ? 0 : (v > 64 ? 64 : v));
            retro::SetGpuBurn(v < 0 ? 0 : (v > 64 ? 64 : v));
            if (v > 0) Log::Info("perf", "GpuBurn=%d passes/frame", v);
        }
    }

    const char* pc = strstr(buf, "\"PatchScreenCB\"");
    if (pc != nullptr)
    {
        const char* c = strchr(pc + 15, ':');
        if (c != nullptr) { InterlockedExchange(&g_patchScreenCB, atoi(c + 1) != 0 ? 1 : 0); }
    }

    const char* ss = strstr(buf, "\"SkipShadows\"");
    if (ss != nullptr)
    {
        const char* c = strchr(ss + 13, ':');
        if (c != nullptr) { InterlockedExchange(&g_skipShadows, atoi(c + 1) != 0 ? 1 : 0); }
    }

    const char* pd = strstr(buf, "\"PassDump\"");
    if (pd != nullptr)
    {
        const char* c = strchr(pd + 10, ':');
        if (c != nullptr)
        {
            const int v = atoi(c + 1);
            if (v > 0)
            {
                memset(g_psLogged, 0, sizeof(g_psLogged));
                InterlockedExchange(&g_passDump, v);
                Log::Info("pass", "PassDump budget = %d frames", v);
            }
        }
    }

    // ShaderStrip block: "SkipMaxIndexCount" / "SkipOrthographic".
    const char* sm = strstr(buf, "\"SkipMaxIndexCount\"");
    if (sm != nullptr)
    {
        const char* c = strchr(sm + 19, ':');
        if (c != nullptr) { const int v = atoi(c + 1); if (v >= 0 && v <= 4096) InterlockedExchange(&g_skipMaxIndex, v); }
    }
    const char* so = strstr(buf, "\"SkipOrthographic\"");
    if (so != nullptr)
    {
        const char* c = strchr(so + 18, ':');
        if (c != nullptr)
        {
            InterlockedExchange(&g_skipOrtho,
                strstr(c, "true") != nullptr || atoi(c + 1) != 0 ? 1 : 0);
        }
    }

    // "SkipPS": ["0xhash", ...] — pixel-shader bytecode hashes whose draws are dropped.
    const char* sp = strstr(buf, "\"SkipPS\"");
    if (sp != nullptr)
    {
        const char* c = strchr(sp + 8, '[');
        const char* e = (c != nullptr) ? strchr(c, ']') : nullptr;
        g_skipPsN = 0;
        while (c != nullptr && e != nullptr)
        {
            const char* h = strstr(c, "0x");
            if (h == nullptr || h >= e || g_skipPsN >= 64)
                break;
            g_skipPs[g_skipPsN++] = _strtoui64(h + 2, nullptr, 16);
            c = const_cast<char*>(h) + 2;
            if (c >= e)
                break;
        }
        if (g_skipPsN > 0)
            Log::Info("pass", "SkipPS: %u shader hashes blocklisted", g_skipPsN);
    }

    // Terminator base (ONNX): "AiModel", "AiWidth"/"AiHeight", "AiTemporal", "AiBlend", "AiStride".
    // AiDevice is accepted but unused — ORT-DML always runs on the game's adapter.
    {
        char model[MAX_PATH] = "retro_ai.onnx";
        int aiW = 320, aiH = 180, temporal = 0, stride = 1;
        float blend = 0.5f;
        const char* am = strstr(buf, "\"AiModel\"");
        if (am != nullptr)
        {
            const char* q = strchr(am + 9, ':');
            q = (q != nullptr) ? strchr(q, '"') : nullptr;
            const char* qe = (q != nullptr) ? strchr(q + 1, '"') : nullptr;
            if (qe != nullptr && qe - q - 1 > 0 && qe - q - 1 < MAX_PATH)
            {
                memset(model, 0, sizeof(model));
                memcpy(model, q + 1, static_cast<size_t>(qe - q - 1));
            }
        }
        auto readInt = [&](const char* key, int& v) {
            const char* k = strstr(buf, key);
            const char* c = (k != nullptr) ? strchr(k + strlen(key), ':') : nullptr;
            if (c != nullptr) v = atoi(c + 1);
        };
        readInt("\"AiWidth\"", aiW);
        readInt("\"AiHeight\"", aiH);
        readInt("\"AiTemporal\"", temporal);
        readInt("\"AiStride\"", stride);
        const char* ab = strstr(buf, "\"AiBlend\"");
        const char* abc = (ab != nullptr) ? strchr(ab + 9, ':') : nullptr;
        if (abc != nullptr) blend = static_cast<float>(atof(abc + 1));
        ai::SetConfig(model, aiW, aiH, temporal, blend, stride);
    }

    // Terminator Vision: HUD / lens / look. Every key optional, all hot-reloadable (F11).
    {
        hud::Config hc;
        cfg::ReadBool(buf, "TermHud", &hc.enabled);
        cfg::ReadBool(buf, "TermHudText", &hc.text);
        cfg::ReadBool(buf, "TermBrackets", &hc.brackets);
        cfg::ReadBool(buf, "TermReticle", &hc.reticle);
        float speed = 1.0f;
        if (cfg::ReadFloat(buf, "TermHudSpeed", &speed) && speed >= 0.0f && speed <= 10.0f)
            hc.speed = speed;
        hud::SetConfig(hc);
        bool lens = true;
        cfg::ReadBool(buf, "TermLens", &lens);
        bool base = true;
        cfg::ReadBool(buf, "TermBase", &base);
        bool baseOnnx = true;
        cfg::ReadBool(buf, "TermBaseOnnx", &baseOnnx);
        int maxTargets = 6;
        cfg::ReadInt(buf, "HudMaxTargets", &maxTargets);
        maxTargets = maxTargets < 0 ? 0 : (maxTargets > 16 ? 16 : maxTargets);
        float gain = 1.0f;
        cfg::ReadFloat(buf, "TermEdgeGain", &gain);
        gain = gain < 0.0f ? 0.0f : (gain > 8.0f ? 8.0f : gain);
        float tint[3] = { 1.0f, 0.0f, 0.0f };
        cfg::ReadFloat3(buf, "TermTint", tint);
        for (float& t : tint)
            t = t < 0.0f ? 0.0f : (t > 2.0f ? 2.0f : t);
        retro::SetTermLook(gain, tint);
        ai::SetLook(gain, tint);
        float crt = 0.10f, vig = 0.45f, scan = 0.35f;
        cfg::ReadFloat(lensBlock, "Crt", &crt);
        cfg::ReadFloat(lensBlock, "Vignette", &vig);
        cfg::ReadFloat(lensBlock, "Scanlines", &scan);
        crt  = crt  < 0.0f ? 0.0f : (crt  > 0.6f ? 0.6f : crt);
        vig  = vig  < 0.0f ? 0.0f : (vig  > 1.0f ? 1.0f : vig);
        scan = scan < 0.0f ? 0.0f : (scan > 1.0f ? 1.0f : scan);
        retro::SetLens(crt, vig, scan);
        InterlockedExchange(&g_termLens, lens ? 1 : 0);
        InterlockedExchange(&g_termBase, base ? 1 : 0);
        InterlockedExchange(&g_termBaseOnnx, baseOnnx ? 1 : 0);
        InterlockedExchange(&g_hudMaxTargets, maxTargets);
        InterlockedExchange(&g_termBrackets, (hc.enabled && hc.brackets && maxTargets > 0) ? 1 : 0);
        g_termFallbackLogged = false; // re-report the base-layer source after a reload
        Log::Info("term", "hud=%d text=%d brackets=%d reticle=%d speed=%.2f lens=%d maxTargets=%d "
                  "edgeGain=%.2f tint=%.2f,%.2f,%.2f lens(crt=%.2f vig=%.2f scan=%.2f)",
                  hc.enabled, hc.text, hc.brackets, hc.reticle, hc.speed, lens, maxTargets,
                  gain, tint[0], tint[1], tint[2], crt, vig, scan);
    }

    // Terminator detector (person/weapon boxes via a second ONNX session — YOLOv8-style
    // output). DetModel empty = off (mesh-AABB scan keeps feeding brackets).
    {
        char detModel[260] = "yolov8n.onnx";
        cfg::ReadString(buf, "DetModel", detModel, sizeof(detModel));
        char detModel2[260] = "";
        cfg::ReadString(buf, "DetModel2", detModel2, sizeof(detModel2));
        float detConf = 0.35f;
        cfg::ReadFloat(buf, "DetConf", &detConf);
        int detStride = 2;
        cfg::ReadInt(buf, "DetStride", &detStride);
        bool detSight = true;
        cfg::ReadBool(buf, "DetSight", &detSight);
        float detLen = 0.55f;
        cfg::ReadFloat(buf, "DetSightLen", &detLen);
        char classes[128] = "person";
        cfg::ReadString(buf, "DetClasses", classes, sizeof(classes));
        char weapons[128] = "";
        cfg::ReadString(buf, "DetWeapon", weapons, sizeof(weapons));
        char classes2[128] = "";
        cfg::ReadString(buf, "DetClasses2", classes2, sizeof(classes2));
        char weapons2[128] = "";
        cfg::ReadString(buf, "DetWeapon2", weapons2, sizeof(weapons2));
        int detDump = 0;
        cfg::ReadInt(buf, "DetDump", &detDump);
        char detModel3[260] = "";
        cfg::ReadString(buf, "DetModel3", detModel3, sizeof(detModel3));
        float detClsConf = 0.80f;
        cfg::ReadFloat(buf, "DetClsConf", &detClsConf);
        det::SetConfig(detModel, detModel2, detConf, detStride, detSight, detLen,
                       classes, weapons, classes2, weapons2, detDump, detModel3, detClsConf);
    }
}
