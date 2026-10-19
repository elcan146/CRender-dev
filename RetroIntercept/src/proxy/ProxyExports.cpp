// ProxyExports.cpp — the d3d11.dll export surface.
//   - 49 pass-through exports: naked jmp stubs to slots resolved from the real DLL.
//     Signature-free by design (works regardless of arg list; stdcall stack intact).
//   - 2 wrapped exports: D3D11CreateDevice / D3D11CreateDeviceAndSwapChain log args and
//     capture the returned device/context/swap-chain for Phase 3 vtable hooking.
#include "ProxyDll.h"
#include "../core/Log.h"
#include "../d3d11/HookManager.h"

// ---- forwarder slots ----------------------------------------------------------
#define FWD_ENTRY(Name) extern "C" void* g_fwd_##Name = nullptr;
#include "FwdList.inc"
#undef FWD_ENTRY

// ---- naked trampolines (x86) --------------------------------------------------
// If the slot is still null, resolve first (covers callers that reach an export before
// the init thread finished). Then tail-jump: original return address and args untouched.
// On x64 MSVC has no __asm — equivalent stubs are generated in fwd_stubs_x64.asm.
#ifndef _WIN64
#define FWD_STUB(Name)                                            \
    extern "C" __declspec(naked) void Name()                      \
    {                                                             \
        __asm cmp dword ptr [g_fwd_##Name], 0                     \
        __asm jne short go_##Name                                 \
        __asm call EnsureRealD3D11Impl                            \
        __asm go_##Name:                                          \
        __asm jmp dword ptr [g_fwd_##Name]                        \
    }

#define FWD_ENTRY(Name) FWD_STUB(Name)
#include "FwdList.inc"
#undef FWD_ENTRY
#endif

// ---- wrapped exports ----------------------------------------------------------
extern "C" HRESULT WINAPI D3D11CreateDevice(
    IDXGIAdapter* pAdapter,
    D3D_DRIVER_TYPE DriverType,
    HMODULE Software,
    UINT Flags,
    const D3D_FEATURE_LEVEL* pFeatureLevels,
    UINT FeatureLevels,
    UINT SDKVersion,
    ID3D11Device** ppDevice,
    D3D_FEATURE_LEVEL* pFeatureLevel,
    ID3D11DeviceContext** ppImmediateContext)
{
    proxy::EnsureRealD3D11();
    if (g_pfnD3D11CreateDevice == nullptr)
    {
        Log::Error("proxy", "D3D11CreateDevice called but real DLL unresolved");
        return E_FAIL;
    }

    const HRESULT hr = g_pfnD3D11CreateDevice(pAdapter, DriverType, Software, Flags,
                                              pFeatureLevels, FeatureLevels, SDKVersion,
                                              ppDevice, pFeatureLevel, ppImmediateContext);
    if (SUCCEEDED(hr))
    {
        Log::Info("proxy",
                  "D3D11CreateDevice(drv=%d flags=0x%08lX sdk=%u) -> dev=%p ctx=%p",
                  static_cast<int>(DriverType), static_cast<unsigned long>(Flags), SDKVersion,
                  ppDevice ? *ppDevice : nullptr,
                  ppImmediateContext ? *ppImmediateContext : nullptr);
        proxy::SetCaptured(ppDevice ? *ppDevice : nullptr,
                           ppImmediateContext ? *ppImmediateContext : nullptr, nullptr);
        hooks::OnDeviceCreated(ppDevice ? *ppDevice : nullptr,
                               ppImmediateContext ? *ppImmediateContext : nullptr, nullptr);
    }
    else
    {
        Log::Warn("proxy", "D3D11CreateDevice failed hr=0x%08lX", static_cast<unsigned long>(hr));
    }
    return hr;
}

extern "C" HRESULT WINAPI D3D11CreateDeviceAndSwapChain(
    IDXGIAdapter* pAdapter,
    D3D_DRIVER_TYPE DriverType,
    HMODULE Software,
    UINT Flags,
    const D3D_FEATURE_LEVEL* pFeatureLevels,
    UINT FeatureLevels,
    UINT SDKVersion,
    const DXGI_SWAP_CHAIN_DESC* pSwapChainDesc,
    IDXGISwapChain** ppSwapChain,
    ID3D11Device** ppDevice,
    D3D_FEATURE_LEVEL* pFeatureLevel,
    ID3D11DeviceContext** ppImmediateContext)
{
    proxy::EnsureRealD3D11();
    if (g_pfnD3D11CreateDeviceAndSwapChain == nullptr)
    {
        Log::Error("proxy", "D3D11CreateDeviceAndSwapChain called but real DLL unresolved");
        return E_FAIL;
    }

    const HRESULT hr = g_pfnD3D11CreateDeviceAndSwapChain(
        pAdapter, DriverType, Software, Flags, pFeatureLevels, FeatureLevels, SDKVersion,
        pSwapChainDesc, ppSwapChain, ppDevice, pFeatureLevel, ppImmediateContext);
    if (SUCCEEDED(hr))
    {
        Log::Info("proxy",
                  "D3D11CreateDeviceAndSwapChain(drv=%d flags=0x%08lX sdk=%u w=%u h=%u)"
                  " -> sc=%p dev=%p ctx=%p",
                  static_cast<int>(DriverType), static_cast<unsigned long>(Flags), SDKVersion,
                  pSwapChainDesc ? pSwapChainDesc->BufferDesc.Width : 0u,
                  pSwapChainDesc ? pSwapChainDesc->BufferDesc.Height : 0u,
                  ppSwapChain ? *ppSwapChain : nullptr,
                  ppDevice ? *ppDevice : nullptr,
                  ppImmediateContext ? *ppImmediateContext : nullptr);
        proxy::SetCaptured(ppDevice ? *ppDevice : nullptr,
                           ppImmediateContext ? *ppImmediateContext : nullptr,
                           ppSwapChain ? *ppSwapChain : nullptr);
        hooks::OnDeviceCreated(ppDevice ? *ppDevice : nullptr,
                               ppImmediateContext ? *ppImmediateContext : nullptr,
                               ppSwapChain ? *ppSwapChain : nullptr);
    }
    else
    {
        Log::Warn("proxy", "D3D11CreateDeviceAndSwapChain failed hr=0x%08lX",
                  static_cast<unsigned long>(hr));
    }
    return hr;
}
