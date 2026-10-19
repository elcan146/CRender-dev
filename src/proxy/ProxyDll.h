// ProxyDll.h — real-d3d11 resolver and captured-object accessors (Phase 2 pass-through).
#pragma once

#include <Windows.h>
#include <d3d11.h>
#include <dxgi.h>

// Function-pointer type matching the SDK signature (d3d11.h ships PFN_D3D11_CREATE_DEVICE
// but not the AndSwapChain variant on all SDKs — declare it ourselves).
using FnD3D11CreateDeviceAndSwapChain = HRESULT(WINAPI*)(
    IDXGIAdapter*, D3D_DRIVER_TYPE, HMODULE, UINT,
    const D3D_FEATURE_LEVEL*, UINT, UINT,
    const DXGI_SWAP_CHAIN_DESC*, IDXGISwapChain**, ID3D11Device**,
    D3D_FEATURE_LEVEL*, ID3D11DeviceContext**);

namespace proxy
{

// Idempotent lazy resolver: LoadLibrary(SysWOW64\d3d11.dll) + GetProcAddress every export.
// Safe on any normal thread — must NOT be called from DllMain (loader lock).
BOOL EnsureRealD3D11() noexcept;

// Objects captured by the wrapped creation calls. Raw, non-owning — Phase 3 vtable hooks
// take over ownership tracking.
ID3D11Device*        CapturedDevice() noexcept;
ID3D11DeviceContext* CapturedImmediateContext() noexcept;
IDXGISwapChain*      CapturedSwapChain() noexcept;

void SetCaptured(ID3D11Device* dev, ID3D11DeviceContext* ctx, IDXGISwapChain* sc) noexcept;

} // namespace proxy

// asm-callable resolver used by naked forwarder stubs.
extern "C" void EnsureRealD3D11Impl();

// Resolved pointers for the wrapped exports.
extern PFN_D3D11_CREATE_DEVICE          g_pfnD3D11CreateDevice;
extern FnD3D11CreateDeviceAndSwapChain  g_pfnD3D11CreateDeviceAndSwapChain;

// One FARPROC slot per pass-through export (defined in ProxyExports.cpp).
#define FWD_ENTRY(Name) extern "C" void* g_fwd_##Name;
#include "FwdList.inc"
#undef FWD_ENTRY
