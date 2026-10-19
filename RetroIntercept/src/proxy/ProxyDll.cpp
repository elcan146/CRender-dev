// ProxyDll.cpp — resolves the real system d3d11.dll and fills the export table.
//
// Why not .def forwarders ("X=d3d11.X")? Our module is also named d3d11.dll — a forwarder
// would resolve to ourselves and loop. Why not a renamed copy of the system DLL? That
// means redistributing a version-locked system binary. Instead: 49 naked jmp stubs hop to
// addresses resolved at runtime from the real SysWOW64 d3d11.dll via absolute path.
#include "ProxyDll.h"
#include "../core/Log.h"

PFN_D3D11_CREATE_DEVICE         g_pfnD3D11CreateDevice = nullptr;
FnD3D11CreateDeviceAndSwapChain g_pfnD3D11CreateDeviceAndSwapChain = nullptr;

namespace
{

HMODULE g_real = nullptr;
volatile LONG g_loadState = 0; // 0 = not started, 1 = loading, 2 = ready, 3 = failed

// Observed (non-owning) pointers for Phase 3 vtable hooking.
ID3D11Device*        g_device = nullptr;
ID3D11DeviceContext* g_context = nullptr;
IDXGISwapChain*      g_swapChain = nullptr;

struct NameSlot
{
    const char* name;
    void** slot;
};

BOOL LoadRealAndResolve() noexcept
{
    wchar_t path[MAX_PATH] = {};
    const UINT len = GetSystemDirectoryW(path, MAX_PATH); // WOW64 -> SysWOW64: correct 32-bit dll
    if (len == 0 || len >= MAX_PATH - 16)
    {
        Log::Error("proxy", "GetSystemDirectoryW failed, err=%lu", GetLastError());
        return FALSE;
    }
    lstrcatW(path, L"\\d3d11.dll");

    g_real = LoadLibraryW(path);
    if (g_real == nullptr)
    {
        Log::Error("proxy", "LoadLibraryW(%ls) failed, err=%lu", path, GetLastError());
        return FALSE;
    }
    Log::Info("proxy", "real d3d11 loaded from %ls (hmod=%p)", path, g_real);

    NameSlot table[] = {
#define FWD_ENTRY(Name) { #Name, reinterpret_cast<void**>(&g_fwd_##Name) },
#include "FwdList.inc"
#undef FWD_ENTRY
        { "D3D11CreateDevice",            reinterpret_cast<void**>(&g_pfnD3D11CreateDevice) },
        { "D3D11CreateDeviceAndSwapChain", reinterpret_cast<void**>(&g_pfnD3D11CreateDeviceAndSwapChain) },
    };

    BOOL ok = TRUE;
    for (const NameSlot& e : table)
    {
        *e.slot = reinterpret_cast<void*>(GetProcAddress(g_real, e.name));
        if (*e.slot == nullptr)
        {
            Log::Error("proxy", "GetProcAddress(%s) failed, err=%lu", e.name, GetLastError());
            ok = FALSE;
        }
    }
    return ok;
}

} // namespace

extern "C" void EnsureRealD3D11Impl()
{
    const LONG prev = InterlockedCompareExchange(&g_loadState, 1, 0);
    if (prev == 0)
    {
        const BOOL ok = LoadRealAndResolve();
        InterlockedExchange(&g_loadState, ok ? 2 : 3);
    }
    else
    {
        // Another thread is resolving; wait for it to finish.
        while (InterlockedCompareExchange(&g_loadState, 0, 0) == 1)
            SwitchToThread();
    }
}

BOOL proxy::EnsureRealD3D11() noexcept
{
    EnsureRealD3D11Impl();
    return InterlockedCompareExchange(&g_loadState, 0, 0) == 2;
}

ID3D11Device* proxy::CapturedDevice() noexcept { return g_device; }

ID3D11DeviceContext* proxy::CapturedImmediateContext() noexcept { return g_context; }

IDXGISwapChain* proxy::CapturedSwapChain() noexcept { return g_swapChain; }

void proxy::SetCaptured(ID3D11Device* dev, ID3D11DeviceContext* ctx, IDXGISwapChain* sc) noexcept
{
    if (dev != nullptr)
        g_device = dev;
    if (ctx != nullptr)
        g_context = ctx;
    if (sc != nullptr)
        g_swapChain = sc;
}
