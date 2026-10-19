// Entry point: minimal DllMain, real init deferred to a worker thread (loader-lock rules).
#include <Windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#include "core/Log.h"
#include "proxy/ProxyDll.h"
#include "d3d11/HookManager.h"

namespace
{

volatile LONG g_running = 1;
HANDLE g_initThread = nullptr;
PVOID  g_veh = nullptr;

// --- crash forensics ---------------------------------------------------------
// Engines with their own SEH/UEF (Ubisoft's client crash reporter) swallow
// faults and exit silently — no WER event, no dump, and our DLL never sees
// DLL_PROCESS_DETACH. A vectored handler runs BEFORE the UEF and cannot be
// suppressed, so we log the faulting address + owning module ourselves.
struct ModRange { const char* tag; UINT_PTR lo, hi; };
ModRange      g_mods[256] = {};
char          g_tags[256][24] = {}; // stable name storage — points into nothing else
volatile LONG g_modCount = 0;
UINT_PTR      g_seenIps[64] = {};
volatile LONG g_seenCount = 0;

void SnapshotModule(HMODULE m, const char* tag) noexcept
{
    if (m == nullptr)
        return;
    MODULEINFO mi = {};
    if (GetModuleInformation(GetCurrentProcess(), m, &mi, sizeof(mi)) == FALSE)
        return;
    const UINT_PTR base = reinterpret_cast<UINT_PTR>(mi.lpBaseOfDll);
    const LONG n = g_modCount;
    for (LONG i = 0; i < n; ++i)
        if (g_mods[i].lo == base)
            return;
    if (n < static_cast<LONG>(sizeof(g_mods) / sizeof(g_mods[0])))
    {
        lstrcpynA(g_tags[n], tag != nullptr ? tag : "mod", sizeof(g_tags[n]));
        g_mods[n] = { g_tags[n], base, base + mi.SizeOfImage };
        InterlockedExchange(&g_modCount, n + 1);
    }
}

// Resolve an address to "tag+0xRVA" into out (must hold >= 64 bytes).
void ResolveIp(UINT_PTR ip, char* out) noexcept
{
    const LONG n = g_modCount;
    for (LONG i = 0; i < n; ++i)
        if (ip >= g_mods[i].lo && ip < g_mods[i].hi)
        {
            wsprintfA(out, "%s+0x%llX", g_mods[i].tag,
                      static_cast<unsigned long long>(ip - g_mods[i].lo));
            return;
        }
    wsprintfA(out, "?+0x0");
}

// Runs on the ticker thread only. Enumerating modules takes the loader lock, so
// it must never happen inside the exception handler (the faulting thread may own
// that lock or be mid-load -> deadlock).
void SnapshotModules(HMODULE self) noexcept
{
    SnapshotModule(self, "retrointercept");
    SnapshotModule(GetModuleHandleW(nullptr), "exe");
    HMODULE mods[256] = {};
    DWORD need = 0;
    if (EnumProcessModules(GetCurrentProcess(), mods, sizeof(mods), &need))
    {
        const DWORD cnt = (need / sizeof(HMODULE)) < 256 ? (need / sizeof(HMODULE)) : 256;
        for (DWORD i = 0; i < cnt; ++i)
        {
            char name[24] = "mod";
            GetModuleBaseNameA(GetCurrentProcess(), mods[i], name, sizeof(name));
            SnapshotModule(mods[i], name);
        }
    }
}

#ifdef _WIN64
#define CTX_IP(c) ((c)->Rip)
#define CTX_SP(c) ((c)->Rsp)
#define CTX_BP(c) ((c)->Rbp)
#define CTX_A(c)  ((c)->Rax)
#define CTX_B(c)  ((c)->Rbx)
#define CTX_C(c)  ((c)->Rcx)
#define CTX_D(c)  ((c)->Rdx)
#else
#define CTX_IP(c) ((c)->Eip)
#define CTX_SP(c) ((c)->Esp)
#define CTX_BP(c) ((c)->Ebp)
#define CTX_A(c)  ((c)->Eax)
#define CTX_B(c)  ((c)->Ebx)
#define CTX_C(c)  ((c)->Ecx)
#define CTX_D(c)  ((c)->Edx)
#endif

thread_local bool t_inVeh = false;

LONG WINAPI CrashVeh(EXCEPTION_POINTERS* ep) noexcept
{
    const EXCEPTION_RECORD* er = ep != nullptr ? ep->ExceptionRecord : nullptr;
    if (er == nullptr)
        return EXCEPTION_CONTINUE_SEARCH;
    const DWORD c = er->ExceptionCode;
    // Hard faults only (error severity 0xC...). Informational/debug/C++/CLR/RPC
    // exceptions are routine first-chance traffic in many engines and DRM.
    // Stack overflow is skipped: the handler runs on the exhausted stack and any
    // logging there turns a recoverable fault into an instant process kill.
    if ((c & 0xF0000000u) != 0xC0000000u || c == EXCEPTION_STACK_OVERFLOW || t_inVeh)
        return EXCEPTION_CONTINUE_SEARCH;
    const UINT_PTR ip = reinterpret_cast<UINT_PTR>(er->ExceptionAddress);
    const LONG seen = g_seenCount;
    for (LONG i = 0; i < seen; ++i)
        if (g_seenIps[i] == ip)
            return EXCEPTION_CONTINUE_SEARCH; // already logged this address
    t_inVeh = true; // a fault inside our own logging must not recurse
    if (seen < static_cast<LONG>(sizeof(g_seenIps) / sizeof(g_seenIps[0])))
    {
        g_seenIps[seen] = ip;
        InterlockedExchange(&g_seenCount, seen + 1);
    }

    char where[64];
    ResolveIp(ip, where);
    const ULONG_PTR ex0 = er->NumberParameters > 0 ? er->ExceptionInformation[0] : 0;
    const ULONG_PTR ex1 = er->NumberParameters > 1 ? er->ExceptionInformation[1] : 0;
    Log::Error("crash", "exception %08lX at %p (%s) tid=%lu flags=%lx ex0=%llx ex1=%llx",
               c, er->ExceptionAddress, where,
               GetCurrentThreadId(), er->ExceptionFlags,
               static_cast<unsigned long long>(ex0), static_cast<unsigned long long>(ex1));

    if (ep->ContextRecord != nullptr)
    {
        const CONTEXT* cr = ep->ContextRecord;
        char rip[64];
        ResolveIp(static_cast<UINT_PTR>(CTX_IP(cr)), rip);
        Log::Error("crash", "  ip=%p (%s) sp=%p bp=%p a=%llx b=%llx c=%llx d=%llx",
                   reinterpret_cast<void*>(static_cast<UINT_PTR>(CTX_IP(cr))), rip,
                   reinterpret_cast<void*>(static_cast<UINT_PTR>(CTX_SP(cr))),
                   reinterpret_cast<void*>(static_cast<UINT_PTR>(CTX_BP(cr))),
                   static_cast<unsigned long long>(CTX_A(cr)), static_cast<unsigned long long>(CTX_B(cr)),
                   static_cast<unsigned long long>(CTX_C(cr)), static_cast<unsigned long long>(CTX_D(cr)));

        // Stack scan: resolve qwords on [rsp, rsp+0x400) that look like return
        // addresses into module+RVA — crude but gives the call chain.
        ULONG_PTR lo = 0, hi = 0;
        GetCurrentThreadStackLimits(&lo, &hi);
        const ULONG_PTR* sp = reinterpret_cast<const ULONG_PTR*>(static_cast<UINT_PTR>(CTX_SP(cr)));
        const ULONG_PTR* end = sp + 128;
        if (reinterpret_cast<ULONG_PTR>(end) > hi)
            end = reinterpret_cast<const ULONG_PTR*>(hi);
        __try
        {
            int hits = 0;
            for (; sp < end && hits < 10; ++sp)
            {
                const ULONG_PTR v = *sp;
                const LONG n2 = g_modCount;
                for (LONG i = 0; i < n2; ++i)
                    if (v >= g_mods[i].lo && v < g_mods[i].hi)
                    {
                        Log::Error("crash", "  stk+%04llx -> %s+0x%llX",
                                   static_cast<unsigned long long>(
                                       reinterpret_cast<ULONG_PTR>(sp) - static_cast<ULONG_PTR>(CTX_SP(cr))),
                                   g_mods[i].tag,
                                   static_cast<unsigned long long>(v - g_mods[i].lo));
                        ++hits;
                        break;
                    }
            }
        }
        __except (EXCEPTION_EXECUTE_HANDLER)
        {
        }
    }
    t_inVeh = false;
    return EXCEPTION_CONTINUE_SEARCH;
}

DWORD WINAPI InitThread(LPVOID module) noexcept
{
    Log::Init();
    Log::Info("proxy", "RetroIntercept d3d11.dll loaded (Phase 4: retro modes + hotkeys)");

    g_veh = AddVectoredExceptionHandler(1, &CrashVeh); // first in the chain
    SnapshotModules(static_cast<HMODULE>(module));

    // Pre-resolve the real d3d11 here (LoadLibrary is legal on a normal thread) so the
    // game's first D3D11CreateDevice call doesn't pay the load cost. Stubs self-heal if
    // an export is still hit first.
    if (!proxy::EnsureRealD3D11())
        Log::Error("proxy", "real d3d11 resolution failed — forwarded exports will fault");

    hooks::EnsureConfigLoaded(); // no-op if the game's first device already loaded it

    // Hotkey + stats ticker: 50 ms poll is cheap and responsive.
    int tick = 0;
    while (InterlockedCompareExchange(&g_running, 0, 0) == 1)
    {
        Sleep(50);
        hooks::HotkeyTick();
        hooks::StatsTick();
        if (((++tick) & 63) == 0) // every ~3.2s pick up late-loaded modules
            SnapshotModules(static_cast<HMODULE>(module));
    }
    return 0;
}

} // namespace

BOOL APIENTRY DllMain(HMODULE module, DWORD reason, LPVOID reserved)
{
    if (reason == DLL_PROCESS_ATTACH)
    {
        DisableThreadLibraryCalls(module);
        // No LoadLibrary/COM/file work inside DllMain — defer to a thread.
        g_initThread = CreateThread(nullptr, 0, &InitThread, module, 0, nullptr);
    }
    else if (reason == DLL_PROCESS_DETACH)
    {
        // reserved != NULL means process termination: skip cleanup, CRT handles it.
        if (reserved == nullptr)
        {
            // Stop the ticker, then wait briefly for it to leave our code before the
            // module is unmapped.
            InterlockedExchange(&g_running, 0);
            if (g_initThread != nullptr)
            {
                WaitForSingleObject(g_initThread, 1000);
                CloseHandle(g_initThread);
                g_initThread = nullptr;
            }
            if (g_veh != nullptr)
            {
                RemoveVectoredExceptionHandler(g_veh);
                g_veh = nullptr;
            }
            hooks::Shutdown();
            Log::Info("proxy", "RetroIntercept unloaded");
            Log::Shutdown();
        }
    }
    return TRUE;
}
