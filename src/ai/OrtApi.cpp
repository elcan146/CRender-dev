// OrtApi — see header. All failures log once and latch.
// Built only when RETRO_HAVE_ORT=1 (headers/DLLs are x86-only).
#include "OrtApi.h" // IWYU pragma: keep
#if RETRO_HAVE_ORT
#include "../core/Log.h"

#include <Windows.h>
#include <cstdio>

namespace
{

const OrtApi*              g_api   = nullptr;
const OrtDmlApi*           g_dml   = nullptr;
ai::FnDMLCreateDevice      g_dmlCreate = nullptr;
PFN_D3D12_CREATE_DEVICE    g_d3d12Create = nullptr;
bool                       g_tried = false;

using FnOrtGetApiBase = const OrtApiBase*(ORT_API_CALL*)();

HMODULE LoadFromDir(const char* dir, const wchar_t* name) noexcept
{
    wchar_t wdir[MAX_PATH] = {};
    MultiByteToWideChar(CP_ACP, 0, dir, -1, wdir, MAX_PATH);
    wchar_t path[MAX_PATH] = {};
    _snwprintf_s(path, _TRUNCATE, L"%s\\%s", wdir, name);
    // ALTERED_SEARCH_PATH: the DLL's own deps (DirectML.dll, VC runtime) resolve from its folder.
    HMODULE m = LoadLibraryExW(path, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (m == nullptr)
        Log::Warn("ai", "LoadLibrary %ls failed (err=%lu)", path, GetLastError());
    return m;
}

} // namespace

bool ai::Load(const char* dir) noexcept
{
    if (g_api != nullptr && g_dml != nullptr)
        return true;
    if (g_tried)
        return false;
    g_tried = true;

    HMODULE d3d12 = LoadLibraryW(L"d3d12.dll");
    if (d3d12 == nullptr) { Log::Warn("ai", "no d3d12.dll — ONNX path unavailable"); return false; }
    g_d3d12Create = reinterpret_cast<PFN_D3D12_CREATE_DEVICE>(GetProcAddress(d3d12, "D3D12CreateDevice"));

    // Pin the redistributable DirectML.dll first so onnxruntime.dll binds to it rather
    // than the (older) in-box System32 copy.
    HMODULE dml = LoadFromDir(dir, L"DirectML.dll");
    if (dml == nullptr)
        return false;
    g_dmlCreate = reinterpret_cast<FnDMLCreateDevice>(GetProcAddress(dml, "DMLCreateDevice"));

    HMODULE ort = LoadFromDir(dir, L"onnxruntime.dll");
    if (ort == nullptr)
        return false;
    const auto getBase = reinterpret_cast<FnOrtGetApiBase>(GetProcAddress(ort, "OrtGetApiBase"));
    if (getBase == nullptr || g_d3d12Create == nullptr || g_dmlCreate == nullptr)
    {
        Log::Warn("ai", "missing export (OrtGetApiBase=%p D3D12CreateDevice=%p DMLCreateDevice=%p)",
                  reinterpret_cast<void*>(getBase), reinterpret_cast<void*>(g_d3d12Create),
                  reinterpret_cast<void*>(g_dmlCreate));
        return false;
    }
    const OrtApiBase* base = getBase();
    Log::Info("ai", "onnxruntime %s loaded (header API %d)", base->GetVersionString(), ORT_API_VERSION);
    g_api = base->GetApi(ORT_API_VERSION);
    if (g_api == nullptr)
    {
        Log::Warn("ai", "ORT runtime does not support API version %d", ORT_API_VERSION);
        return false;
    }
    const void* dmlApi = nullptr;
    if (!Ok(g_api->GetExecutionProviderApi("DML", ORT_API_VERSION, &dmlApi), "GetExecutionProviderApi(DML)") ||
        dmlApi == nullptr)
    {
        g_api = nullptr;
        return false;
    }
    g_dml = static_cast<const OrtDmlApi*>(dmlApi);
    return true;
}

const OrtApi*    ai::Api() noexcept { return g_api; }
const OrtDmlApi* ai::Dml() noexcept { return g_dml; }
ai::FnDMLCreateDevice  ai::DmlCreateDevice() noexcept { return g_dmlCreate; }
PFN_D3D12_CREATE_DEVICE ai::D3D12Create() noexcept { return g_d3d12Create; }

bool ai::Ok(OrtStatus* st, const char* what) noexcept
{
    if (st == nullptr)
        return true;
    if (g_api != nullptr)
    {
        Log::Warn("ai", "%s failed: %s", what, g_api->GetErrorMessage(st));
        g_api->ReleaseStatus(st);
    }
    else
        Log::Warn("ai", "%s failed", what);
    return false;
}

#endif // RETRO_HAVE_ORT
