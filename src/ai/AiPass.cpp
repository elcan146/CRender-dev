// AiPass — see header for the pipeline overview.
//
// Failure discipline: every init stage logs "ai fail: <stage>" and latches g_failed;
// Ensure() then returns false and the caller (Terminator mode) switches to the HLSL
// fallback. F11 (config reload) clears the latch and retries the session build.
#include "AiPass.h"
#include "../core/Log.h"

#include <Windows.h>

#if !RETRO_HAVE_ORT

// Built without third_party/onnxruntime (run tools/ai/fetch_ort.ps1) — Terminator uses the HLSL fallback.
void ai::SetConfig(const char*, int, int, int, float, int) noexcept {}
void ai::SetLook(float, const float*) noexcept {}
bool ai::Ensure(ID3D11Device*) noexcept
{
    static bool warned = false;
    if (!warned) { warned = true; Log::Warn("ai", "built without ONNX Runtime — ONNX path disabled"); }
    return false;
}
void ai::Apply(ID3D11DeviceContext*, IDXGISwapChain*) noexcept {}
void ai::ReleaseBackbuffer() noexcept {}
void ai::TakeStats(double* avgMs, long long* runs) noexcept { *avgMs = 0.0; *runs = 0; }
void ai::Reset() noexcept {}

#else

#include "OrtApi.h"
#include "../d3d11/RetroPass.h"
#include "../d3d11/StateBlock.h"

#include <d3d11_1.h>
#include <d3d12.h>
#include <cstdio>
#include <cstring>

namespace
{

struct Cfg
{
    char  model[MAX_PATH];
    int   w, h;
    int   temporal; // 0 single, 1 lerp, 2 two-input model
    int   stride;
    float blend;
};

Cfg           g_cfg     = { "retro_ai.onnx", 320, 180, 0, 1, 0.5f }; // active (Present thread)
Cfg           g_pending = g_cfg;                                      // written by SetConfig
volatile LONG g_dirty   = 0;
SRWLOCK       g_cfgLock = SRWLOCK_INIT;

// Terminator look (TermEdgeGain / TermTint), applied in the output pass.
float         g_lookGain    = 1.0f;
float         g_lookTint[3] = { 1.0f, 0.0f, 0.0f };

bool          g_failed = false;
bool          g_ready  = false;
ID3D11Device* g_dev    = nullptr; // non-owning

// ---- D3D12 / DML side ----
ID3D12Device*       g_d12   = nullptr;
ID3D12CommandQueue* g_queue = nullptr;
ID3D12CommandAllocator* g_cmdAlloc = nullptr;
ID3D12GraphicsCommandList* g_cmdList = nullptr;
ID3D12Fence*        g_fence = nullptr;
UINT64              g_fenceVal = 0;
HANDLE              g_fenceEv  = nullptr;
IDMLDevice*         g_dml   = nullptr;

// Interop: D3D11 cannot open D3D12 BUFFERS (OpenSharedResource1 -> E_INVALIDARG), only
// 2D textures. So the bridge is three shared R32_FLOAT planar textures (input, output
// ping-pong) at W x 3H — byte-identical to NCHW fp32 — while the DML tensors live on
// private D3D12 buffers bound via CreateGPUAllocationFromD3DResource. Each Run is wrapped
// in CopyTextureRegion tex->buf / buf->tex on our D3D12 queue (straight memcpy; requires
// AiWidth*4 to be a multiple of 256, i.e. width % 64 == 0). Zero CPU copies.
// Sync (v1): D3D11 event query before the copies, D3D12 fence wait after.
// [0] = input, [1]/[2] = output ping-pong — private D3D12 buffers for DML tensors
ID3D12Resource*            g_res12[3]  = {};
ID3D12Resource*            g_tex12[3]  = {};
ID3D11Texture2D*           g_tex11[3]  = {};
ID3D11RenderTargetView*    g_inRTV     = nullptr; // on g_tex11[0]
ID3D11ShaderResourceView*  g_outSRV[2] = {};      // on g_tex11[1], g_tex11[2]

// ---- ORT side ----
OrtEnv*            g_env     = nullptr;
OrtSessionOptions* g_opts    = nullptr;
OrtSession*        g_session = nullptr;
OrtMemoryInfo*     g_memInfo = nullptr;
void*              g_alloc[3] = {};
OrtValue*          g_val[3]   = {};
OrtIoBinding*      g_bind[2]  = {};
char               g_inName[2][128] = {};
char               g_outName[128]   = {};

// ---- D3D11 pass objects ----
ID3D11PixelShader*  g_psIn   = nullptr;
ID3D11PixelShader*  g_psOut  = nullptr;
ID3D11Buffer*       g_cb     = nullptr;
ID3D11SamplerState* g_linear = nullptr;
ID3D11Query*        g_query  = nullptr;

// ---- backbuffer chain ----
ID3D11Texture2D*          g_bb      = nullptr;
ID3D11Texture2D*          g_copy    = nullptr;
ID3D11ShaderResourceView* g_copySRV = nullptr;
ID3D11RenderTargetView*   g_bbRTV   = nullptr;
UINT                      g_bbW = 0, g_bbH = 0, g_bbSamples = 1;
DXGI_FORMAT               g_bbFmt = DXGI_FORMAT_UNKNOWN;

// ---- per-frame ----
int               g_parity = 0;  // binding used by the next Run
int               g_shown  = 0;  // output buffer holding the latest result
long long         g_runs   = 0;
unsigned          g_frame  = 0;
volatile LONGLONG g_statTicks = 0;
volatile LONGLONG g_statRuns  = 0;

// Input pass: sample the backbuffer copy at model res, write NCHW planar into the
// shared W x 3H texture (channel c occupies rows [c*h, c*h+h)).
const char kPsInSrc[] =
    "Texture2D src : register(t0);"
    "SamplerState smp : register(s0);"
    "cbuffer P : register(b0) { float2 aiDim; float2 bbDim; float blend; float lerpOn;"
    "                          float edgeGain; float lookOn; float3 tint; float srgbOn; };"
    "float4 main(float4 p : SV_Position) : SV_Target {"
    "  uint h = (uint)aiDim.y;"
    "  uint plane = (uint)p.y / h;"                  // row band -> channel
    "  float2 uv = float2(p.x, p.y - plane * h) / aiDim;"
    "  float3 c = saturate(src.SampleLevel(smp, uv, 0).rgb);"
    "  if (srgbOn > 0.5) c = pow(c, 0.4545);"          // undo _SRGB linearize
    "  return float4(plane == 0 ? c.r : (plane == 1 ? c.g : c.b), 0.0, 0.0, 0.0);"
    "}";

// Output pass: gather NCHW planar model output (+ optional lerp with previous output).
// lookOn: re-grade the Terminator encoding (R = bg + e, G = B = 0.7 e) with TermEdgeGain
// and TermTint; skipped at defaults so other models pass through untouched.
const char kPsOutSrc[] =
    "Texture2D<float> cur : register(t0);"
    "Texture2D<float> prv : register(t1);"
    "cbuffer P : register(b0) { float2 aiDim; float2 bbDim; float blend; float lerpOn; float edgeGain; float lookOn;"
    "                          float3 tint; float pad; };"
    // R32_FLOAT has no HW filtering — bilinear fetch of one NCHW plane at suv.
    "float bil(Texture2D<float> t, float ch, float2 suv, float w, float h) {"
    "  float2 b = floor(suv);"
    "  float2 f = suv - b;"
    "  int2 i0 = clamp((int2)b, int2(0, 0), int2((int)w - 1, (int)h - 1));"
    "  int2 i1 = clamp(i0 + int2(1, 1), int2(0, 0), int2((int)w - 1, (int)h - 1));"
    "  float a = t.Load(int3(i0.x, (int)(i0.y + ch * h), 0)).x;"
    "  float c = t.Load(int3(i1.x, (int)(i0.y + ch * h), 0)).x;"
    "  float e = t.Load(int3(i0.x, (int)(i1.y + ch * h), 0)).x;"
    "  float g = t.Load(int3(i1.x, (int)(i1.y + ch * h), 0)).x;"
    "  return lerp(lerp(a, c, f.x), lerp(e, g, f.x), f.y);"
    "}"
    "float4 main(float4 p : SV_Position) : SV_Target {"
    "  float w = aiDim.x; float h = aiDim.y;"
    "  float2 suv = p.xy * aiDim / bbDim - 0.5;"
    "  float3 c = float3(bil(cur, 0.0, suv, w, h),"
    "                  bil(cur, 1.0, suv, w, h),"
    "                  bil(cur, 2.0, suv, w, h));"
    "  if (lerpOn > 0.5) c = lerp(c, float3(bil(prv, 0.0, suv, w, h),"
    "                               bil(prv, 1.0, suv, w, h),"
    "                               bil(prv, 2.0, suv, w, h)), blend);"
    "  if (lookOn > 0.5) {"
    "    float e = saturate(c.g / 0.7);"
    "    float e2 = saturate(e * edgeGain);"
    "    c = max(c.r - e, 0.0) * tint + float3(e2, 0.7 * e2, 0.7 * e2);"
    "  }"
    "  return float4(saturate(c), 1.0);"
    "}";

template <class T>
void Rel(T*& p) noexcept
{
    if (p != nullptr) { p->Release(); p = nullptr; }
}

bool Fail(const char* what, HRESULT hr = S_OK) noexcept
{
    Log::Warn("ai", "ai fail: %s (hr=0x%08lX) — ONNX path disabled until F11", what,
              static_cast<unsigned long>(hr));
    g_failed = true;
    return false;
}

UINT TensorElems() noexcept { return static_cast<UINT>(g_cfg.w) * static_cast<UINT>(g_cfg.h) * 3u; }

void ReleaseBB() noexcept
{
    Rel(g_copySRV);
    Rel(g_copy);
    Rel(g_bbRTV);
    Rel(g_bb);
    g_bbW = g_bbH = 0;
}

// Session + interop resources (everything that depends on model/dims/temporal).
void Teardown() noexcept
{
    const OrtApi* api = ai::Api();
    const OrtDmlApi* dml = ai::Dml();
    if (api != nullptr)
    {
        for (auto*& b : g_bind)  if (b != nullptr) { api->ReleaseIoBinding(b); b = nullptr; }
        for (auto*& v : g_val)   if (v != nullptr) { api->ReleaseValue(v); v = nullptr; }
        if (g_session != nullptr) { api->ReleaseSession(g_session); g_session = nullptr; }
        if (g_opts != nullptr)    { api->ReleaseSessionOptions(g_opts); g_opts = nullptr; }
        if (g_memInfo != nullptr) { api->ReleaseMemoryInfo(g_memInfo); g_memInfo = nullptr; }
        if (g_env != nullptr)     { api->ReleaseEnv(g_env); g_env = nullptr; }
    }
    for (auto*& a : g_alloc)
        if (a != nullptr) { if (dml != nullptr) ai::Ok(dml->FreeGPUAllocation(a), "FreeGPUAllocation"); a = nullptr; }
    Rel(g_inRTV);
    Rel(g_outSRV[0]); Rel(g_outSRV[1]);
    for (auto*& t : g_tex11) Rel(t);
    for (auto*& t : g_tex12) Rel(t);
    for (auto*& r : g_res12) Rel(r);
    Rel(g_cmdList);
    Rel(g_cmdAlloc);
    Rel(g_fence);
    Rel(g_queue);
    Rel(g_dml);
    Rel(g_d12);
    if (g_fenceEv != nullptr) { CloseHandle(g_fenceEv); g_fenceEv = nullptr; }
    g_fenceVal = 0;
    g_parity = g_shown = 0;
    g_runs = 0;
    g_frame = 0;
    g_ready = false;
}

void ApplyPendingConfig() noexcept
{
    if (InterlockedCompareExchange(&g_dirty, 0, 0) == 0)
        return;
    AcquireSRWLockExclusive(&g_cfgLock);
    const Cfg p = g_pending;
    InterlockedExchange(&g_dirty, 0);
    ReleaseSRWLockExclusive(&g_cfgLock);

    const bool structural = strcmp(p.model, g_cfg.model) != 0 || p.w != g_cfg.w || p.h != g_cfg.h ||
                            (p.temporal == 2) != (g_cfg.temporal == 2);
    g_cfg = p;
    if (structural || g_failed)
    {
        Teardown();
        g_failed = false;
    }
}

bool CreateD3D12() noexcept
{
    IDXGIDevice* xd = nullptr;
    IDXGIAdapter* ad = nullptr;
    if (FAILED(g_dev->QueryInterface(IID_PPV_ARGS(&xd))) || xd == nullptr)
        return Fail("IDXGIDevice");
    const HRESULT ahr = xd->GetAdapter(&ad);
    xd->Release();
    if (FAILED(ahr) || ad == nullptr)
        return Fail("GetAdapter", ahr);
    DXGI_ADAPTER_DESC desc = {};
    ad->GetDesc(&desc);
    // Same adapter as the game's D3D11 device — shared-heap interop requires it.
    HRESULT hr = ai::D3D12Create()(ad, D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), reinterpret_cast<void**>(&g_d12));
    ad->Release();
    if (FAILED(hr) || g_d12 == nullptr)
        return Fail("D3D12CreateDevice", hr);
    Log::Info("ai", "D3D12 device on '%ls' (luid %08lX:%08lX)", desc.Description,
              static_cast<unsigned long>(desc.AdapterLuid.HighPart),
              static_cast<unsigned long>(desc.AdapterLuid.LowPart));

    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (FAILED(hr = g_d12->CreateCommandQueue(&qd, IID_PPV_ARGS(&g_queue))))
        return Fail("CreateCommandQueue", hr);
    if (FAILED(hr = g_d12->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&g_fence))))
        return Fail("CreateFence", hr);
    if (FAILED(hr = g_d12->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&g_cmdAlloc))))
        return Fail("CreateCommandAllocator", hr);
    if (FAILED(hr = g_d12->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, g_cmdAlloc, nullptr,
                                           IID_PPV_ARGS(&g_cmdList))))
        return Fail("CreateCommandList", hr);
    g_cmdList->Close(); // created open; we Reset it per frame
    g_fenceEv = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (g_fenceEv == nullptr)
        return Fail("CreateEvent");
    if (FAILED(hr = ai::DmlCreateDevice()(g_d12, DML_CREATE_DEVICE_FLAG_NONE, IID_PPV_ARGS(&g_dml))))
        return Fail("DMLCreateDevice", hr);
    return true;
}

// Private D3D12 committed buffer — a DML tensor (not shared; D3D11 never sees it).
bool CreateTensorBuffer(int idx, UINT64 bytes) noexcept
{
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width            = bytes;
    rd.Height           = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels        = 1;
    rd.Format           = DXGI_FORMAT_UNKNOWN;
    rd.SampleDesc.Count = 1;
    rd.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags            = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS; // required by DML
    const HRESULT hr = g_d12->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                                                      D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                      IID_PPV_ARGS(&g_res12[idx]));
    if (FAILED(hr))
        return Fail("CreateCommittedResource(tensor)", hr);
    return true;
}

// Shared R32_FLOAT W x 3H texture — the D3D11<->D3D12 bridge. SIMULTANEOUS_ACCESS keeps
// it promotable from COMMON on any queue/device; ALLOW_RENDER_TARGET on the input so
// D3D11 gets an RTV for the planar write pass (D3D11 cannot open UAV-flagged shared
// resources — E_INVALIDARG from OpenSharedResource1).
bool CreateSharedTexture(int idx, bool rt) noexcept
{
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension        = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width            = static_cast<UINT64>(g_cfg.w);
    rd.Height           = static_cast<UINT>(g_cfg.h) * 3;
    rd.DepthOrArraySize = 1;
    rd.MipLevels        = 1;
    rd.Format           = DXGI_FORMAT_R32_FLOAT;
    rd.SampleDesc.Count = 1;
    rd.Layout           = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    rd.Flags            = D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS |
                          (rt ? D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET : D3D12_RESOURCE_FLAG_NONE);
    HRESULT hr = g_d12->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_SHARED, &rd,
                                                D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                IID_PPV_ARGS(&g_tex12[idx]));
    if (FAILED(hr))
        return Fail("CreateCommittedResource(shared tex)", hr);
    HANDLE h = nullptr;
    if (FAILED(hr = g_d12->CreateSharedHandle(g_tex12[idx], nullptr, GENERIC_ALL, nullptr, &h)) || h == nullptr)
        return Fail("CreateSharedHandle", hr);
    ID3D11Device1* d1 = nullptr;
    hr = g_dev->QueryInterface(IID_PPV_ARGS(&d1));
    if (SUCCEEDED(hr) && d1 != nullptr)
    {
        hr = d1->OpenSharedResource1(h, IID_PPV_ARGS(&g_tex11[idx]));
        d1->Release();
    }
    CloseHandle(h);
    if (FAILED(hr) || g_tex11[idx] == nullptr)
        return Fail("OpenSharedResource1(texture)", hr);
    if (idx == 0)
    {
        D3D11_TEXTURE2D_DESC td = {};
        g_tex11[idx]->GetDesc(&td);
        Log::Info("ai", "shared texture %ux%u fmt=%u opened on D3D11: bind=0x%X misc=0x%X",
                  td.Width, td.Height, td.Format, td.BindFlags, td.MiscFlags);
    }
    return true;
}

// tex->buf or buf->tex on our queue; the caller fences after the final submit.
bool SubmitCopy(ID3D12Resource* dst, ID3D12Resource* src, bool dstIsBuffer) noexcept
{
    const UINT w = static_cast<UINT>(g_cfg.w), h3 = static_cast<UINT>(g_cfg.h) * 3;
    D3D12_TEXTURE_COPY_LOCATION texLoc = {};
    texLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    texLoc.SubresourceIndex = 0;
    D3D12_TEXTURE_COPY_LOCATION bufLoc = {};
    bufLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    bufLoc.PlacedFootprint.Offset = 0;
    bufLoc.PlacedFootprint.Footprint.Format   = DXGI_FORMAT_R32_FLOAT;
    bufLoc.PlacedFootprint.Footprint.Width    = w;
    bufLoc.PlacedFootprint.Footprint.Height   = h3;
    bufLoc.PlacedFootprint.Footprint.Depth    = 1;
    bufLoc.PlacedFootprint.Footprint.RowPitch = w * sizeof(float); // requires w % 64 == 0

    if (FAILED(g_cmdAlloc->Reset()) || FAILED(g_cmdList->Reset(g_cmdAlloc, nullptr)))
        return false;
    if (dstIsBuffer)
    {
        bufLoc.pResource = dst;
        texLoc.pResource = src;
        g_cmdList->CopyTextureRegion(&bufLoc, 0, 0, 0, &texLoc, nullptr);
    }
    else
    {
        texLoc.pResource = dst;
        bufLoc.pResource = src;
        g_cmdList->CopyTextureRegion(&texLoc, 0, 0, 0, &bufLoc, nullptr);
    }
    if (FAILED(g_cmdList->Close()))
        return false;
    ID3D12CommandList* lists[] = { g_cmdList };
    g_queue->ExecuteCommandLists(1, lists);
    return true;
}

bool CheckTensor(bool input, size_t i, char* nameOut) noexcept
{
    const OrtApi* api = ai::Api();
    OrtAllocator* alloc = nullptr;
    if (!ai::Ok(api->GetAllocatorWithDefaultOptions(&alloc), "GetAllocatorWithDefaultOptions"))
        return Fail("allocator");
    char* nm = nullptr;
    OrtStatus* st = input ? api->SessionGetInputName(g_session, i, alloc, &nm)
                          : api->SessionGetOutputName(g_session, i, alloc, &nm);
    if (!ai::Ok(st, "Session*Name") || nm == nullptr)
        return Fail("tensor name");
    strncpy_s(nameOut, 128, nm, _TRUNCATE);
    api->AllocatorFree(alloc, nm);

    OrtTypeInfo* ti = nullptr;
    st = input ? api->SessionGetInputTypeInfo(g_session, i, &ti) : api->SessionGetOutputTypeInfo(g_session, i, &ti);
    if (!ai::Ok(st, "Session*TypeInfo") || ti == nullptr)
        return Fail("type info");
    const OrtTensorTypeAndShapeInfo* tsi = nullptr;
    ONNXTensorElementDataType et = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    size_t nd = 0;
    int64_t dims[8] = {};
    bool ok = ai::Ok(api->CastTypeInfoToTensorInfo(ti, &tsi), "CastTypeInfoToTensorInfo") && tsi != nullptr &&
              ai::Ok(api->GetTensorElementType(tsi, &et), "GetTensorElementType") &&
              ai::Ok(api->GetDimensionsCount(tsi, &nd), "GetDimensionsCount") && nd == 4 &&
              ai::Ok(api->GetDimensions(tsi, dims, nd), "GetDimensions");
    api->ReleaseTypeInfo(ti);
    Log::Info("ai", "%s[%u] '%s' type=%d dims=[%lld,%lld,%lld,%lld]", input ? "input" : "output",
              static_cast<unsigned>(i), nameOut, static_cast<int>(et),
              dims[0], dims[1], dims[2], dims[3]);
    if (!ok || et != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        return Fail("tensor must be fp32 rank-4 NCHW");
    if (dims[0] < 0 || dims[1] < 0 || dims[2] < 0 || dims[3] < 0)
        return Fail("dynamic-shape model rejected — export with fixed [1,3,AiHeight,AiWidth]");
    if (dims[0] != 1 || dims[1] != 3 || dims[2] != g_cfg.h || dims[3] != g_cfg.w)
    {
        Log::Warn("ai", "model shape [%lld,%lld,%lld,%lld] != config [1,3,%d,%d] (AiHeight/AiWidth)",
                  dims[0], dims[1], dims[2], dims[3], g_cfg.h, g_cfg.w);
        return Fail("shape mismatch");
    }
    return true;
}

bool CreateSession() noexcept
{
    const OrtApi* api = ai::Api();
    const OrtDmlApi* dml = ai::Dml();

    char full[MAX_PATH] = {};
    const bool absolute = g_cfg.model[0] == '\\' || g_cfg.model[0] == '/' ||
                          (g_cfg.model[0] != 0 && g_cfg.model[1] == ':');
    if (absolute)
        strncpy_s(full, g_cfg.model, _TRUNCATE);
    else
        _snprintf_s(full, _TRUNCATE, "%s\\%s", Log::ModuleDir(), g_cfg.model);
    if (GetFileAttributesA(full) == INVALID_FILE_ATTRIBUTES)
    {
        Log::Warn("ai", "model not found: %s", full);
        return Fail("model file");
    }
    wchar_t wfull[MAX_PATH] = {};
    MultiByteToWideChar(CP_UTF8, 0, full, -1, wfull, MAX_PATH);

    if (!ai::Ok(api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "retro", &g_env), "CreateEnv") ||
        !ai::Ok(api->CreateSessionOptions(&g_opts), "CreateSessionOptions") ||
        !ai::Ok(api->DisableMemPattern(g_opts), "DisableMemPattern") ||
        !ai::Ok(api->SetSessionExecutionMode(g_opts, ORT_SEQUENTIAL), "SetSessionExecutionMode") ||
        !ai::Ok(api->SetIntraOpNumThreads(g_opts, 1), "SetIntraOpNumThreads") ||
        !ai::Ok(api->SetSessionGraphOptimizationLevel(g_opts, ORT_ENABLE_ALL), "SetGraphOptimizationLevel") ||
        !ai::Ok(dml->SessionOptionsAppendExecutionProvider_DML1(g_opts, g_dml, g_queue), "AppendExecutionProvider_DML1"))
        return Fail("session options");

    LARGE_INTEGER f = {}, t0 = {}, t1 = {};
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t0);
    if (!ai::Ok(api->CreateSession(g_env, wfull, g_opts, &g_session), "CreateSession") || g_session == nullptr)
        return Fail("CreateSession");
    QueryPerformanceCounter(&t1);
    Log::Info("ai", "session loaded %s in %.0f ms", full,
              static_cast<double>(t1.QuadPart - t0.QuadPart) * 1000.0 / static_cast<double>(f.QuadPart));

    size_t nIn = 0, nOut = 0;
    if (!ai::Ok(api->SessionGetInputCount(g_session, &nIn), "SessionGetInputCount") ||
        !ai::Ok(api->SessionGetOutputCount(g_session, &nOut), "SessionGetOutputCount"))
        return Fail("io count");
    const size_t wantIn = g_cfg.temporal == 2 ? 2 : 1;
    if (nIn != wantIn || nOut < 1)
    {
        Log::Warn("ai", "model has %u inputs / %u outputs; AiTemporal=%d needs %u input(s)",
                  static_cast<unsigned>(nIn), static_cast<unsigned>(nOut), g_cfg.temporal,
                  static_cast<unsigned>(wantIn));
        return Fail("model io count");
    }
    for (size_t i = 0; i < nIn; ++i)
        if (!CheckTensor(true, i, g_inName[i]))
            return false;
    return CheckTensor(false, 0, g_outName);
}

bool CreateInterop() noexcept
{
    const OrtApi* api = ai::Api();
    const OrtDmlApi* dml = ai::Dml();
    const UINT n = TensorElems();
    const UINT64 bytes = static_cast<UINT64>(n) * sizeof(float);
    if ((g_cfg.w % 64) != 0) // row pitch must be a multiple of 256 for CopyTextureRegion
        return Fail("AiWidth must be a multiple of 64");

    for (int i = 0; i < 3; ++i)
        if (!CreateTensorBuffer(i, bytes))
            return false;
    // Every shared texture gets ALLOW_RENDER_TARGET — D3D11 OpenSharedResource1 rejects
    // non-RT-flagged D3D12 resources with E_INVALIDARG.
    for (int i = 0; i < 3; ++i)
        if (!CreateSharedTexture(i, true))
            return false;

    HRESULT hr = g_dev->CreateRenderTargetView(g_tex11[0], nullptr, &g_inRTV);
    if (FAILED(hr))
        return Fail("input texture RTV", hr);
    for (int i = 0; i < 2; ++i)
        if (FAILED(hr = g_dev->CreateShaderResourceView(g_tex11[1 + i], nullptr, &g_outSRV[i])))
            return Fail("output texture SRV", hr);

    if (!ai::Ok(api->CreateMemoryInfo("DML", OrtDeviceAllocator, 0, OrtMemTypeDefault, &g_memInfo), "CreateMemoryInfo(DML)"))
        return Fail("memory info");
    const int64_t shape[4] = { 1, 3, g_cfg.h, g_cfg.w };
    for (int i = 0; i < 3; ++i)
    {
        if (!ai::Ok(dml->CreateGPUAllocationFromD3DResource(g_res12[i], &g_alloc[i]), "CreateGPUAllocationFromD3DResource"))
            return Fail("GPU allocation");
        if (!ai::Ok(api->CreateTensorWithDataAsOrtValue(g_memInfo, g_alloc[i], static_cast<size_t>(bytes), shape, 4,
                                                        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &g_val[i]),
                    "CreateTensorWithDataAsOrtValue"))
            return Fail("tensor value");
    }
    // binding[p]: input0 = in, [input1 = out[1-p]], output = out[p]
    for (int p = 0; p < 2; ++p)
    {
        if (!ai::Ok(api->CreateIoBinding(g_session, &g_bind[p]), "CreateIoBinding") ||
            !ai::Ok(api->BindInput(g_bind[p], g_inName[0], g_val[0]), "BindInput(frame)") ||
            (g_cfg.temporal == 2 &&
             !ai::Ok(api->BindInput(g_bind[p], g_inName[1], g_val[1 + (1 - p)]), "BindInput(prev)")) ||
            !ai::Ok(api->BindOutput(g_bind[p], g_outName, g_val[1 + p]), "BindOutput"))
            return Fail("io binding");
    }
    return true;
}

bool CreatePassObjects() noexcept
{
    if (g_psIn != nullptr && g_psOut != nullptr && g_cb != nullptr && g_linear != nullptr && g_query != nullptr)
        return true;
    if (g_dev->GetFeatureLevel() < D3D_FEATURE_LEVEL_11_0)
        return Fail("feature level < 11_0 (PS UAV writes)");
    ID3DBlob* b = retro::CompileShader(kPsInSrc, "main", "ps_5_0");
    if (b == nullptr)
        return Fail("compile input PS");
    HRESULT hr = g_dev->CreatePixelShader(b->GetBufferPointer(), b->GetBufferSize(), nullptr, &g_psIn);
    b->Release();
    if (FAILED(hr))
        return Fail("create input PS", hr);
    b = retro::CompileShader(kPsOutSrc, "main", "ps_5_0");
    if (b == nullptr)
        return Fail("compile output PS");
    hr = g_dev->CreatePixelShader(b->GetBufferPointer(), b->GetBufferSize(), nullptr, &g_psOut);
    b->Release();
    if (FAILED(hr))
        return Fail("create output PS", hr);

    D3D11_BUFFER_DESC cd = {};
    cd.ByteWidth = 48;
    cd.Usage = D3D11_USAGE_DEFAULT;
    cd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    if (FAILED(hr = g_dev->CreateBuffer(&cd, nullptr, &g_cb)))
        return Fail("CB", hr);
    D3D11_SAMPLER_DESC sd = {};
    sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    sd.MaxLOD = D3D11_FLOAT32_MAX;
    if (FAILED(hr = g_dev->CreateSamplerState(&sd, &g_linear)))
        return Fail("sampler", hr);
    D3D11_QUERY_DESC qd = { D3D11_QUERY_EVENT, 0 };
    if (FAILED(hr = g_dev->CreateQuery(&qd, &g_query)))
        return Fail("event query", hr);
    return true;
}

// True when the backbuffer is an _SRGB variant — sampling linearizes texels
// (~0.5 -> ~0.21), so the input PS re-encodes to sRGB (pow 1/2.2) for the model.
bool IsSrgbFmt(DXGI_FORMAT f) noexcept
{
    return f == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB ||
           f == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB ||
           f == DXGI_FORMAT_B8G8R8X8_UNORM_SRGB;
}

bool EnsureBackbuffer(IDXGISwapChain* sc) noexcept
{
    ID3D11Texture2D* bb = nullptr;
    if (FAILED(sc->GetBuffer(0, IID_PPV_ARGS(&bb))) || bb == nullptr)
        return false;
    if (bb == g_bb && g_copySRV != nullptr && g_bbRTV != nullptr)
    {
        bb->Release();
        return true;
    }
    ReleaseBB();
    D3D11_TEXTURE2D_DESC bd = {};
    bb->GetDesc(&bd);
    D3D11_TEXTURE2D_DESC cd = bd;
    cd.Usage = D3D11_USAGE_DEFAULT;
    cd.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    cd.CPUAccessFlags = 0;
    cd.MiscFlags = 0;
    cd.SampleDesc.Count = 1;
    cd.SampleDesc.Quality = 0;
    if (FAILED(g_dev->CreateTexture2D(&cd, nullptr, &g_copy)) ||
        FAILED(g_dev->CreateShaderResourceView(g_copy, nullptr, &g_copySRV)) ||
        FAILED(g_dev->CreateRenderTargetView(bb, nullptr, &g_bbRTV)))
    {
        Log::Warn("ai", "backbuffer chain create failed fmt=%u", bd.Format);
        bb->Release();
        ReleaseBB();
        return false;
    }
    g_bb = bb; // keep the reference — pointer identity check
    g_bbW = bd.Width; g_bbH = bd.Height;
    g_bbSamples = bd.SampleDesc.Count;
    g_bbFmt = bd.Format;
    Log::Info("ai", "backbuffer chain %ux%u fmt=%u -> model %dx%d", g_bbW, g_bbH, g_bbFmt, g_cfg.w, g_cfg.h);
    return true;
}

bool WaitD3D11Idle(ID3D11DeviceContext* ctx) noexcept
{
    ctx->End(g_query);
    BOOL done = FALSE;
    const DWORD t0 = GetTickCount();
    while (ctx->GetData(g_query, &done, sizeof(done), 0) == S_FALSE)
    {
        if (GetTickCount() - t0 > 500)
        {
            Log::Warn("ai", "D3D11 event query timeout — skipping inference this frame");
            return false;
        }
        SwitchToThread();
    }
    return true;
}

bool WaitD3D12Idle() noexcept
{
    const UINT64 v = ++g_fenceVal;
    if (FAILED(g_queue->Signal(g_fence, v)))
        return false;
    if (g_fence->GetCompletedValue() < v)
    {
        if (FAILED(g_fence->SetEventOnCompletion(v, g_fenceEv)))
            return false;
        if (WaitForSingleObject(g_fenceEv, 2000) != WAIT_OBJECT_0)
        {
            Log::Warn("ai", "D3D12 fence wait timeout");
            return false;
        }
    }
    return true;
}

} // namespace

void ai::SetConfig(const char* model, int width, int height, int temporal, float blend, int stride) noexcept
{
    AcquireSRWLockExclusive(&g_cfgLock);
    if (model != nullptr && model[0] != 0)
        strncpy_s(g_pending.model, model, _TRUNCATE);
    g_pending.w        = (width  >= 8 && width  <= 4096) ? width  : 320;
    g_pending.h        = (height >= 8 && height <= 4096) ? height : 180;
    g_pending.temporal = (temporal >= 0 && temporal <= 2) ? temporal : 0;
    g_pending.blend    = (blend >= 0.0f && blend <= 1.0f) ? blend : 0.5f;
    g_pending.stride   = (stride >= 1 && stride <= 60) ? stride : 1;
    InterlockedExchange(&g_dirty, 1);
    ReleaseSRWLockExclusive(&g_cfgLock);
    Log::Info("ai", "config: model=%s %dx%d temporal=%d blend=%.2f stride=%d", g_pending.model,
              g_pending.w, g_pending.h, g_pending.temporal, g_pending.blend, g_pending.stride);
}

void ai::SetLook(float edgeGain, const float* tint) noexcept
{
    g_lookGain = edgeGain;
    if (tint != nullptr)
        for (int i = 0; i < 3; ++i)
            g_lookTint[i] = tint[i];
}

bool ai::Ensure(ID3D11Device* dev) noexcept
{
    if (dev == nullptr)
        return false;
    if (g_dev != nullptr && g_dev != dev)
    {
        ReleaseBB();
        Teardown();
        Rel(g_psIn); Rel(g_psOut); Rel(g_cb); Rel(g_linear); Rel(g_query);
        g_failed = false;
    }
    g_dev = dev;
    ApplyPendingConfig();
    if (g_failed)
        return false;
    if (g_ready)
        return true;
    if (retro::FullscreenVS() == nullptr)
        return Fail("retro pass VS unavailable");
    if (!ai::Load(Log::ModuleDir()))
        return Fail("onnxruntime.dll / DirectML.dll load");
    if (!CreatePassObjects() || !CreateD3D12() || !CreateSession() || !CreateInterop())
    {
        Teardown();
        g_failed = true;
        return false;
    }
    g_ready = true;
    Log::Info("ai", "ONNX pipeline ready: %dx%d temporal=%d stride=%d", g_cfg.w, g_cfg.h,
              g_cfg.temporal, g_cfg.stride);
    return true;
}

void ai::Apply(ID3D11DeviceContext* ctx, IDXGISwapChain* sc) noexcept
{
    if (!g_ready || g_failed || ctx == nullptr || sc == nullptr)
        return;
    if (!EnsureBackbuffer(sc))
        return;

    retro::StateBlock saved;
    saved.Capture(ctx);

    const bool lookOn = g_lookGain != 1.0f || g_lookTint[0] != 1.0f || g_lookTint[1] != 0.0f ||
                        g_lookTint[2] != 0.0f;
    const float cb[12] = { static_cast<float>(g_cfg.w), static_cast<float>(g_cfg.h),
                           static_cast<float>(g_bbW), static_cast<float>(g_bbH),
                           g_cfg.blend, (g_cfg.temporal == 1 && g_runs >= 2) ? 1.0f : 0.0f,
                           g_lookGain, lookOn ? 1.0f : 0.0f,
                           g_lookTint[0], g_lookTint[1], g_lookTint[2],
                           IsSrgbFmt(g_bbFmt) ? 1.0f : 0.0f };
    ctx->UpdateSubresource(g_cb, 0, nullptr, cb, 0, 0);
    retro::BindFullscreenState(ctx);
    ctx->PSSetConstantBuffers(0, 1, &g_cb);

    const unsigned frame = g_frame++;
    const bool runNow = g_runs == 0 || g_cfg.stride <= 1 || (frame % static_cast<unsigned>(g_cfg.stride)) == 0;
    if (runNow)
    {
        // 1) backbuffer -> NCHW planar input buffer
        if (g_bbSamples > 1)
            ctx->ResolveSubresource(g_copy, 0, g_bb, 0, g_bbFmt);
        else
            ctx->CopyResource(g_copy, g_bb);
        ctx->PSSetShaderResources(0, 1, &g_copySRV);
        ctx->PSSetSamplers(0, 1, &g_linear);
        ctx->PSSetShader(g_psIn, nullptr, 0);
        ctx->OMSetRenderTargets(1, &g_inRTV, nullptr);
        const D3D11_VIEWPORT avp = { 0.0f, 0.0f, static_cast<float>(g_cfg.w),
                                     static_cast<float>(g_cfg.h) * 3.0f, 0.0f, 1.0f };
        ctx->RSSetViewports(1, &avp);
        ctx->Draw(3, 0);
        ID3D11RenderTargetView* nullRtv = nullptr;
        ctx->OMSetRenderTargets(1, &nullRtv, nullptr);
        ID3D11ShaderResourceView* nullSrv = nullptr;
        ctx->PSSetShaderResources(0, 1, &nullSrv);

        // 2) tex->buf copy, inference, buf->tex copy — all on the D3D12 queue
        //    (input-side CPU sync for v1)
        if (WaitD3D11Idle(ctx))
        {
            LARGE_INTEGER t0 = {}, t1 = {};
            QueryPerformanceCounter(&t0);
            const OrtApi* api = ai::Api();
            const bool ok = SubmitCopy(g_res12[0], g_tex12[0], true) &&
                            ai::Ok(api->RunWithBinding(g_session, nullptr, g_bind[g_parity]), "RunWithBinding") &&
                            SubmitCopy(g_tex12[1 + g_parity], g_res12[1 + g_parity], false) &&
                            WaitD3D12Idle();
            QueryPerformanceCounter(&t1);
            if (!ok)
            {
                saved.Restore(ctx);
                Fail("inference run");
                return;
            }
            if (g_runs == 0)
                Log::Info("ai", "first inference ok");
            InterlockedExchangeAdd64(&g_statTicks, t1.QuadPart - t0.QuadPart);
            InterlockedIncrement64(&g_statRuns);
            g_shown = g_parity;
            g_parity ^= 1;
            ++g_runs;
        }
    }

    // 3) model output (+ temporal lerp) -> backbuffer
    if (g_runs > 0)
    {
        ID3D11ShaderResourceView* srvs[2] = { g_outSRV[g_shown], g_outSRV[g_shown ^ 1] };
        ctx->PSSetShaderResources(0, 2, srvs);
        ctx->PSSetShader(g_psOut, nullptr, 0);
        ctx->OMSetRenderTargets(1, &g_bbRTV, nullptr);
        const D3D11_VIEWPORT bvp = { 0.0f, 0.0f, static_cast<float>(g_bbW), static_cast<float>(g_bbH), 0.0f, 1.0f };
        ctx->RSSetViewports(1, &bvp);
        ctx->Draw(3, 0);
        ID3D11ShaderResourceView* nullSrvs[2] = { nullptr, nullptr };
        ctx->PSSetShaderResources(0, 2, nullSrvs);
    }
    saved.Restore(ctx);
}

void ai::ReleaseBackbuffer() noexcept
{
    ReleaseBB();
}

void ai::TakeStats(double* avgMs, long long* runs) noexcept
{
    const LONGLONG ticks = InterlockedExchange64(&g_statTicks, 0);
    const LONGLONG n = InterlockedExchange64(&g_statRuns, 0);
    LARGE_INTEGER f = {};
    QueryPerformanceFrequency(&f);
    *runs = n;
    *avgMs = (n > 0 && f.QuadPart > 0)
                 ? static_cast<double>(ticks) * 1000.0 / static_cast<double>(f.QuadPart) / static_cast<double>(n)
                 : 0.0;
}

void ai::Reset() noexcept
{
    ReleaseBB();
    Teardown();
    Rel(g_psIn); Rel(g_psOut); Rel(g_cb); Rel(g_linear); Rel(g_query);
    g_dev = nullptr;
    g_failed = false;
}

#endif // RETRO_HAVE_ORT
