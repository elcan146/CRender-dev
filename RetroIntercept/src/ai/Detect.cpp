// Detect — see header. YOLOv8-style decode: output [1, 4+nCls, N] (or transposed
// [1, N, 4+nCls]) of cx,cy,w,h + per-class scores, normalized letterboxed coords
// (the square input is sampled stretch — output xywh are already 0..1 of the frame).
// NMS per class (IoU 0.45). Results feed term::Tracker; sight lines pair weapon-class
// boxes with their containing/nearest person box.
#include "Detect.h"
#include "../core/Log.h"

#include <Windows.h>

#if !RETRO_HAVE_ORT

void det::SetConfig(const char*, const char*, float, int, bool, float,
                    const char*, const char*, const char*, const char*, int,
                    const char*, float) noexcept {}
bool det::Ensure(ID3D11Device*) noexcept { return false; }
void det::Run(ID3D11DeviceContext*, IDXGISwapChain*) noexcept {}
const term::Track* det::Update(float, int, int* count) noexcept { if (count) *count = 0; return nullptr; }
const char* const* det::Labels() noexcept { return nullptr; }
const hud::Line* det::SightLines(int* n) noexcept { if (n) *n = 0; return nullptr; }
const char* det::Label(int) noexcept { return "OBJ"; }
bool det::Active() noexcept { return false; }
void det::ReleaseBackbuffer() noexcept {}
void det::Reset() noexcept {}

#else

#include "OrtApi.h"
#include "../d3d11/RetroPass.h"
#include "../d3d11/StateBlock.h"

#include <d3d11_1.h>
#include <d3d12.h>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>

float Clamp01(float v) noexcept { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }

namespace
{

// ---- COCO-80 label table (used to resolve DetClasses/DetWeapon names -> ids) ----
const char* kCoco[80] = {
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog",
    "horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag",
    "tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat",
    "baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup",
    "fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot",
    "hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table",
    "toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush",
};

bool IEquals(const char* a, const char* b) noexcept
{
    while (*a != 0 && *b != 0)
        if (tolower(static_cast<unsigned char>(*a++)) != tolower(static_cast<unsigned char>(*b++)))
            return false;
    return *a == 0 && *b == 0;
}

// token: COCO name (case-insensitive) or numeric class id. Returns id or -1.
int ClassId(const char* tok, size_t len) noexcept
{
    char name[48];
    if (len == 0 || len >= sizeof(name))
        return -1;
    memcpy(name, tok, len);
    name[len] = 0;
    while (len > 0 && isspace(static_cast<unsigned char>(name[len - 1]))) name[--len] = 0;
    char* s = name;
    while (isspace(static_cast<unsigned char>(*s))) ++s;
    if (*s == 0)
        return -1;
    if (isdigit(static_cast<unsigned char>(*s)))
        return atoi(s);
    for (int i = 0; i < 80; ++i)
        if (IEquals(s, kCoco[i]))
            return i;
    return -1;
}

// ---- config ----
det::Config   g_cfg;
det::Config   g_pending;
volatile LONG g_dirty   = 0;
SRWLOCK       g_cfgLock = SRWLOCK_INIT;

// Class-space: session-0 ids are as-is; session-1 ids are offset +1000 so both
// models' boxes share one tracker/label space.
constexpr int kSess1Base = 1000;
// Labels for det_weapons.onnx (JoaoAssalim YOLOv8n: guns/knife). Numeric ids are
// shown as OBJ nn for anything past this table.
const char* kSess1Names[] = { "GUN", "KNIFE" };
constexpr int kSess1NamesN = static_cast<int>(sizeof(kSess1Names) / sizeof(kSess1Names[0]));

// ---- per-session state ----
struct Sess
{
    char            model[260] = {};
    OrtSession*     session = nullptr;
    OrtIoBinding*   bind    = nullptr;
    void*           inAlloc = nullptr;
    OrtValue*       inVal   = nullptr;
    ID3D12Resource* inBuf   = nullptr;
    ID3D12Resource* inTex12 = nullptr;
    ID3D11Texture2D* inTex11 = nullptr;
    ID3D11RenderTargetView* inRTV = nullptr;
    char            inName[128] = {};
    int             inW = 0, inH = 0;    // model input dims
    int             nCls = 0;            // 4 + nCls rows in the output
    bool            obb = false;         // YOLO-OBB layout: 4 + nCls + 1(angle) rows
    bool            classifier = false;  // sess2: [1,C] logits, P(weapon|crop)
    // bb-normalized source rect fed into this session's input — sess1 cascades
    // onto the previous pass's best person box (optical zoom for weapons).
    float           rectX = 0.0f, rectY = 0.0f, rectW = 1.0f, rectH = 1.0f;
};

// ---- device/session state ----
bool          g_failed = false;
bool          g_ready  = false;
ID3D11Device* g_dev    = nullptr; // non-owning
Sess          g_sess[3];

ID3D12Device*              g_d12      = nullptr;
ID3D12CommandQueue*        g_queue    = nullptr;
ID3D12CommandAllocator*    g_cmdAlloc = nullptr;
ID3D12GraphicsCommandList* g_cmdList  = nullptr;
ID3D12Fence*               g_fence    = nullptr;
UINT64                     g_fenceVal = 0;
HANDLE                     g_fenceEv  = nullptr;
IDMLDevice*                g_dml      = nullptr;

OrtEnv*            g_env     = nullptr;
OrtSessionOptions* g_opts    = nullptr;
OrtMemoryInfo*     g_gpuInfo = nullptr;
OrtMemoryInfo*     g_cpuInfo = nullptr;

// ---- D3D11 pass objects ----
ID3D11PixelShader*    g_psIn   = nullptr;
ID3D11Buffer*         g_cb     = nullptr;
ID3D11SamplerState*   g_linear = nullptr;
ID3D11Query*          g_query  = nullptr;

// ---- backbuffer chain (detector samples the RAW frame, before Terminator base) ----
ID3D11Texture2D*          g_bb      = nullptr;
ID3D11Texture2D*          g_copy    = nullptr;
ID3D11ShaderResourceView* g_copySRV = nullptr;
UINT                      g_bbW = 0, g_bbH = 0, g_bbSamples = 1;
DXGI_FORMAT               g_bbFmt = DXGI_FORMAT_UNKNOWN;

// ---- per-frame decode state ----
unsigned    g_frame = 0;
float       g_runMs = 0.0f;  // measured cost of the last full inference pass
float       g_bestScore = 0.0f;
int         g_rawCand = 0; // best class score seen (even below conf) — diag
term::Det   g_dets[64];
int         g_nDets = 0;
term::Tracker g_trk;
const char* g_labels[term::Tracker::kMax] = {};
hud::Line   g_sight[32];
int         g_nSight = 0;

const char kPsInSrc[] =
    "Texture2D src : register(t0);"
    "SamplerState smp : register(s0);"
    "cbuffer P : register(b0) { float2 inDim; float2 bbDim; float srgbOn; float3 pad; float4 srcRect; };"
    "float4 main(float4 p : SV_Position) : SV_Target {"
    "  uint h = (uint)inDim.y;"
    "  uint plane = (uint)p.y / h;"
    "  float2 uv = float2(p.x, p.y - plane * h) / inDim;"
    "  float inA = inDim.x / inDim.y, bbA = bbDim.x / bbDim.y;"  // aspect-preserving letterbox
    "  float cw = min(1.0, bbA / inA), ch = min(1.0, inA / bbA);"
    "  float2 luv = float2((uv.x - (1.0 - cw) * 0.5) / cw, (uv.y - (1.0 - ch) * 0.5) / ch);"
    "  float2 suv = srcRect.xy + luv * srcRect.zw;"                    // map into crop region
    "  float3 c = (luv.x >= 0.0 && luv.x <= 1.0 && luv.y >= 0.0 && luv.y <= 1.0)"
    "           ? saturate(src.SampleLevel(smp, suv, 0).rgb) : float3(0.0, 0.0, 0.0);"
    "  if (srgbOn > 0.5) c = pow(c, 0.4545);"                    // undo _SRGB linearize
    "  return float4(plane == 0 ? c.r : (plane == 1 ? c.g : c.b), 0.0, 0.0, 0.0);"
    "}";

template <class T>
void Rel(T*& p) noexcept
{
    if (p != nullptr) { p->Release(); p = nullptr; }
}

bool Fail(const char* what, HRESULT hr = S_OK) noexcept
{
    Log::Warn("det", "det fail: %s (hr=0x%08lX) — detector disabled until F11", what,
              static_cast<unsigned long>(hr));
    g_failed = true;
    return false;
}

void ApplyPendingConfig() noexcept
{
    if (InterlockedCompareExchange(&g_dirty, 0, 0) == 0)
        return;
    AcquireSRWLockExclusive(&g_cfgLock);
    const det::Config p = g_pending;
    InterlockedExchange(&g_dirty, 0);
    ReleaseSRWLockExclusive(&g_cfgLock);
    if (strcmp(p.model, g_cfg.model) != 0 || strcmp(p.model2, g_cfg.model2) != 0)
    {
        g_cfg = p;
        g_failed = false; // model change retries the session build
        if (g_ready) { det::Reset(); }
    }
    else
        g_cfg = p;
}

void ReleaseBB() noexcept
{
    Rel(g_copySRV);
    Rel(g_copy);
    Rel(g_bb);
    g_bbW = g_bbH = 0;
}

void TeardownSess(Sess& s) noexcept
{
    const OrtApi* api = ai::Api();
    const OrtDmlApi* dml = ai::Dml();
    if (api != nullptr)
    {
        if (s.bind != nullptr) { api->ReleaseIoBinding(s.bind); s.bind = nullptr; }
        if (s.inVal != nullptr) { api->ReleaseValue(s.inVal); s.inVal = nullptr; }
        if (s.session != nullptr) { api->ReleaseSession(s.session); s.session = nullptr; }
    }
    if (s.inAlloc != nullptr && dml != nullptr) ai::Ok(dml->FreeGPUAllocation(s.inAlloc), "FreeGPUAllocation");
    s.inAlloc = nullptr;
    Rel(s.inRTV);
    Rel(s.inTex11);
    Rel(s.inTex12);
    Rel(s.inBuf);
    s.inW = s.inH = s.nCls = 0;
}

void Teardown() noexcept
{
    const OrtApi* api = ai::Api();
    for (Sess& s : g_sess)
        TeardownSess(s);
    if (api != nullptr)
    {
        if (g_opts != nullptr) { api->ReleaseSessionOptions(g_opts); g_opts = nullptr; }
        if (g_gpuInfo != nullptr) { api->ReleaseMemoryInfo(g_gpuInfo); g_gpuInfo = nullptr; }
        if (g_cpuInfo != nullptr) { api->ReleaseMemoryInfo(g_cpuInfo); g_cpuInfo = nullptr; }
        if (g_env != nullptr) { api->ReleaseEnv(g_env); g_env = nullptr; }
    }
    Rel(g_cmdList);
    Rel(g_cmdAlloc);
    Rel(g_fence);
    Rel(g_queue);
    Rel(g_dml);
    Rel(g_d12);
    if (g_fenceEv != nullptr) { CloseHandle(g_fenceEv); g_fenceEv = nullptr; }
    g_fenceVal = 0;
    g_ready = false;
    g_trk.Clear();
    g_nDets = 0;
    g_nSight = 0;
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
    HRESULT hr = ai::D3D12Create()(ad, D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device),
                                   reinterpret_cast<void**>(&g_d12));
    ad->Release();
    if (FAILED(hr) || g_d12 == nullptr)
        return Fail("D3D12CreateDevice", hr);

    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (FAILED(hr = g_d12->CreateCommandQueue(&qd, IID_PPV_ARGS(&g_queue))) ||
        FAILED(hr = g_d12->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&g_fence))) ||
        FAILED(hr = g_d12->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&g_cmdAlloc))) ||
        FAILED(hr = g_d12->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, g_cmdAlloc, nullptr,
                                           IID_PPV_ARGS(&g_cmdList))))
        return Fail("D3D12 objects", hr);
    g_cmdList->Close();
    g_fenceEv = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (g_fenceEv == nullptr)
        return Fail("CreateEvent");
    if (FAILED(hr = ai::DmlCreateDevice()(g_d12, DML_CREATE_DEVICE_FLAG_NONE, IID_PPV_ARGS(&g_dml))))
        return Fail("DMLCreateDevice", hr);
    return true;
}

// Reads the model's declared input dims; fills s.inW/s.inH + s.inName.
bool ReadInputShape(Sess& s) noexcept
{
    const OrtApi* api = ai::Api();
    OrtAllocator* alloc = nullptr;
    if (!ai::Ok(api->GetAllocatorWithDefaultOptions(&alloc), "GetAllocatorWithDefaultOptions"))
        return Fail("allocator");
    char* nm = nullptr;
    OrtStatus* st = api->SessionGetInputName(s.session, 0, alloc, &nm);
    if (!ai::Ok(st, "SessionGetInputName") || nm == nullptr)
        return Fail("input name");
    strncpy_s(s.inName, nm, _TRUNCATE);
    api->AllocatorFree(alloc, nm);

    OrtTypeInfo* ti = nullptr;
    st = api->SessionGetInputTypeInfo(s.session, 0, &ti);
    if (!ai::Ok(st, "SessionGetInputTypeInfo") || ti == nullptr)
        return Fail("input type info");
    const OrtTensorTypeAndShapeInfo* tsi = nullptr;
    ONNXTensorElementDataType et = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    size_t nd = 0;
    int64_t dims[8] = {};
    const bool ok = ai::Ok(api->CastTypeInfoToTensorInfo(ti, &tsi), "CastTypeInfoToTensorInfo") &&
                    tsi != nullptr &&
                    ai::Ok(api->GetTensorElementType(tsi, &et), "GetTensorElementType") &&
                    ai::Ok(api->GetDimensionsCount(tsi, &nd), "GetDimensionsCount") && nd == 4 &&
                    ai::Ok(api->GetDimensions(tsi, dims, nd), "GetDimensions");
    api->ReleaseTypeInfo(ti);
    if (!ok || et != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || dims[0] != 1 || dims[1] != 3 ||
        dims[2] <= 0 || dims[3] <= 0)
        return Fail("detector input must be fixed [1,3,H,W] fp32");
    s.inH = static_cast<int>(dims[2]);
    s.inW = static_cast<int>(dims[3]);
    if ((s.inW % 64) != 0)
        return Fail("detector input width must be a multiple of 64 (row pitch)");
    return true;
}

// Output must be [1, R, N] or [1, N, R] with R = 4 + nCls, R < N.
bool ReadOutputShape(Sess& s) noexcept
{
    const OrtApi* api = ai::Api();
    size_t nOut = 0;
    if (!ai::Ok(api->SessionGetOutputCount(s.session, &nOut), "SessionGetOutputCount") || nOut != 1)
        return Fail("detector must have exactly 1 output (export without NMS)");
    OrtTypeInfo* ti = nullptr;
    if (!ai::Ok(api->SessionGetOutputTypeInfo(s.session, 0, &ti), "SessionGetOutputTypeInfo") || ti == nullptr)
        return Fail("output type info");
    const OrtTensorTypeAndShapeInfo* tsi = nullptr;
    ONNXTensorElementDataType et = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    size_t nd = 0;
    int64_t dims[8] = {};
    const bool ok = ai::Ok(api->CastTypeInfoToTensorInfo(ti, &tsi), "CastTypeInfoToTensorInfo") &&
                    tsi != nullptr &&
                    ai::Ok(api->GetTensorElementType(tsi, &et), "GetTensorElementType") &&
                    ai::Ok(api->GetDimensionsCount(tsi, &nd), "GetDimensionsCount") &&
                    (s.classifier ? nd == 2 : nd == 3) &&
                    ai::Ok(api->GetDimensions(tsi, dims, nd), "GetDimensions");
    api->ReleaseTypeInfo(ti);
    if (s.classifier)
    {
        // Micro-classifier: [1, C] logits — C==1 (sigmoid) or C>=2 (softmax)
        if (!ok || et != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || dims[0] != 1 ||
            dims[1] < 1 || dims[1] > 8)
            return Fail("classifier output must be fp32 [1,C]");
        s.nCls = static_cast<int>(dims[1]);
        return true;
    }
    if (!ok || et != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || dims[0] != 1)
        return Fail("detector output must be fp32 [1,R,N]");
    const int64_t r = dims[1] < dims[2] ? dims[1] : dims[2]; // rows = 4 + nCls (the smaller dim)
    if (r < 5 || r > 4 + 255)
        return Fail("unexpected detector output layout");
    s.nCls = static_cast<int>(r) - (s.obb ? 5 : 4);
    return true;
}

bool CreateSession(Sess& s, int idx) noexcept
{
    const OrtApi* api = ai::Api();
    const OrtDmlApi* dml = ai::Dml();

    char full[MAX_PATH] = {};
    const bool absolute = s.model[0] == '\\' || s.model[0] == '/' ||
                          (s.model[0] != 0 && s.model[1] == ':');
    if (absolute)
        strncpy_s(full, s.model, _TRUNCATE);
    else
        _snprintf_s(full, _TRUNCATE, "%s\\%s", Log::ModuleDir(), s.model);
    if (GetFileAttributesA(full) == INVALID_FILE_ATTRIBUTES)
    {
        Log::Warn("det", "detector model %d not found: %s — session off", idx, full);
        return false; // missing model2 shouldn't kill session 0
    }
    wchar_t wfull[MAX_PATH] = {};
    MultiByteToWideChar(CP_UTF8, 0, full, -1, wfull, MAX_PATH);

    if (g_env == nullptr &&
        (!ai::Ok(api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "retro-det", &g_env), "CreateEnv") ||
         !ai::Ok(api->CreateSessionOptions(&g_opts), "CreateSessionOptions") ||
         !ai::Ok(api->DisableMemPattern(g_opts), "DisableMemPattern") ||
         !ai::Ok(api->SetSessionExecutionMode(g_opts, ORT_SEQUENTIAL), "SetSessionExecutionMode") ||
         !ai::Ok(api->SetIntraOpNumThreads(g_opts, 1), "SetIntraOpNumThreads") ||
         !ai::Ok(api->SetSessionGraphOptimizationLevel(g_opts, ORT_ENABLE_ALL), "SetGraphOptimizationLevel") ||
         !ai::Ok(api->CreateMemoryInfo("DML", OrtDeviceAllocator, 0, OrtMemTypeDefault, &g_gpuInfo),
                 "CreateMemoryInfo(DML)") ||
         !ai::Ok(api->CreateMemoryInfo("Cpu", OrtArenaAllocator, 0, OrtMemTypeDefault, &g_cpuInfo),
                 "CreateMemoryInfo(Cpu)") ||
         !ai::Ok(dml->SessionOptionsAppendExecutionProvider_DML1(g_opts, g_dml, g_queue),
                 "AppendExecutionProvider_DML1")))
        return Fail("session options/env");

    LARGE_INTEGER f = {}, t0 = {}, t1 = {};
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t0);
    if (!ai::Ok(api->CreateSession(g_env, wfull, g_opts, &s.session), "CreateSession") || s.session == nullptr)
        return Fail("CreateSession");
    QueryPerformanceCounter(&t1);
    Log::Info("det", "detector session %s loaded in %.0f ms", full,
              static_cast<double>(t1.QuadPart - t0.QuadPart) * 1000.0 / static_cast<double>(f.QuadPart));

    if (!ReadInputShape(s) || !ReadOutputShape(s))
        return false;
    Log::Info("det", "detector[%d] in=%s [1,3,%d,%d] classes=%d", idx, s.inName, s.inH, s.inW, s.nCls);
    return true;
}

bool CreateInterop(Sess& s) noexcept
{
    const OrtApi* api = ai::Api();
    const OrtDmlApi* dml = ai::Dml();
    const UINT64 bytes = static_cast<UINT64>(s.inW) * static_cast<UINT64>(s.inH) * 3u * sizeof(float);

    // Private D3D12 buffer = the DML input tensor.
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
    rd.Flags            = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    HRESULT hr = g_d12->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                                                D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                IID_PPV_ARGS(&s.inBuf));
    if (FAILED(hr))
        return Fail("detector input buffer", hr);

    // Shared R32_FLOAT W x 3H texture — the D3D11<->D3D12 bridge (same trick as AiPass).
    rd.Dimension        = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width            = static_cast<UINT64>(s.inW);
    rd.Height           = static_cast<UINT>(s.inH) * 3;
    rd.Format           = DXGI_FORMAT_R32_FLOAT;
    rd.Layout           = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    rd.Flags            = D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS |
                          D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    hr = g_d12->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_SHARED, &rd,
                                        D3D12_RESOURCE_STATE_COMMON, nullptr,
                                        IID_PPV_ARGS(&s.inTex12));
    if (FAILED(hr))
        return Fail("detector shared texture", hr);
    HANDLE h = nullptr;
    if (FAILED(hr = g_d12->CreateSharedHandle(s.inTex12, nullptr, GENERIC_ALL, nullptr, &h)) || h == nullptr)
        return Fail("CreateSharedHandle", hr);
    ID3D11Device1* d1 = nullptr;
    hr = g_dev->QueryInterface(IID_PPV_ARGS(&d1));
    if (SUCCEEDED(hr) && d1 != nullptr)
    {
        hr = d1->OpenSharedResource1(h, IID_PPV_ARGS(&s.inTex11));
        d1->Release();
    }
    CloseHandle(h);
    if (FAILED(hr) || s.inTex11 == nullptr)
        return Fail("OpenSharedResource1", hr);
    if (FAILED(hr = g_dev->CreateRenderTargetView(s.inTex11, nullptr, &s.inRTV)))
        return Fail("input RTV", hr);

    if (!ai::Ok(dml->CreateGPUAllocationFromD3DResource(s.inBuf, &s.inAlloc), "CreateGPUAllocationFromD3DResource"))
        return Fail("GPU allocation");
    const int64_t shape[4] = { 1, 3, s.inH, s.inW };
    if (!ai::Ok(api->CreateTensorWithDataAsOrtValue(g_gpuInfo, s.inAlloc, static_cast<size_t>(bytes),
                                                    shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &s.inVal),
                "CreateTensorWithDataAsOrtValue"))
        return Fail("input tensor value");
    if (!ai::Ok(api->CreateIoBinding(s.session, &s.bind), "CreateIoBinding") ||
        !ai::Ok(api->BindInput(s.bind, s.inName, s.inVal), "BindInput"))
        return Fail("io binding");
    return true;
}

bool CreatePassObjects() noexcept
{
    if (g_psIn != nullptr && g_cb != nullptr && g_linear != nullptr && g_query != nullptr)
        return true;
    ID3DBlob* b = retro::CompileShader(kPsInSrc, "main", "ps_5_0");
    if (b == nullptr)
        return Fail("compile input PS");
    HRESULT hr = g_dev->CreatePixelShader(b->GetBufferPointer(), b->GetBufferSize(), nullptr, &g_psIn);
    b->Release();
    if (FAILED(hr))
        return Fail("create input PS", hr);
    D3D11_BUFFER_DESC cd = {};
    cd.ByteWidth = 48;
    cd.Usage     = D3D11_USAGE_DEFAULT;
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
    if (bb == g_bb && g_copySRV != nullptr)
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
    cd.SampleDesc.Count = 1;
    cd.SampleDesc.Quality = 0;
    HRESULT hTex = g_dev->CreateTexture2D(&cd, nullptr, &g_copy);
    if (FAILED(hTex) || g_copy == nullptr)
    {
        Log::Warn("det", "bb copy tex create failed hr=0x%08lX fmt=%u",
                  static_cast<unsigned long>(hTex), bd.Format);
        bb->Release();
        ReleaseBB();
        return false;
    }
    HRESULT hSrv = g_dev->CreateShaderResourceView(g_copy, nullptr, &g_copySRV);
    if (FAILED(hSrv) || g_copySRV == nullptr)
    {
        Log::Warn("det", "bb copy SRV create failed hr=0x%08lX fmt=%u",
                  static_cast<unsigned long>(hSrv), bd.Format);
        bb->Release();
        ReleaseBB();
        return false;
    }
    g_bb = bb;
    g_bbW = bd.Width; g_bbH = bd.Height;
    g_bbSamples = bd.SampleDesc.Count;
    g_bbFmt = bd.Format;
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
            return false;
        SwitchToThread();
    }
    return true;
}

bool WaitD3D12Idle() noexcept
{
    const UINT64 v = ++g_fenceVal;
    if (FAILED(g_queue->Signal(g_fence, v)))
        return false;
    if (g_fence->GetCompletedValue() >= v)
        return true;
    if (FAILED(g_fence->SetEventOnCompletion(v, g_fenceEv)))
        return false;
    return WaitForSingleObject(g_fenceEv, 2000) == WAIT_OBJECT_0;
}

// shared tex -> tensor buffer on our queue, then fence the batch.
bool SubmitInputCopy(Sess& s) noexcept
{
    const UINT w = static_cast<UINT>(s.inW), h3 = static_cast<UINT>(s.inH) * 3;
    D3D12_TEXTURE_COPY_LOCATION texLoc = {};
    texLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    texLoc.pResource = s.inTex12;
    texLoc.SubresourceIndex = 0;
    D3D12_TEXTURE_COPY_LOCATION bufLoc = {};
    bufLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    bufLoc.pResource = s.inBuf;
    bufLoc.PlacedFootprint.Offset = 0;
    bufLoc.PlacedFootprint.Footprint.Format   = DXGI_FORMAT_R32_FLOAT;
    bufLoc.PlacedFootprint.Footprint.Width    = w;
    bufLoc.PlacedFootprint.Footprint.Height   = h3;
    bufLoc.PlacedFootprint.Footprint.Depth    = 1;
    bufLoc.PlacedFootprint.Footprint.RowPitch = w * sizeof(float);
    if (FAILED(g_cmdAlloc->Reset()) || FAILED(g_cmdList->Reset(g_cmdAlloc, nullptr)))
        return false;
    g_cmdList->CopyTextureRegion(&bufLoc, 0, 0, 0, &texLoc, nullptr);
    if (FAILED(g_cmdList->Close()))
        return false;
    ID3D12CommandList* lists[] = { g_cmdList };
    g_queue->ExecuteCommandLists(1, lists);
    return true;
}

// out: fp32 [1, R, N] or [1, N, R], R = 4 + nCls. Keeps the session's configured
// classes only; session 1 ids are stored offset +kSess1Base.
void Decode(const Sess& s, int idx, const float* out, size_t elems, int64_t d1, int64_t d2) noexcept
{
    const bool channelsFirst = d1 < d2;            // [1,R,N] vs [1,N,R]
    const int R = static_cast<int>(channelsFirst ? d1 : d2);
    const int N = static_cast<int>(channelsFirst ? d2 : d1);
    if (R != s.nCls + 4 || N <= 0 || static_cast<size_t>(R) * static_cast<size_t>(N) > elems)
        return;
    const int* classList = idx == 0 ? g_cfg.classes   : g_cfg.classes2;
    const int  nClasses  = idx == 0 ? g_cfg.nClasses  : g_cfg.nClasses2;
    const int  clsBase   = idx == 0 ? 0 : kSess1Base;

    auto at = [&](int r, int c) -> float
    {
        return channelsFirst ? out[static_cast<size_t>(r) * N + c]
                             : out[static_cast<size_t>(c) * R + r];
    };

    // Ultralytics exports emit input-PIXEL coords (0..inW/inH); some pipelines
    // normalize to 0..1 — sniff the first columns and scale accordingly.
    float xS = 1.0f, yS = 1.0f;
    {
        float mx = 0.0f;
        for (int c = 0; c < N && c < 64; ++c)
            for (int r = 0; r < 4; ++r)
            {
                const float v = at(r, c);
                if (v > mx)
                    mx = v;
            }
        if (mx > 1.5f) { xS = 1.0f / static_cast<float>(s.inW); yS = 1.0f / static_cast<float>(s.inH); }
    }

    term::Det raw[64];
    int nRaw = 0;
    for (int c = 0; c < N && nRaw < 64; ++c)
    {
        int bestCls = -1;
        float best = g_cfg.conf;
        for (int k = 0; k < nClasses; ++k)
        {
            const int cls = classList[k];
            if (cls < 0 || cls >= s.nCls)
                continue;
            const float sc = at(4 + cls, c);
            if (sc > g_bestScore)
                g_bestScore = sc;
            if (sc > best) { best = sc; bestCls = cls; }
        }
        if (bestCls < 0)
            continue;
        ++g_rawCand;
        const float cx = at(0, c) * xS, cy = at(1, c) * yS;
        const float bw = at(2, c) * xS, bh = at(3, c) * yS;
        // Undo the input letterbox (content cw x ch, centered), then map the
        // crop-space box back to screen space through the session's src rect.
        const float inA = static_cast<float>(s.inW) / static_cast<float>(s.inH);
        const float srcH = s.rectH * static_cast<float>(g_bbH);
        const float bbA = srcH > 0.0f
                        ? (s.rectW * static_cast<float>(g_bbW)) / srcH : inA;
        const float cw = (bbA / inA < 1.0f) ? bbA / inA : 1.0f;
        const float ch = (inA / bbA < 1.0f) ? inA / bbA : 1.0f;
        const float px = (1.0f - cw) * 0.5f, py = (1.0f - ch) * 0.5f;
        float u0, u1, v0, v1;
        float ang = 0.0f;
        if (s.obb)
        {
            // OBB: the extra last row is the angle (radians). Rotate the
            // half-extent axes in input space (square input => isotropic),
            // map each rotated corner through letterbox+crop, take envelope.
            ang = at(4 + s.nCls, c);
            const float ca = cosf(ang), sa = sinf(ang);
            const float ax = ca * bw * 0.5f, ay = sa * bw * 0.5f;   // w axis
            const float bx = -sa * bh * 0.5f, by = ca * bh * 0.5f;  // h axis
            u0 = u1 = s.rectX + (cx - px) / cw * s.rectW;
            v0 = v1 = s.rectY + (cy - py) / ch * s.rectH;
            for (int q = 0; q < 4; ++q)
            {
                const float su = (q & 1) ? 1.0f : -1.0f, sv = (q & 2) ? 1.0f : -1.0f;
                const float mu = s.rectX + (cx + ax * su + bx * sv - px) / cw * s.rectW;
                const float mv = s.rectY + (cy + ay * su + by * sv - py) / ch * s.rectH;
                if (mu < u0) u0 = mu;
                if (mu > u1) u1 = mu;
                if (mv < v0) v0 = mv;
                if (mv > v1) v1 = mv;
            }
        }
        else
        {
            u0 = s.rectX + (cx - bw * 0.5f - px) / cw * s.rectW;
            u1 = s.rectX + (cx + bw * 0.5f - px) / cw * s.rectW;
            v0 = s.rectY + (cy - bh * 0.5f - py) / ch * s.rectH;
            v1 = s.rectY + (cy + bh * 0.5f - py) / ch * s.rectH;
        }
        term::Det& d = raw[nRaw];
        d.x0 = Clamp01(u0); d.y0 = Clamp01(v0);
        d.x1 = Clamp01(u1); d.y1 = Clamp01(v1);
        d.score = best;
        d.cls = bestCls + clsBase;
        d.ang = ang;
        if (d.x1 - d.x0 < 0.002f || d.y1 - d.y0 < 0.002f)
            continue;
        // Crop-fill rejection: a deferred sess1 scans person-anchored crops and
        // a 1-class weapons model trained on weapon-dominated frames hallucinates
        // "gun" spanning the whole person. Real weapons occupy a modest slice of
        // the already-expanded crop — drop boxes that fill most of it.
        if (idx == 1 && s.rectW < 0.99f &&
            (d.x1 - d.x0) * (d.y1 - d.y0) > 0.60f * s.rectW * s.rectH)
            continue;
        ++nRaw;
    }

    // Score-descending order — NMS must let the strongest box suppress weaker
    // overlaps; decode order is arbitrary so sort first or duplicates stack.
    for (int i = 1; i < nRaw; ++i)
    {
        const term::Det v = raw[i];
        int j = i - 1;
        while (j >= 0 && raw[j].score < v.score) { raw[j + 1] = raw[j]; --j; }
        raw[j + 1] = v;
    }

    // Per-class NMS (IoU 0.45), best score first.
    bool dead[64] = {};
    for (int i = 0; i < nRaw; ++i)
    {
        if (dead[i])
            continue;
        for (int j = i + 1; j < nRaw; ++j)
        {
            if (dead[j] || raw[j].cls != raw[i].cls || raw[j].score > raw[i].score)
                continue;
            const float ix0 = raw[i].x0 > raw[j].x0 ? raw[i].x0 : raw[j].x0;
            const float iy0 = raw[i].y0 > raw[j].y0 ? raw[i].y0 : raw[j].y0;
            const float ix1 = raw[i].x1 < raw[j].x1 ? raw[i].x1 : raw[j].x1;
            const float iy1 = raw[i].y1 < raw[j].y1 ? raw[i].y1 : raw[j].y1;
            if (ix1 <= ix0 || iy1 <= iy0)
                continue;
            const float inter = (ix1 - ix0) * (iy1 - iy0);
            const float ua = (raw[i].x1 - raw[i].x0) * (raw[i].y1 - raw[i].y0) +
                             (raw[j].x1 - raw[j].x0) * (raw[j].y1 - raw[j].y0) - inter;
            if (ua > 0.0f && inter / ua > 0.45f)
                dead[j] = true;
        }
    }
    for (int i = 0; i < nRaw && g_nDets < g_cfg.maxDets; ++i)
        if (!dead[i])
            g_dets[g_nDets++] = raw[i];
}

// Runs ORT on s's already-rendered input. Detectors decode into g_dets;
// classifiers return P(weapon) via clsProb (softmax cls-1 / sigmoid single).
bool InferSess(Sess& s, int idx, OrtAllocator* alloc, float* clsProb) noexcept
{
    const OrtApi* api = ai::Api();
    if (!SubmitInputCopy(s))
    {
        Log::Warn("det", "tex->buf copy failed sess=%d", idx);
        return false;
    }
    api->ClearBoundOutputs(s.bind); // void
    // Bind every declared output to CPU memory (GetBoundOutputValues returns them all).
    size_t nOut = 0;
    if (!ai::Ok(api->SessionGetOutputCount(s.session, &nOut), "SessionGetOutputCount"))
        return false;
    bool bound = true;
    for (size_t k = 0; k < nOut; ++k)
    {
        char* nm = nullptr;
        if (!ai::Ok(api->SessionGetOutputName(s.session, k, alloc, &nm), "output name") || nm == nullptr)
        { bound = false; break; }
        OrtStatus* st = api->BindOutputToDevice(s.bind, nm, g_cpuInfo);
        api->AllocatorFree(alloc, nm);
        if (!ai::Ok(st, "BindOutputToDevice"))
        { bound = false; break; }
    }
    if (!bound)
        return false;
    if (!ai::Ok(api->RunWithBinding(s.session, nullptr, s.bind), "RunWithBinding"))
    {
        Fail("inference run");
        return false;
    }
    if (!WaitD3D12Idle())
        return false;

    OrtValue** vals = nullptr;
    size_t nVals = 0;
    if (!ai::Ok(api->GetBoundOutputValues(s.bind, alloc, &vals, &nVals),
                "GetBoundOutputValues") || vals == nullptr || nVals == 0)
        return false;
    const float* data = nullptr;
    OrtTensorTypeAndShapeInfo* tsi = nullptr;
    if (ai::Ok(api->GetTensorMutableData(vals[0], reinterpret_cast<void**>(
               const_cast<float**>(&data))), "GetTensorMutableData") &&
        data != nullptr &&
        ai::Ok(api->GetTensorTypeAndShape(vals[0], &tsi), "GetTensorTypeAndShape") &&
        tsi != nullptr)
    {
        int64_t dims[4] = {};
        size_t nd = 0;
        if (ai::Ok(api->GetDimensionsCount(tsi, &nd), "GetDimensionsCount") &&
            ai::Ok(api->GetDimensions(tsi, dims, nd), "GetDimensions"))
        {
            if (s.classifier)
            {
                if (nd == 2 && dims[0] == 1 && dims[1] >= 1 && clsProb != nullptr)
                {
                    const int C = static_cast<int>(dims[1]);
                    if (C == 1)
                        *clsProb = 1.0f / (1.0f + expf(-data[0]));
                    else
                    {
                        float mx = data[0];
                        for (int k = 1; k < C; ++k)
                            if (data[k] > mx) mx = data[k];
                        float sum = 0.0f;
                        for (int k = 0; k < C; ++k) sum += expf(data[k] - mx);
                        *clsProb = expf(data[1] - mx) / sum; // class 1 = weapon
                    }
                }
            }
            else if (nd == 3)
            {
                const int64_t d1 = dims[1], d2 = dims[2];
                Decode(s, idx, data, static_cast<size_t>(d1) * static_cast<size_t>(d2), d1, d2);
            }
        }
        api->ReleaseTensorTypeAndShapeInfo(tsi);
    }
    for (size_t k = 0; k < nVals; ++k)
        api->ReleaseValue(vals[k]);
    api->AllocatorFree(alloc, vals);
    return true;
}

} // namespace

bool IsWeaponCls(int cls) noexcept
{
    if (cls >= kSess1Base)
        return cls - kSess1Base < 256 && g_cfg.weapons2[cls - kSess1Base];
    return cls >= 0 && cls < 256 && g_cfg.weapons[cls];
}

// Parses "a,b,c" of COCO names / numeric ids into ids[] (max 16). Returns count.
int ParseCsvIds(const char* csv, int* ids) noexcept
{
    int n = 0;
    if (csv == nullptr)
        return 0;
    for (const char* p = csv; *p != 0 && n < 16;)
    {
        const char* comma = strchr(p, ',');
        const size_t len = comma != nullptr ? static_cast<size_t>(comma - p) : strlen(p);
        const int id = ClassId(p, len);
        if (id >= 0)
            ids[n++] = id;
        p = comma != nullptr ? comma + 1 : p + len;
    }
    return n;
}

void ParseCsvWeapons(const char* csv, bool* flags) noexcept
{
    memset(flags, 0, 256);
    if (csv == nullptr)
        return;
    int ids[16];
    const int n = ParseCsvIds(csv, ids);
    for (int i = 0; i < n; ++i)
        if (ids[i] < 256)
            flags[ids[i]] = true;
}

void det::SetConfig(const char* model, const char* model2, float conf, int stride,
                    bool sight, float sightLen,
                    const char* classesCsv, const char* weaponsCsv,
                    const char* classes2Csv, const char* weapons2Csv, int dump,
                    const char* model3, float clsConf) noexcept
{
    AcquireSRWLockExclusive(&g_cfgLock);
    if (model != nullptr)
        strncpy_s(g_pending.model, model, _TRUNCATE);
    if (model2 != nullptr)
        strncpy_s(g_pending.model2, model2, _TRUNCATE);
    if (model3 != nullptr)
        strncpy_s(g_pending.model3, model3, _TRUNCATE);
    g_pending.clsConf = clsConf < 0.50f ? 0.50f : (clsConf > 0.99f ? 0.99f : clsConf);
    g_pending.conf     = conf < 0.05f ? 0.05f : (conf > 0.95f ? 0.95f : conf);
    g_pending.stride   = stride < 1 ? 1 : (stride > 30 ? 30 : stride);
    g_pending.sight    = sight;
    g_pending.sightLen = sightLen < 0.05f ? 0.05f : (sightLen > 1.5f ? 1.5f : sightLen);
    g_pending.dump     = dump;

    g_pending.nClasses = ParseCsvIds(
        (classesCsv != nullptr && classesCsv[0] != 0) ? classesCsv : "person",
        g_pending.classes);
    if (g_pending.nClasses == 0)
        g_pending.classes[g_pending.nClasses++] = 0; // person fallback
    ParseCsvWeapons(weaponsCsv, g_pending.weapons);

    g_pending.nClasses2 = ParseCsvIds(classes2Csv, g_pending.classes2);
    ParseCsvWeapons(weapons2Csv, g_pending.weapons2);

    InterlockedExchange(&g_dirty, 1);
    ReleaseSRWLockExclusive(&g_cfgLock);
    Log::Info("det", "config: model='%s' model2='%s' conf=%.2f stride=%d sight=%d len=%.2f classes=%d/%d",
              g_pending.model, g_pending.model2, g_pending.conf, g_pending.stride,
              g_pending.sight, g_pending.sightLen, g_pending.nClasses, g_pending.nClasses2);
}

bool det::Ensure(ID3D11Device* dev) noexcept
{
    if (dev == nullptr)
        return false;
    if (g_dev != nullptr && g_dev != dev)
        det::Reset();
    g_dev = dev;
    ApplyPendingConfig();
    if ((g_cfg.model[0] == 0 && g_cfg.model2[0] == 0) || g_cfg.nClasses == 0)
        return false;
    if (g_failed)
        return false;
    if (g_ready)
        return true;
    if (retro::FullscreenVS() == nullptr)
        return Fail("retro pass VS unavailable");
    if (!ai::Load(Log::ModuleDir()))
        return Fail("onnxruntime.dll / DirectML.dll load");
    if (!CreatePassObjects() || !CreateD3D12())
    {
        Teardown();
        g_failed = true;
        return false;
    }
    // Session 0 = primary (people/COCO). Session 1 = optional secondary model
    // (e.g. weapon-trained); its failure only disables that session.
    for (int i = 0; i < 3; ++i)
    {
        const char* mdl = i == 0 ? g_cfg.model : (i == 1 ? g_cfg.model2 : g_cfg.model3);
        if (mdl[0] == 0)
            continue;
        strncpy_s(g_sess[i].model, mdl, _TRUNCATE);
        g_sess[i].classifier = i == 2;
        // OBB models append an angle row ([1, 4+nCls+1, N]) — sniff the
        // filename so yolo11n-obb.onnx / det_weapons_obb.onnx decode correctly.
        {
            char low[260] = {};
            int k = 0;
            for (; mdl[k] != 0 && k < 259; ++k)
                low[k] = static_cast<char>(tolower(static_cast<unsigned char>(mdl[k])));
            g_sess[i].obb = strstr(low, "obb") != nullptr;
        }
        if (CreateSession(g_sess[i], i) && CreateInterop(g_sess[i]))
            continue;
        if (i == 0)
        {
            Teardown();
            g_failed = true;
            return false;
        }
        TeardownSess(g_sess[i]); // secondary is optional — log and drop
    }
    if (g_sess[0].session == nullptr && g_sess[1].session == nullptr)
    {
        Teardown();
        g_failed = true;
        return false;
    }
    g_ready = true;
    Log::Info("det", "detector ready: sess0=%dx%d sess1=%s cls=%s stride=%d",
              g_sess[0].inW, g_sess[0].inH,
              g_sess[1].session != nullptr ? "on" : "off",
              g_sess[2].session != nullptr ? "on" : "off", g_cfg.stride);
    return true;
}

void det::Run(ID3D11DeviceContext* ctx, IDXGISwapChain* sc) noexcept
{
    if (!g_ready || g_failed || ctx == nullptr || sc == nullptr)
        return;
    const unsigned frame = g_frame++;
    // Adaptive backoff: if a full pass costs more than ~33ms, stretch the
    // stride so detection keeps running but can't stall the game.
    int eff = g_cfg.stride;
    if (g_runMs > 33.0f)
    {
        eff = static_cast<int>(g_runMs / 33.0f + 0.5f) * g_cfg.stride;
        if (eff > 120)
            eff = 120;
    }
    if (eff > 1 && (frame % static_cast<unsigned>(eff)) != 0)
        return; // keep last decode between strides
    LARGE_INTEGER qf = {}, qt0 = {}, qt1 = {};
    QueryPerformanceFrequency(&qf);
    QueryPerformanceCounter(&qt0);
    if (!EnsureBackbuffer(sc))
        return;

    retro::StateBlock saved;
    saved.Capture(ctx);

    if (g_bbSamples > 1)
        ctx->ResolveSubresource(g_copy, 0, g_bb, 0, g_bbFmt);
    else
        ctx->CopyResource(g_copy, g_bb);
    ctx->PSSetShaderResources(0, 1, &g_copySRV);
    ctx->PSSetSamplers(0, 1, &g_linear);
    retro::BindFullscreenState(ctx);
    ctx->PSSetShader(g_psIn, nullptr, 0);
    ctx->PSSetConstantBuffers(0, 1, &g_cb);
    // sess1 defers: after sess0 decodes, every person box gets its own zoomed
    // weapons input in THIS pass — no lag, and enemies get scanned too.
    const bool deferSess1 = g_sess[0].session != nullptr && g_sess[0].inRTV != nullptr &&
                            g_sess[1].session != nullptr && g_sess[1].inRTV != nullptr;
    int live = 0;
    for (Sess& s : g_sess)
    {
        if (s.session == nullptr || s.inRTV == nullptr || s.classifier)
            continue;
        if (&s == &g_sess[1] && deferSess1)
            continue;
        s.rectX = 0.0f; s.rectY = 0.0f; s.rectW = 1.0f; s.rectH = 1.0f;
        const float cb[12] = { static_cast<float>(s.inW), static_cast<float>(s.inH),
                              static_cast<float>(g_bbW), static_cast<float>(g_bbH),
                              IsSrgbFmt(g_bbFmt) ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f,
                              0.0f, 0.0f, 1.0f, 1.0f };
        ctx->UpdateSubresource(g_cb, 0, nullptr, cb, 0, 0);
        ctx->OMSetRenderTargets(1, &s.inRTV, nullptr);
        const D3D11_VIEWPORT avp = { 0.0f, 0.0f, static_cast<float>(s.inW),
                                     static_cast<float>(s.inH) * 3.0f, 0.0f, 1.0f };
        ctx->RSSetViewports(1, &avp);
        ctx->Draw(3, 0);
        ++live;
    }
    ID3D11RenderTargetView* nullRtv = nullptr;
    ctx->OMSetRenderTargets(1, &nullRtv, nullptr);
    ID3D11ShaderResourceView* nullSrv = nullptr;
    ctx->PSSetShaderResources(0, 1, &nullSrv);
    saved.Restore(ctx);
    if (live == 0)
        return;

    if (!WaitD3D11Idle(ctx))
        return;

    // Probes at spread-out passes (intro frames are legit black): read back
    // the intermediate copy AND sess-0's sampled input, log means.
    static unsigned s_probeRuns = 0;
    static int s_probeIdx = 0;
    static const unsigned kProbeAt[] = { 30, 200, 600, 1200, 2400 };
    ++s_probeRuns;
    if (s_probeIdx < 5 && s_probeRuns == kProbeAt[s_probeIdx])
    {
        const unsigned atRun = s_probeRuns;
        ++s_probeIdx;
        auto probe = [&](ID3D11Texture2D* tex, bool isFloat, const char* tag)
        {
            if (tex == nullptr)
                return;
            D3D11_TEXTURE2D_DESC td = {};
            tex->GetDesc(&td);
            td.Usage = D3D11_USAGE_STAGING;
            td.BindFlags = 0;
            td.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
            td.MiscFlags = 0;
            ID3D11Texture2D* st = nullptr;
            if (FAILED(g_dev->CreateTexture2D(&td, nullptr, &st)) || st == nullptr)
                return;
            ctx->CopyResource(st, tex);
            D3D11_MAPPED_SUBRESOURCE ms = {};
            if (SUCCEEDED(ctx->Map(st, 0, D3D11_MAP_READ, 0, &ms)))
            {
                double sum = 0.0;
                int n = 0;
                const char* base = static_cast<const char*>(ms.pData);
                for (UINT y = 0; y < td.Height; y += 32)
                {
                    const char* rowp = base + static_cast<size_t>(y) * ms.RowPitch;
                    for (UINT x = 0; x < td.Width; x += 32)
                    {
                        if (isFloat)
                            sum += reinterpret_cast<const float*>(rowp)[x];
                        else
                            sum += static_cast<unsigned char>(rowp[x * 4]) / 255.0;
                        ++n;
                    }
                }
                ctx->Unmap(st, 0);
                Log::Info("det", "probe %s @run%u: mean=%.4f over %d samples",
                          tag, atRun, n > 0 ? sum / n : 0.0, n);
            }
            st->Release();
        };
        probe(g_copy, false, "bbcopy");
        probe(g_sess[0].inTex11, true, "in");
        if (g_cfg.dump != 0 && g_sess[0].inTex11 != nullptr)
        {
            // Reconstruct an RGB PPM from the planar float input — proves to
            // sanity_check.py exactly what the model scored on this frame.
            D3D11_TEXTURE2D_DESC td = {};
            g_sess[0].inTex11->GetDesc(&td);
            td.Usage = D3D11_USAGE_STAGING;
            td.BindFlags = 0;
            td.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
            td.MiscFlags = 0;
            ID3D11Texture2D* st = nullptr;
            if (SUCCEEDED(g_dev->CreateTexture2D(&td, nullptr, &st)) && st != nullptr)
            {
                ctx->CopyResource(st, g_sess[0].inTex11);
                D3D11_MAPPED_SUBRESOURCE ms = {};
                if (SUCCEEDED(ctx->Map(st, 0, D3D11_MAP_READ, 0, &ms)))
                {
                    char path[MAX_PATH];
                    sprintf_s(path, "det_in_%u.ppm", atRun);
                    FILE* f = nullptr;
                    if (fopen_s(&f, path, "wb") == 0 && f != nullptr)
                    {
                        const unsigned w = g_sess[0].inW, h = g_sess[0].inH;
                        fprintf(f, "P6\n%u %u\n255\n", w, h);
                        const char* base = static_cast<const char*>(ms.pData);
                        auto px = [&](unsigned x, unsigned y, unsigned pl) -> unsigned char
                        {
                            const float* row = reinterpret_cast<const float*>(
                                base + (static_cast<size_t>(pl) * h + y) * ms.RowPitch);
                            const float v = row[x] < 0.0f ? 0.0f : (row[x] > 1.0f ? 1.0f : row[x]);
                            return static_cast<unsigned char>(v * 255.0f);
                        };
                        for (unsigned y = 0; y < h; ++y)
                            for (unsigned x = 0; x < w; ++x)
                            {
                                const unsigned char rgb[3] = { px(x, y, 0), px(x, y, 1), px(x, y, 2) };
                                fwrite(rgb, 1, 3, f);
                            }
                        fclose(f);
                        Log::Info("det", "input dumped to %s", path);
                    }
                    ctx->Unmap(st, 0);
                }
                st->Release();
            }
        }
    }

    // ORT run each session: GPU input bound, outputs requested on CPU.
    OrtAllocator* alloc = nullptr;
    if (!ai::Ok(ai::Api()->GetAllocatorWithDefaultOptions(&alloc), "allocator"))
        return;
    g_nDets = 0;
    g_rawCand = 0;
    for (int i = 0; i < 2; ++i)
    {
        Sess& s = g_sess[i];
        if (s.session == nullptr)
            continue;
        // Source rects per infer: sess0 is always the full frame; a deferred
        // sess1 instead scans each person box decoded THIS pass (max 3),
        // expanded so slung weapons stay inside — full frame when none.
        float rects[3][4] = { { 0.0f, 0.0f, 1.0f, 1.0f }, { 0,0,0,0 }, { 0,0,0,0 } };
        int nRect = 1;
        if (i == 1 && deferSess1)
        {
            nRect = 0;
            for (int di = 0; di < g_nDets && nRect < 3; ++di)
            {
                if (g_dets[di].cls != g_cfg.personClass)
                    continue;
                const term::Det& p = g_dets[di];
                // Torso-anchored crop: slung rifles span shoulder→hip diagonally
                // and poke out sideways — wide margin, less leg/background.
                const float pw = p.x1 - p.x0, ph = p.y1 - p.y0;
                float rx = p.x0 - pw * 0.45f, ry = p.y0 - ph * 0.15f;
                float rw = pw * 1.90f, rh = ph * 1.15f;
                if (rx < 0.0f) rx = 0.0f;
                if (ry < 0.0f) ry = 0.0f;
                if (rx + rw > 1.0f) rw = 1.0f - rx;
                if (ry + rh > 1.0f) rh = 1.0f - ry;
                rects[nRect][0] = rx; rects[nRect][1] = ry;
                rects[nRect][2] = rw; rects[nRect][3] = rh;
                ++nRect;
            }
            if (nRect == 0)
                nRect = 1; // rects[0] is the identity — full-frame fallback
        }
        retro::StateBlock saved2;
        if (i == 1 && deferSess1)
            saved2.Capture(ctx); // StateBlock above already restored; grab it again
        for (int ri = 0; ri < nRect; ++ri)
        {
        s.rectX = rects[ri][0]; s.rectY = rects[ri][1];
        s.rectW = rects[ri][2]; s.rectH = rects[ri][3];
        if (i == 1 && deferSess1)
        {
            // Render this crop into sess1's planar input right now.
            ctx->PSSetShaderResources(0, 1, &g_copySRV);
            ctx->PSSetSamplers(0, 1, &g_linear);
            retro::BindFullscreenState(ctx);
            ctx->PSSetShader(g_psIn, nullptr, 0);
            ctx->PSSetConstantBuffers(0, 1, &g_cb);
            const float cb[12] = { static_cast<float>(s.inW), static_cast<float>(s.inH),
                                   s.rectW * static_cast<float>(g_bbW), s.rectH * static_cast<float>(g_bbH),
                                   IsSrgbFmt(g_bbFmt) ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f,
                                   s.rectX, s.rectY, s.rectW, s.rectH };
            ctx->UpdateSubresource(g_cb, 0, nullptr, cb, 0, 0);
            ctx->OMSetRenderTargets(1, &s.inRTV, nullptr);
            const D3D11_VIEWPORT avp = { 0.0f, 0.0f, static_cast<float>(s.inW),
                                         static_cast<float>(s.inH) * 3.0f, 0.0f, 1.0f };
            ctx->RSSetViewports(1, &avp);
            ctx->Draw(3, 0);
            if (!WaitD3D11Idle(ctx))
                break;
        }
        InferSess(s, i, alloc, nullptr);
        if (g_failed)
            return;
        }
        if (i == 1 && deferSess1)
            saved2.Restore(ctx);
    }
    // Cross-crop dedup: overlapping person crops can report the same weapon
    // twice — same class + IoU > 0.6 collapses to the first (score-ordered) hit.
    int wDet = 0;
    for (int a = 0; a < g_nDets; ++a)
    {
        bool dup = false;
        for (int b = 0; b < a; ++b)
        {
            const term::Det& x = g_dets[a];
            const term::Det& y = g_dets[b];
            if (x.cls != y.cls)
                continue;
            const float ix0 = x.x0 > y.x0 ? x.x0 : y.x0, iy0 = x.y0 > y.y0 ? x.y0 : y.y0;
            const float ix1 = x.x1 < y.x1 ? x.x1 : y.x1, iy1 = x.y1 < y.y1 ? x.y1 : y.y1;
            if (ix1 <= ix0 || iy1 <= iy0)
                continue;
            const float inter = (ix1 - ix0) * (iy1 - iy0);
            const float uni = (x.x1 - x.x0) * (x.y1 - y.y0) + (y.x1 - y.x0) * (y.y1 - y.y0) - inter;
            // Weapon-class boxes from different person crops of the same gun
            // diverge more than sess0 NMS survivors — merge at a lower IoU so
            // one gun can't stack duplicate brackets. Person boxes keep 0.6.
            const float dupIou = x.cls >= kSess1Base ? 0.3f : 0.6f;
            if (uni > 0.0f && inter / uni > dupIou) { dup = true; break; }
        }
        if (!dup)
            g_dets[wDet++] = g_dets[a];
    }
    g_nDets = wDet;

    // Stage-2 micro-classifier gate (sess2, optional): each surviving weapon
    // candidate is cropped from the raw frame and scored P(weapon|crop);
    // below DetClsConf it never reaches the tracker.
    if (g_sess[2].session != nullptr && g_sess[2].inRTV != nullptr && g_nDets > 0)
    {
        Sess& cs = g_sess[2];
        retro::StateBlock saved3;
        saved3.Capture(ctx);
        int wCls = 0;
        for (int a = 0; a < g_nDets; ++a)
        {
            const term::Det d = g_dets[a];
            if (!IsWeaponCls(d.cls)) { g_dets[wCls++] = d; continue; }
            const float mx = (d.x1 - d.x0) * 0.25f, my = (d.y1 - d.y0) * 0.25f;
            float rx = d.x0 - mx, ry = d.y0 - my;
            float rw = (d.x1 - d.x0) + mx * 2.0f, rh = (d.y1 - d.y0) + my * 2.0f;
            if (rx < 0.0f) rx = 0.0f;
            if (ry < 0.0f) ry = 0.0f;
            if (rx + rw > 1.0f) rw = 1.0f - rx;
            if (ry + rh > 1.0f) rh = 1.0f - ry;
            cs.rectX = rx; cs.rectY = ry; cs.rectW = rw; cs.rectH = rh;
            ctx->PSSetShaderResources(0, 1, &g_copySRV);
            ctx->PSSetSamplers(0, 1, &g_linear);
            retro::BindFullscreenState(ctx);
            ctx->PSSetShader(g_psIn, nullptr, 0);
            ctx->PSSetConstantBuffers(0, 1, &g_cb);
            const float cb[12] = { static_cast<float>(cs.inW), static_cast<float>(cs.inH),
                                   cs.rectW * static_cast<float>(g_bbW), cs.rectH * static_cast<float>(g_bbH),
                                   IsSrgbFmt(g_bbFmt) ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f,
                                   cs.rectX, cs.rectY, cs.rectW, cs.rectH };
            ctx->UpdateSubresource(g_cb, 0, nullptr, cb, 0, 0);
            ctx->OMSetRenderTargets(1, &cs.inRTV, nullptr);
            const D3D11_VIEWPORT avp = { 0.0f, 0.0f, static_cast<float>(cs.inW),
                                         static_cast<float>(cs.inH) * 3.0f, 0.0f, 1.0f };
            ctx->RSSetViewports(1, &avp);
            ctx->Draw(3, 0);
            if (!WaitD3D11Idle(ctx))
            { g_dets[wCls++] = d; break; }   // can't classify — keep it
            float p = -1.0f;
            InferSess(cs, 2, alloc, &p);
            if (g_failed) { saved3.Restore(ctx); return; }
            if (p >= g_cfg.clsConf || p < 0.0f)
                g_dets[wCls++] = d;
            else
                Log::Info("det", "cls rejected %s P=%.2f", Label(d.cls), p);
        }
        saved3.Restore(ctx);
        g_nDets = wCls;
    }
    QueryPerformanceCounter(&qt1);
    g_runMs = static_cast<float>(static_cast<double>(qt1.QuadPart - qt0.QuadPart) * 1000.0 /
                                 static_cast<double>(qf.QuadPart));
    // Periodic diag: pass cost + best candidate score (catches dead/black input
    // vs. model-almost-fires cases without per-frame spam).
    static unsigned s_runs = 0;
    if (++s_runs % 60 == 0)
    {
        Log::Info("det", "pass %.0f ms dets=%d cand=%d best=%.2f (conf %.2f)",
                  g_runMs, g_nDets, g_rawCand, g_bestScore, g_cfg.conf);
        g_bestScore = 0.0f;
    }
    // Transition log — proves detections reach the tracker without per-frame spam.
    static int s_prevDets = 0;
    if ((g_nDets > 0) != (s_prevDets > 0))
    {
        if (g_nDets > 0)
            Log::Info("det", "%d dets — top %s %.0f%%", g_nDets, Label(g_dets[0].cls),
                      g_dets[0].score * 100.0f);
        else
            Log::Info("det", "all dets dropped");
        s_prevDets = g_nDets;
    }
    static float s_slowLogged = 0.0f;
    if (g_runMs > 66.0f && g_runMs != s_slowLogged)
    {
        s_slowLogged = g_runMs;
        Log::Warn("det", "detector pass %.0f ms dets=%d — stride auto-boosted", g_runMs, g_nDets);
    }
}

const term::Track* det::Update(float dt, int maxTargets, int* count) noexcept
{
    g_trk.Update(g_dets, g_nDets, dt, maxTargets);

    // Temporal confirm gate: a track must match a detection on >=2 passes
    // before it is shown — single-pass flashes never draw a bracket.
    static term::Track s_view[term::Tracker::kMax];
    const term::Track* all = g_trk.Tracks();
    int nv = 0;
    for (int i = 0; i < g_trk.Count(); ++i)
    {
        // Flicker gate: false-positive weapon hits come and go and rarely
        // confirm past a few passes; real weapons persist. Persons confirm
        // at the usual 2.
        const int need = IsWeaponCls(all[i].cls) ? 4 : 2;
        if (all[i].hits >= need)
            s_view[nv++] = all[i];
    }
    const term::Track* tr = s_view;
    if (count != nullptr)
        *count = nv;

    // Labels parallel to the (filtered) track list.
    for (int i = 0; i < nv; ++i)
        g_labels[i] = tr[i].cls >= 0 ? Label(tr[i].cls) : nullptr;

    // Sight lines: weapon track -> owning person (containing box, else nearest).
    g_nSight = 0;
    if (g_cfg.sight)
    {
        for (int i = 0; i < nv && g_nSight < 32; ++i)
        {
            const term::Track& wT = tr[i];
            if (!IsWeaponCls(wT.cls))
                continue;
            const float wcx = (wT.x0 + wT.x1) * 0.5f, wcy = (wT.y0 + wT.y1) * 0.5f;
            const term::Track* owner = nullptr;
            float bestD = 1e9f;
            for (int j = 0; j < nv; ++j)
            {
                if (tr[j].cls != g_cfg.personClass || j == i)
                    continue;
                const term::Track& p = tr[j];
                const bool inside = wcx >= p.x0 && wcx <= p.x1 && wcy >= p.y0 && wcy <= p.y1;
                const float pcx = (p.x0 + p.x1) * 0.5f, pcy = (p.y0 + p.y1) * 0.5f;
                const float d2 = (pcx - wcx) * (pcx - wcx) + (pcy - wcy) * (pcy - wcy);
                if (inside) { owner = &p; break; }
                if (d2 < bestD) { bestD = d2; owner = &p; }
            }
            hud::Line& ln = g_sight[g_nSight];
            if (owner != nullptr)
            {
                // Owner center -> through weapon center -> extended to sightLen.
                const float pcx = (owner->x0 + owner->x1) * 0.5f;
                const float pcy = (owner->y0 + owner->y1) * 0.5f;
                float dx = wcx - pcx, dy = wcy - pcy;
                const float len = sqrtf(dx * dx + dy * dy);
                if (len < 0.01f) { dx = 0.0f; dy = 1.0f; } else { dx /= len; dy /= len; }
                ln.x0 = pcx; ln.y0 = pcy;
                ln.x1 = Clamp01(wcx + dx * g_cfg.sightLen);
                ln.y1 = Clamp01(wcy + dy * g_cfg.sightLen);
            }
            else
            {
                // Unowned weapon: short horizontal scan stub to the right of its box.
                ln.x0 = wT.x1;
                ln.y0 = wcy;
                ln.x1 = Clamp01(wT.x1 + g_cfg.sightLen * 0.4f);
                ln.y1 = wcy;
            }
            ++g_nSight;
        }
    }
    return tr;
}

const char* const* det::Labels() noexcept
{
    return g_labels;
}

const hud::Line* det::SightLines(int* n) noexcept
{
    if (n != nullptr)
        *n = g_nSight;
    return g_sight;
}

const char* det::Label(int cls) noexcept
{
    if (cls >= kSess1Base)
    {
        const int i = cls - kSess1Base;
        if (i < kSess1NamesN)
            return kSess1Names[i];
        static thread_local char buf2[16];
        _snprintf_s(buf2, _TRUNCATE, "DET %02d", i);
        return buf2;
    }
    if (cls >= 0 && cls < 80)
        return kCoco[cls];
    static thread_local char buf[16];
    _snprintf_s(buf, _TRUNCATE, "OBJ %02d", cls);
    return buf;
}

bool det::Active() noexcept
{
    return g_ready && !g_failed;
}

void det::ReleaseBackbuffer() noexcept
{
    ReleaseBB();
}

void det::Reset() noexcept
{
    ReleaseBB();
    Teardown();
    Rel(g_psIn); Rel(g_cb); Rel(g_linear); Rel(g_query);
    g_dev = nullptr;
    g_failed = false;
    g_frame = 0;
}

#endif // RETRO_HAVE_ORT
