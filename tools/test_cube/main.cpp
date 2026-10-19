// Test harness: rotating cube, toggleable heavy/light pixel shader (F1). Used to develop
// and verify the proxy without a real game.
#include <Windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <DirectXMath.h>
#include <wrl/client.h>
#include <cstdio>
#include <cstring>

using Microsoft::WRL::ComPtr;
using namespace DirectX;

namespace
{

constexpr int kWidth = 960;
constexpr int kHeight = 540;

struct Vertex
{
    XMFLOAT3 pos;
    XMFLOAT2 uv;
};

struct FrameCB
{
    XMFLOAT4X4 wvp;  // XMFLOAT4X4: mapped CB memory isn't guaranteed 16B-aligned on x86
    float boost;     // read by the heavy PS so the loop can't be constant-folded
    float pad[3];
};

ComPtr<ID3D11Device>           g_device;
ComPtr<ID3D11DeviceContext>    g_context;
ComPtr<IDXGISwapChain>         g_swapChain;
ComPtr<ID3D11RenderTargetView> g_rtv;
ComPtr<ID3D11Buffer>           g_vb, g_ib, g_cb;
ComPtr<ID3D11VertexShader>     g_vs;
ComPtr<ID3D11PixelShader>      g_psLight, g_psHeavy;
ComPtr<ID3D11InputLayout>      g_layout;
ComPtr<ID3D11RasterizerState>  g_rs;
bool g_heavyPs = true;

const char kVsSrc[] = R"(
cbuffer FrameCB : register(b0) { float4x4 g_wvp; float g_boost; };
struct VSIn  { float3 pos : POSITION; float2 uv : TEXCOORD0; };
struct VSOut { float4 pos : SV_POSITION; float2 uv : TEXCOORD0; };
VSOut main(VSIn i)
{
    VSOut o;
    o.pos = mul(float4(i.pos, 1.0f), g_wvp);
    o.uv  = i.uv;
    return o;
}
)";

// Light PS: trivial color. Baseline for Null-PS / Flat-PS comparisons.
const char kPsLightSrc[] = R"(
float4 main(float4 pos : SV_POSITION, float2 uv : TEXCOORD0) : SV_TARGET
{
    return float4(uv.x, uv.y, 1.0f, 1.0f);
}
)";

// Heavy PS: ~256 light-loop ALU iterations. Deliberately stresses the pixel stage so
// stripping it produces a dramatic FPS delta even on fast GPUs.
const char kPsHeavySrc[] = R"(
cbuffer FrameCB : register(b0) { float4x4 g_wvp; float g_boost; };
float4 main(float4 pos : SV_POSITION, float2 uv : TEXCOORD0) : SV_TARGET
{
    float3 n = normalize(float3(uv.x - 0.5f, uv.y - 0.5f, 0.8f));
    float3 acc = 0.0f;
    for (int i = 0; i < 256; ++i)
    {
        float a = (float)i * 0.024543f * (g_boost + 1.0f);
        float3 l = float3(sin(a), cos(a * 1.37f), sin(a * 0.61f));
        float d  = saturate(dot(n, normalize(l)));
        acc += l * d * d;
    }
    return float4(frac(acc), 1.0f);
}
)";

[[noreturn]] void Fatal(const char* what, HRESULT hr)
{
    char msg[256];
    sprintf_s(msg, "%s failed: hr=0x%08lX", what, static_cast<unsigned long>(hr));
    MessageBoxA(nullptr, msg, "test_cube", MB_ICONERROR | MB_OK);
    ExitProcess(1);
}

ComPtr<ID3DBlob> CompileShader(const char* src, const char* target)
{
    ComPtr<ID3DBlob> blob, errors;
    const HRESULT hr = D3DCompile(src, strlen(src), nullptr, nullptr,
                                  nullptr, "main", target,
                                  D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &blob, &errors);
    if (FAILED(hr))
    {
        const char* detail = errors ? static_cast<const char*>(errors->GetBufferPointer()) : "";
        OutputDebugStringA(detail);
        Fatal("D3DCompile", hr);
    }
    return blob;
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM w, LPARAM l)
{
    switch (msg)
    {
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    case WM_KEYDOWN:
        if (w == VK_F1)
        {
            g_heavyPs = !g_heavyPs;
            InvalidateRect(hwnd, nullptr, FALSE);
        }
        return 0;
    }
    return DefWindowProcW(hwnd, msg, w, l);
}

HWND CreateMainWindow(HINSTANCE inst)
{
    const wchar_t* cls = L"CRenderTestCube";
    WNDCLASSEXW wc = {};
    wc.cbSize        = sizeof(wc);
    wc.lpfnWndProc   = &WndProc;
    wc.hInstance     = inst;
    wc.lpszClassName = cls;
    wc.hCursor       = LoadCursorW(nullptr, IDC_ARROW);
    RegisterClassExW(&wc);

    RECT rc = { 0, 0, kWidth, kHeight };
    AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);
    return CreateWindowExW(0, cls, L"CRender test cube [F1 = toggle PS]",
                           WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
                           rc.right - rc.left, rc.bottom - rc.top,
                           nullptr, nullptr, inst, nullptr);
}

void InitD3D(HWND hwnd)
{
    DXGI_SWAP_CHAIN_DESC sd = {};
    sd.BufferDesc.Width    = kWidth;
    sd.BufferDesc.Height   = kHeight;
    sd.BufferDesc.Format   = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.SampleDesc.Count    = 1;
    sd.BufferUsage         = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.BufferCount         = 2;
    sd.OutputWindow        = hwnd;
    sd.Windowed            = TRUE;
    sd.SwapEffect          = DXGI_SWAP_EFFECT_DISCARD; // pre-Flip model: simple + reliable on all D3D11 stacks

    HRESULT hr = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
#ifdef _DEBUG
                                             D3D11_CREATE_DEVICE_DEBUG,
#else
                                             0,
#endif
                                             nullptr, 0, D3D11_SDK_VERSION,
                                             &sd, &g_swapChain, &g_device, nullptr, &g_context);
    if (FAILED(hr))
        Fatal("D3D11CreateDeviceAndSwapChain", hr);

    ComPtr<ID3D11Texture2D> back;
    hr = g_swapChain->GetBuffer(0, IID_PPV_ARGS(&back));
    if (FAILED(hr))
        Fatal("GetBuffer", hr);
    hr = g_device->CreateRenderTargetView(back.Get(), nullptr, &g_rtv);
    if (FAILED(hr))
        Fatal("CreateRenderTargetView", hr);

    // Cube: 8 verts, 36 indices (12 triangles). Matches Phase 6 acceptance exactly.
    const Vertex verts[8] = {
        { { -1.f, -1.f, -1.f }, { 0.f, 0.f } }, { { 1.f, -1.f, -1.f }, { 1.f, 0.f } },
        { {  1.f,  1.f, -1.f }, { 1.f, 1.f } }, { { -1.f, 1.f, -1.f }, { 0.f, 1.f } },
        { { -1.f, -1.f,  1.f }, { 0.f, 0.f } }, { { 1.f, -1.f,  1.f }, { 1.f, 0.f } },
        { {  1.f,  1.f,  1.f }, { 1.f, 1.f } }, { { -1.f, 1.f,  1.f }, { 0.f, 1.f } },
    };
    const UINT16 indices[36] = {
        0,1,2, 0,2,3,  4,6,5, 4,7,6,  4,5,1, 4,1,0,
        3,2,6, 3,6,7,  1,5,6, 1,6,2,  4,0,3, 4,3,7,
    };

    D3D11_BUFFER_DESC bd = {};
    bd.ByteWidth = sizeof(verts);
    bd.Usage     = D3D11_USAGE_IMMUTABLE;
    bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    D3D11_SUBRESOURCE_DATA init = {};
    init.pSysMem = verts;
    hr = g_device->CreateBuffer(&bd, &init, &g_vb);
    if (FAILED(hr))
        Fatal("CreateBuffer VB", hr);

    bd.ByteWidth = sizeof(indices);
    bd.BindFlags = D3D11_BIND_INDEX_BUFFER;
    init.pSysMem = indices;
    hr = g_device->CreateBuffer(&bd, &init, &g_ib);
    if (FAILED(hr))
        Fatal("CreateBuffer IB", hr);

    bd.ByteWidth      = sizeof(FrameCB);
    bd.Usage          = D3D11_USAGE_DYNAMIC;
    bd.BindFlags      = D3D11_BIND_CONSTANT_BUFFER;
    bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    hr = g_device->CreateBuffer(&bd, nullptr, &g_cb);
    if (FAILED(hr))
        Fatal("CreateBuffer CB", hr);

    ComPtr<ID3DBlob> vsBlob  = CompileShader(kVsSrc,      "vs_4_0");
    ComPtr<ID3DBlob> psLight = CompileShader(kPsLightSrc, "ps_4_0");
    ComPtr<ID3DBlob> psHeavy = CompileShader(kPsHeavySrc, "ps_4_0");

    hr = g_device->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(),
                                      nullptr, &g_vs);
    if (FAILED(hr))
        Fatal("CreateVertexShader", hr);
    hr = g_device->CreatePixelShader(psLight->GetBufferPointer(), psLight->GetBufferSize(),
                                     nullptr, &g_psLight);
    if (FAILED(hr))
        Fatal("CreatePixelShader light", hr);
    hr = g_device->CreatePixelShader(psHeavy->GetBufferPointer(), psHeavy->GetBufferSize(),
                                     nullptr, &g_psHeavy);
    if (FAILED(hr))
        Fatal("CreatePixelShader heavy", hr);

    const D3D11_INPUT_ELEMENT_DESC elems[2] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,                            D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,    0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
    };
    hr = g_device->CreateInputLayout(elems, 2, vsBlob->GetBufferPointer(),
                                     vsBlob->GetBufferSize(), &g_layout);
    if (FAILED(hr))
        Fatal("CreateInputLayout", hr);

    D3D11_RASTERIZER_DESC rd = {};
    rd.FillMode              = D3D11_FILL_SOLID;
    rd.CullMode              = D3D11_CULL_BACK;
    rd.DepthClipEnable       = TRUE;
    rd.ScissorEnable         = FALSE;
    hr = g_device->CreateRasterizerState(&rd, &g_rs);
    if (FAILED(hr))
        Fatal("CreateRasterizerState", hr);
}

void RenderFrame(ULONGLONG ticks)
{
    const float clear[4] = { 0.02f, 0.02f, 0.05f, 1.0f };
    g_context->ClearRenderTargetView(g_rtv.Get(), clear);
    g_context->OMSetRenderTargets(1, g_rtv.GetAddressOf(), nullptr);

    const float t = static_cast<float>(ticks) * 0.001f;
    const XMMATRIX world = XMMatrixRotationY(t) * XMMatrixRotationX(t * 0.7f);
    const XMMATRIX view  = XMMatrixLookAtLH(XMVectorSet(0, 0, -4, 1), XMVectorZero(),
                                            XMVectorSet(0, 1, 0, 0));
    const XMMATRIX proj  = XMMatrixPerspectiveFovLH(XM_PIDIV4,
                                                  static_cast<float>(kWidth) / kHeight,
                                                  0.1f, 100.f);

    D3D11_MAPPED_SUBRESOURCE mapped = {};
    if (SUCCEEDED(g_context->Map(g_cb.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped)))
    {
        FrameCB* cb = static_cast<FrameCB*>(mapped.pData);
        XMStoreFloat4x4(&cb->wvp, XMMatrixTranspose(world * view * proj));
        cb->boost = 1.0f;
        cb->pad[0] = cb->pad[1] = cb->pad[2] = 0.f;
        g_context->Unmap(g_cb.Get(), 0);
    }

    const UINT stride = sizeof(Vertex);
    const UINT offset = 0;
    g_context->IASetInputLayout(g_layout.Get());
    g_context->IASetVertexBuffers(0, 1, g_vb.GetAddressOf(), &stride, &offset);
    g_context->IASetIndexBuffer(g_ib.Get(), DXGI_FORMAT_R16_UINT, 0);
    g_context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    g_context->VSSetShader(g_vs.Get(), nullptr, 0);
    g_context->VSSetConstantBuffers(0, 1, g_cb.GetAddressOf());
    g_context->PSSetShader(g_heavyPs ? g_psHeavy.Get() : g_psLight.Get(), nullptr, 0);
    g_context->RSSetState(g_rs.Get());

    D3D11_VIEWPORT vp = {};
    vp.Width    = static_cast<float>(kWidth);
    vp.Height   = static_cast<float>(kHeight);
    vp.MaxDepth = 1.0f;
    g_context->RSSetViewports(1, &vp);

    g_context->DrawIndexed(36, 0, 0);

    g_swapChain->Present(0, 0); // uncapped: exposes the heavy-PS cost
}

} // namespace

int WINAPI wWinMain(HINSTANCE inst, HINSTANCE, PWSTR, int showCmd)
{
    HWND hwnd = CreateMainWindow(inst);
    if (hwnd == nullptr)
        return 1;
    ShowWindow(hwnd, showCmd);
    InitD3D(hwnd);

    ULONGLONG lastFpsTick = GetTickCount64();
    unsigned frames = 0;

    MSG msg = {};
    while (msg.message != WM_QUIT)
    {
        if (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE))
        {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
            continue;
        }

        RenderFrame(GetTickCount64());
        ++frames;

        const ULONGLONG now = GetTickCount64();
        if (now - lastFpsTick >= 500)
        {
            const double fps = frames * 1000.0 / static_cast<double>(now - lastFpsTick);
            wchar_t title[128];
            swprintf_s(title, L"CRender test cube — %s PS — %.0f fps [F1 toggles]",
                       g_heavyPs ? L"HEAVY" : L"light", fps);
            SetWindowTextW(hwnd, title);
            frames = 0;
            lastFpsTick = now;
        }
    }
    return static_cast<int>(msg.wParam);
}
