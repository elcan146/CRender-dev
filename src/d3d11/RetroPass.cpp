// RetroPass — fullscreen post-process applied inside the Present detour.
//
// RetroFX: pixelate + 3-bit palette quantize + scanlines (no assets).
// Ascii:   screen -> cell grid -> cell-average luminance -> procedurally
//          generated 8x8 glyph atlas (density ramp ' ', '.', ':', '-', '=',
//          '+', '*', '#', '%', '@'), green-phosphor tint.
//
// All shaders compile at runtime via GetProcAddress("D3DCompile") — keeps the
// proxy's import table KERNEL32-only. Everything is created lazily on first use.
#include "RetroPass.h"
#include "StateBlock.h"
#include "../core/Log.h"

#include <Windows.h>
#include <d3dcompiler.h> // types only — resolved via GetProcAddress
#include <cstring>

namespace
{

using FnD3DCompile = HRESULT(WINAPI*)(LPCVOID, SIZE_T, LPCSTR, const D3D_SHADER_MACRO*,
                                     ID3DInclude*, LPCSTR, LPCSTR, UINT, UINT,
                                     ID3DBlob**, ID3DBlob**);

FnD3DCompile g_compile = nullptr;
bool         g_compileTried = false;

bool ResolveCompiler() noexcept
{
    if (g_compile != nullptr)
        return true;
    if (g_compileTried)
        return false;
    g_compileTried = true;
    HMODULE d3dc = LoadLibraryW(L"d3dcompiler_47.dll");
    if (d3dc == nullptr)
        d3dc = LoadLibraryW(L"d3dcompiler_43.dll");
    if (d3dc == nullptr)
    {
        Log::Warn("retro", "no d3dcompiler_4x.dll — retro pass unavailable");
        return false;
    }
    g_compile = reinterpret_cast<FnD3DCompile>(GetProcAddress(d3dc, "D3DCompile"));
    return g_compile != nullptr;
}

ID3DBlob* Compile(const char* src, const char* entry, const char* target) noexcept
{
    if (!ResolveCompiler())
        return nullptr;
    ID3DBlob* blob = nullptr;
    ID3DBlob* errors = nullptr;
    const HRESULT hr = g_compile(src, strlen(src), nullptr, nullptr, nullptr,
                                 entry, target, 0, 0, &blob, &errors);
    if (FAILED(hr) || blob == nullptr)
    {
        if (errors != nullptr)
        {
            Log::Warn("retro", "%s compile: %s", entry,
                      static_cast<const char*>(errors->GetBufferPointer()));
            errors->Release();
        }
        return nullptr;
    }
    if (errors != nullptr)
        errors->Release();
    return blob;
}

// GpuBurn throttle shader — pure ALU work per pixel, no textures. Each fullscreen
// invocation costs ~kIter MAD/sin ops at backbuffer res, so N passes ≈ a fixed
// slice of per-frame GPU time whatever the hardware.
const char kBurnPsSrc[] =
    "float4 main(float4 p : SV_Position) : SV_Target {"
    "  float x = p.x * 1.618 + p.y;"
    "  [loop] for (int i = 0; i < 400; ++i)"
    "    x = sin(x * 1.01) * 1.0003 + frac(x) * 0.5 + 0.001;"
    "  return float4(x, x, x, 1);"
    "}";

// Integer-scale blit: point-sample the shrunken scene into the native backbuffer.
// cbuffer = { offX, offY, drawW, drawH } — the letterboxed draw region in dst px.
const char kBlitPsSrc[] =
    "Texture2D src : register(t0);"
    "SamplerState smp : register(s0);"
    "cbuffer B : register(b0) { float2 drawOff; float2 drawDim; };"
    "float4 main(float4 p : SV_Position) : SV_Target {"
    "  float2 uv = (p.xy - drawOff) / drawDim;"  // 0..1 across the drawn region
    "  return src.Sample(smp, uv);"
    "}";

// ---- resources ----------------------------------------------------------------

ID3D11Device*            g_dev = nullptr;   // non-owning
ID3D11VertexShader*      g_vs = nullptr;
ID3D11PixelShader*       g_psRetro = nullptr;
ID3D11PixelShader*       g_psAscii = nullptr;
ID3D11PixelShader*       g_psTermBase = nullptr;
ID3D11PixelShader*       g_psLens = nullptr;
ID3D11PixelShader*       g_psBurn = nullptr;
ID3D11Texture2D*         g_burnTex = nullptr;
ID3D11RenderTargetView*  g_burnRTV = nullptr;
int                      g_burnPasses = 0; // GpuBurn config — heavy passes per frame
UINT                     g_burnW = 0, g_burnH = 0;
ID3D11SamplerState*      g_samp = nullptr;
ID3D11SamplerState*      g_sampLin = nullptr; // bilinear — Ascii cell-average taps
ID3D11RasterizerState*   g_rs = nullptr;
ID3D11BlendState*        g_blend = nullptr;
ID3D11DepthStencilState* g_ds = nullptr;
ID3D11Buffer*            g_params = nullptr;
ID3D11Texture2D*         g_atlas = nullptr;
ID3D11ShaderResourceView* g_atlasSRV = nullptr;

// backbuffer-sized copy chain
ID3D11Texture2D*          g_copy = nullptr;
ID3D11ShaderResourceView* g_copySRV = nullptr;
ID3D11RenderTargetView*   g_bbRTV = nullptr;
ID3D11Texture2D*          g_bbPtr = nullptr; // backbuffer object we cached
UINT                      g_bbW = 0, g_bbH = 0;
UINT                      g_bbSamples = 1;   // backbuffer MSAA count (ResolveSubresource path)
DXGI_FORMAT               g_bbFmt = DXGI_FORMAT_R8G8B8A8_UNORM;
bool                      g_failed = false;
IDXGISwapChain*           g_lastSC = nullptr;

// Upscale-present chain (owned by HookManager): dst backbuffer RTV + blit PS.
ID3D11PixelShader*        g_psBlit   = nullptr;
ID3D11RenderTargetView*   g_upRTV    = nullptr;
IDXGISwapChain*           g_upSC     = nullptr;
UINT                      g_upW = 0, g_upH = 0;
ID3D11Buffer*             g_upCB     = nullptr;

// Post-target chain (ApplyTo/PostRTV): whatever texture is finally presented — the
// game bb, or our native-size present chain's bb in the RenderScale path. Kept apart
// from the main chain so alternating sources never thrash either cache.
ID3D11Texture2D*          g_postCopy = nullptr;
ID3D11ShaderResourceView* g_postSRV  = nullptr;
ID3D11RenderTargetView*   g_postRTV  = nullptr;
ID3D11Texture2D*          g_postBB   = nullptr; // owning ref, pointer identity
UINT                      g_postW = 0, g_postH = 0, g_postSamples = 1;
DXGI_FORMAT               g_postFmt = DXGI_FORMAT_UNKNOWN;

using retro::StateBlock;

// retro.json tunables — PixelPx (RetroFX block), CellPx (Ascii cell),
// Crt (barrel curvature strength), Vignette (corner darkening).
float g_pixelPx  = 4.0f;
float g_cellPx   = 8.0f;
float g_crt      = 0.12f;
float g_vignette = 0.35f;
float g_palette  = 0.0f; // 0=green 1=amber 2=mono 3=cyan 4=color
float g_scalePx  = 1.0f; // RenderScale divisor — floors pixelPx/cellPx
// Terminator: TermEdgeGain / TermTint (base layer), TerminatorLens block (lens).
float g_termGain    = 1.0f;
float g_termTint[3] = { 1.0f, 0.0f, 0.0f };
float g_lensCrt     = 0.10f;
float g_lensVig     = 0.45f;
float g_lensScan    = 0.35f;

// ---- shaders -------------------------------------------------------------------

// Fullscreen triangle from SV_VertexID — no input layout needed.
const char kVsSrc[] =
    "float4 main(uint id : SV_VertexID) : SV_Position {"
    "  float2 uv = float2((id << 1) & 2, id & 2);"
    "  return float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);"
    "}";

// Params cbuffer (48 B): scDim = drawn region size in output px, drawOff = its
// top-left (letterbox) — effects run at OUTPUT resolution, sampling the scene.
const char kPsRetroSrc[] =
    "Texture2D scene : register(t0);"
    "SamplerState smp : register(s0);"
    "cbuffer P : register(b0) { float2 scDim; float pixelPx; float cellPx;"
    "                          float crt; float vignette; float palette; float pad0;"
    "                          float2 drawOff; float2 pad1; }"
    "static const float bayer[16] = { 0,8,2,10, 12,4,14,6, 3,11,1,9, 15,7,13,5 };"
    "float4 main(float4 pos : SV_Position) : SV_Target {"
    "  float2 uv = (pos.xy - drawOff) / scDim;"
    "  float2 ndc = uv * 2.0 - 1.0;"                                  // -1..1
    "  float2 curved = ndc * (1.0 + crt * dot(ndc, ndc));"            // barrel
    "  uv = curved * 0.5 + 0.5;"
    "  if (any(uv < 0.0) || any(uv > 1.0)) return float4(0.0,0.0,0.0,1.0);"
    "  float2 bpos = uv * scDim / pixelPx;"                           // block space
    "  float2 cell = floor(bpos);"
    "  float3 c = scene.SampleLevel(smp, (cell + 0.5) * pixelPx / scDim, 0).rgb;"
    "  int2 d = (int2)cell & 3;"                                     // ordered dither
    "  float t = (bayer[d.y * 4 + d.x] + 0.5) / 16.0;"
    "  c = saturate(floor(c * 7.0 + t) / 7.0);"                      // 3-bit/channel
    "  float fy = frac(bpos.y);"                                       // per-block scanline
    "  float sl = pixelPx >= 3.0 ? lerp(0.72, 1.08, sin(fy * 3.14159))"
    "                            : 0.85 + 0.15 * step(0.5, frac(pos.y * 0.5));"
    "  float vig = 1.0 - vignette * smoothstep(0.4, 1.4, length(ndc));"
    "  return float4(saturate(c * sl * vig), 1.0);"
    "}";

const char kPsAsciiSrc[] =
    "Texture2D scene : register(t0);"
    "Texture2D atlas : register(t1);"
    "SamplerState smp : register(s0);"
    "SamplerState lin : register(s1);"
    "cbuffer P : register(b0) { float2 scDim; float pixelPx; float cellPx;"
    "                          float crt; float vignette; float palette; float pad0;"
    "                          float2 drawOff; float2 pad1; }"
    "static const float NGLYPH = 10.0;"
    "float4 main(float4 pos : SV_Position) : SV_Target {"
    "  float2 p = pos.xy - drawOff;"
    "  float2 cellIdx = floor(p / cellPx);"
    "  float2 cuv = cellPx / scDim;"
    "  float2 base = cellIdx * cuv;"
    "  float3 col ="
    "    scene.SampleLevel(lin, base + cuv * float2(0.25,0.25), 0).rgb +"
    "    scene.SampleLevel(lin, base + cuv * float2(0.75,0.25), 0).rgb +"
    "    scene.SampleLevel(lin, base + cuv * float2(0.25,0.75), 0).rgb +"
    "    scene.SampleLevel(lin, base + cuv * float2(0.75,0.75), 0).rgb;"
    "  col *= 0.25;"
    "  float lum = dot(col, float3(0.299, 0.587, 0.114));"
    "  lum = saturate(pow(lum, 0.7) * 1.1);"                          // lift darks so the ramp is used
    "  int g = (int)min(lum * (NGLYPH - 1.0) + 0.5, NGLYPH - 1.0);"
    "  int2 tx = (int2)min(floor(frac(p / cellPx) * 8.0), 7.0);"     // exact glyph texel
    "  float mask = atlas.Load(int3(g * 8 + tx.x, tx.y, 0)).r;"
    "  float3 phosphor;"
    "  if      (palette > 3.5) phosphor = col / max(lum, 0.05);"       // scene color — hue+sat
    "  else if (palette > 2.5) phosphor = float3(0.35, 0.85, 1.00);"  // cyan
    "  else if (palette > 1.5) phosphor = float3(0.85, 0.85, 0.85);"  // mono
    "  else if (palette > 0.5) phosphor = float3(1.00, 0.63, 0.10);"  // amber
    "  else                    phosphor = float3(0.20, 1.00, 0.35);"  // green
    "  float3 fg = phosphor * (0.35 + 0.65 * lum);"
    "  float3 bg = phosphor * lum * 0.07;"                            // faint fill keeps silhouettes
    "  return float4(saturate(lerp(bg, fg, mask)), 1.0);"
    "}";

// Terminator base layer (HLSL fallback for terminator_vision.onnx — identical math:
// luma -> Sobel (zero 1px border) -> soft-threshold edges -> crimson routing).
// cbuffer: scDim, edgeGain (TermEdgeGain), tint (TermTint, applied to the void).
const char kPsTermBaseSrc[] =
    "Texture2D scene : register(t0);"
    "cbuffer P : register(b0) { float2 scDim; float edgeGain; float pad0;"
    "                          float3 tint; float pad1; float4 pad2; }"
    "float L(int2 p) { return dot(scene.Load(int3(p, 0)).rgb, float3(0.299, 0.587, 0.114)); }"
    "float4 main(float4 pos : SV_Position) : SV_Target {"
    "  int2 p = int2(pos.xy);"
    "  int2 mx = int2(scDim) - 1;"
    "  float y = L(min(p, mx));"
    "  float gx = 0.0, gy = 0.0;"
    "  if (p.x > 0 && p.y > 0 && p.x < mx.x && p.y < mx.y) {"
    "    float a = L(p + int2(-1,-1)), b = L(p + int2(0,-1)), c = L(p + int2(1,-1));"
    "    float d = L(p + int2(-1, 0)),                         f = L(p + int2(1, 0));"
    "    float g = L(p + int2(-1, 1)), h = L(p + int2(0, 1)), i = L(p + int2(1, 1));"
    "    gx = (c + 2.0 * f + i) - (a + 2.0 * d + g);"
    "    gy = (g + 2.0 * h + i) - (a + 2.0 * b + c);"
    "  }"
    "  float e  = saturate(sqrt(gx * gx + gy * gy + 1e-6) * 2.0 - 0.05);"   // EDGE_GAIN / EDGE_THRESHOLD
    "  float e2 = saturate(e * edgeGain);"
    "  float bg = 0.525 * y;"                                                // 1.5 * BG_LUMA
    "  return float4(saturate(bg * tint + float3(e2, 0.7 * e2, 0.7 * e2)), 1.0);"
    "}";

// Terminator lens: CRT barrel + scanlines + red-tuned vignette, no pixelate/palette.
const char kPsLensSrc[] =
    "Texture2D scene : register(t0);"
    "SamplerState smp : register(s0);"
    "SamplerState lin : register(s1);"
    "cbuffer P : register(b0) { float2 scDim; float2 pad0;"
    "                          float crt; float vignette; float scan; float pad1;"
    "                          float2 drawOff; float2 pad2; }"
    "float4 main(float4 pos : SV_Position) : SV_Target {"
    "  float2 uv = (pos.xy - drawOff) / scDim;"
    "  float2 ndc = uv * 2.0 - 1.0;"
    "  uv = ndc * (1.0 + crt * dot(ndc, ndc)) * 0.5 + 0.5;"
    "  if (any(uv < 0.0) || any(uv > 1.0)) return float4(0.0, 0.0, 0.0, 1.0);"
    "  float3 c = scene.SampleLevel(lin, uv, 0).rgb;"
    "  c *= 1.0 - scan * (0.5 + 0.5 * cos(pos.y * 2.0943951));"          // 3 px period
    "  float v = vignette * smoothstep(0.35, 1.45, length(ndc));"
    "  c *= float3(1.0 - 0.55 * v, 1.0 - v, 1.0 - v);"                    // corners sink to deep red
    "  return float4(saturate(c), 1.0);"
    "}";

// 8x8 bitmap atlas. [0..9] = Ascii density ramp (index order matches the luminance
// pick in the shader); [10..] = 5x7 HUD text glyphs (see kTextChars / GlyphIndex).
const unsigned char kGlyphs[][8] = {
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00}, // ' '
    {0x00,0x00,0x00,0x00,0x00,0x18,0x18,0x00}, // '.'
    {0x00,0x00,0x18,0x18,0x00,0x18,0x18,0x00}, // ':'
    {0x00,0x00,0x00,0x3C,0x3C,0x00,0x00,0x00}, // '-'
    {0x00,0x00,0x3C,0x00,0x00,0x3C,0x00,0x00}, // '='
    {0x00,0x18,0x18,0x7E,0x18,0x18,0x00,0x00}, // '+'
    {0x00,0x66,0x3C,0x7E,0x3C,0x66,0x00,0x00}, // '*'
    {0x24,0x7E,0x24,0x24,0x7E,0x24,0x00,0x00}, // '#'
    {0x62,0x64,0x08,0x10,0x26,0x46,0x00,0x00}, // '%'
    {0x3C,0x7E,0x7E,0x7E,0x7E,0x3C,0x00,0x00}, // '@' (densest)
    {0x38,0x44,0x4C,0x54,0x64,0x44,0x38,0x00}, // 0
    {0x10,0x30,0x10,0x10,0x10,0x10,0x38,0x00}, // 1
    {0x38,0x44,0x04,0x08,0x10,0x20,0x7C,0x00}, // 2
    {0x7C,0x08,0x10,0x08,0x04,0x44,0x38,0x00}, // 3
    {0x08,0x18,0x28,0x48,0x7C,0x08,0x08,0x00}, // 4
    {0x7C,0x40,0x78,0x04,0x04,0x44,0x38,0x00}, // 5
    {0x18,0x20,0x40,0x78,0x44,0x44,0x38,0x00}, // 6
    {0x7C,0x04,0x08,0x10,0x20,0x20,0x20,0x00}, // 7
    {0x38,0x44,0x44,0x38,0x44,0x44,0x38,0x00}, // 8
    {0x38,0x44,0x44,0x3C,0x04,0x08,0x30,0x00}, // 9
    {0x38,0x44,0x44,0x44,0x7C,0x44,0x44,0x00}, // A
    {0x78,0x44,0x44,0x78,0x44,0x44,0x78,0x00}, // B
    {0x38,0x44,0x40,0x40,0x40,0x44,0x38,0x00}, // C
    {0x70,0x48,0x44,0x44,0x44,0x48,0x70,0x00}, // D
    {0x7C,0x40,0x40,0x78,0x40,0x40,0x7C,0x00}, // E
    {0x7C,0x40,0x40,0x78,0x40,0x40,0x40,0x00}, // F
    {0x38,0x44,0x40,0x5C,0x44,0x44,0x3C,0x00}, // G
    {0x44,0x44,0x44,0x7C,0x44,0x44,0x44,0x00}, // H
    {0x38,0x10,0x10,0x10,0x10,0x10,0x38,0x00}, // I
    {0x1C,0x08,0x08,0x08,0x08,0x48,0x30,0x00}, // J
    {0x44,0x48,0x50,0x60,0x50,0x48,0x44,0x00}, // K
    {0x40,0x40,0x40,0x40,0x40,0x40,0x7C,0x00}, // L
    {0x44,0x6C,0x54,0x54,0x44,0x44,0x44,0x00}, // M
    {0x44,0x44,0x64,0x54,0x4C,0x44,0x44,0x00}, // N
    {0x38,0x44,0x44,0x44,0x44,0x44,0x38,0x00}, // O
    {0x78,0x44,0x44,0x78,0x40,0x40,0x40,0x00}, // P
    {0x38,0x44,0x44,0x44,0x54,0x48,0x34,0x00}, // Q
    {0x78,0x44,0x44,0x78,0x50,0x48,0x44,0x00}, // R
    {0x3C,0x40,0x40,0x38,0x04,0x04,0x78,0x00}, // S
    {0x7C,0x10,0x10,0x10,0x10,0x10,0x10,0x00}, // T
    {0x44,0x44,0x44,0x44,0x44,0x44,0x38,0x00}, // U
    {0x44,0x44,0x44,0x44,0x44,0x28,0x10,0x00}, // V
    {0x44,0x44,0x44,0x54,0x54,0x54,0x28,0x00}, // W
    {0x44,0x44,0x28,0x10,0x28,0x44,0x44,0x00}, // X
    {0x44,0x44,0x44,0x28,0x10,0x10,0x10,0x00}, // Y
    {0x7C,0x04,0x08,0x10,0x20,0x40,0x7C,0x00}, // Z
    {0x00,0x04,0x08,0x10,0x20,0x40,0x00,0x00}, // /
    {0x08,0x10,0x20,0x40,0x20,0x10,0x08,0x00}, // <
    {0x20,0x10,0x08,0x04,0x08,0x10,0x20,0x00}, // >
    {0x38,0x20,0x20,0x20,0x20,0x20,0x38,0x00}, // [
    {0x38,0x08,0x08,0x08,0x08,0x08,0x38,0x00}, // ]
    {0x00,0x00,0x00,0x00,0x00,0x00,0x7C,0x00}, // _
    {0x10,0x10,0x10,0x10,0x10,0x00,0x10,0x00}, // !
    {0x38,0x44,0x04,0x08,0x10,0x00,0x10,0x00}, // ?
};
constexpr UINT kGlyphCount = static_cast<UINT>(sizeof(kGlyphs) / sizeof(kGlyphs[0]));
// Text glyphs start right after the 10-entry density ramp, in this order.
constexpr char kTextChars[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ/<>[]_!?";
constexpr UINT kTextBase = 10;
static_assert(kTextBase + sizeof(kTextChars) - 1 == sizeof(kGlyphs) / sizeof(kGlyphs[0]),
              "kTextChars must map 1:1 onto the text glyphs");
constexpr UINT kGlyphCell  = 8;

bool BuildAtlas(ID3D11Device* dev) noexcept
{
    unsigned char px[kGlyphCell][kGlyphCell * kGlyphCount] = {};
    for (UINT g = 0; g < kGlyphCount; ++g)
        for (UINT y = 0; y < kGlyphCell; ++y)
            for (UINT x = 0; x < kGlyphCell; ++x)
                px[y][g * kGlyphCell + x] = (kGlyphs[g][y] & (0x80 >> x)) ? 255 : 0;

    D3D11_TEXTURE2D_DESC td = {};
    td.Width            = kGlyphCell * kGlyphCount;
    td.Height           = kGlyphCell;
    td.MipLevels        = 1;
    td.ArraySize        = 1;
    td.Format           = DXGI_FORMAT_R8_UNORM;
    td.SampleDesc.Count = 1;
    td.Usage            = D3D11_USAGE_IMMUTABLE;
    td.BindFlags        = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA sd = {};
    sd.pSysMem     = px;
    sd.SysMemPitch = kGlyphCell * kGlyphCount;
    if (FAILED(dev->CreateTexture2D(&td, &sd, &g_atlas)) || g_atlas == nullptr)
        return false;
    return SUCCEEDED(dev->CreateShaderResourceView(g_atlas, nullptr, &g_atlasSRV));
}

// Create/refresh the backbuffer-sized copy chain when dimensions or buffer change.
bool EnsureCopyChain(IDXGISwapChain* sc) noexcept
{
    ID3D11Texture2D* bb = nullptr;
    if (FAILED(sc->GetBuffer(0, __uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&bb))) || bb == nullptr)
        return false;

    D3D11_TEXTURE2D_DESC bd = {};
    bb->GetDesc(&bd);
    if (bb == g_bbPtr && g_copy != nullptr && g_copySRV != nullptr && g_bbRTV != nullptr)
    {
        bb->Release();
        return true;
    }

    if (sc != g_lastSC)
    {
        g_lastSC = sc;
        Log::Info("retro", "presenting on swapchain %p", sc);
    }
    Log::Info("retro", "copy chain rebuild: bb=%p %ux%u fmt=%u samples=%u",
              bb, bd.Width, bd.Height, bd.Format, bd.SampleDesc.Count);

    if (g_copy != nullptr)    g_copy->Release();
    if (g_copySRV != nullptr) g_copySRV->Release();
    if (g_bbRTV != nullptr)   g_bbRTV->Release();
    g_copy = nullptr; g_copySRV = nullptr; g_bbRTV = nullptr; g_bbPtr = nullptr;

    // our sampling texture is always non-MSAA — MSAA sources go through
    // ResolveSubresource in Apply instead of CopyResource.
    D3D11_TEXTURE2D_DESC cd = bd;
    cd.Usage            = D3D11_USAGE_DEFAULT;
    cd.BindFlags        = D3D11_BIND_SHADER_RESOURCE;
    cd.CPUAccessFlags   = 0;
    cd.MiscFlags        = 0;
    cd.SampleDesc.Count = 1;
    cd.SampleDesc.Quality = 0;
    if (FAILED(g_dev->CreateTexture2D(&cd, nullptr, &g_copy)) || g_copy == nullptr)
    {
        Log::Warn("retro", "copy texture create failed fmt=%u", cd.Format);
        bb->Release();
        return false;
    }
    if (FAILED(g_dev->CreateShaderResourceView(g_copy, nullptr, &g_copySRV)) || g_copySRV == nullptr)
    {
        bb->Release();
        return false;
    }
    if (FAILED(g_dev->CreateRenderTargetView(bb, nullptr, &g_bbRTV)) || g_bbRTV == nullptr)
    {
        Log::Warn("retro", "backbuffer RTV create failed fmt=%u samples=%u", bd.Format, bd.SampleDesc.Count);
        bb->Release();
        return false;
    }
    g_bbPtr = bb; // keep the reference — we compare pointer identity
    g_bbW = bd.Width; g_bbH = bd.Height;
    g_bbSamples = bd.SampleDesc.Count;
    g_bbFmt = bd.Format;
    Log::Info("retro", "copy chain ready %ux%u fmt=%u", bd.Width, bd.Height, bd.Format);
    return true;
}

void ReleasePostChain() noexcept
{
    if (g_postSRV != nullptr)  { g_postSRV->Release();  g_postSRV = nullptr; }
    if (g_postCopy != nullptr) { g_postCopy->Release(); g_postCopy = nullptr; }
    if (g_postRTV != nullptr)  { g_postRTV->Release();  g_postRTV = nullptr; }
    if (g_postBB != nullptr)   { g_postBB->Release();   g_postBB = nullptr; }
    g_postW = g_postH = 0;
}

// Copy texture + RTV for an arbitrary present target (caller keeps its own ref).
bool EnsurePostChain(ID3D11Texture2D* target) noexcept
{
    if (target == nullptr || g_dev == nullptr)
        return false;
    if (target == g_postBB && g_postCopy != nullptr && g_postSRV != nullptr && g_postRTV != nullptr)
        return true;
    ReleasePostChain();
    D3D11_TEXTURE2D_DESC bd = {};
    target->GetDesc(&bd);
    D3D11_TEXTURE2D_DESC cd = bd;
    cd.MipLevels = 1;
    cd.ArraySize = 1;
    cd.Usage = D3D11_USAGE_DEFAULT;
    cd.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    cd.CPUAccessFlags = 0;
    cd.MiscFlags = 0;
    cd.SampleDesc.Count = 1;
    cd.SampleDesc.Quality = 0;
    if (FAILED(g_dev->CreateTexture2D(&cd, nullptr, &g_postCopy)) || g_postCopy == nullptr ||
        FAILED(g_dev->CreateShaderResourceView(g_postCopy, nullptr, &g_postSRV)) || g_postSRV == nullptr ||
        FAILED(g_dev->CreateRenderTargetView(target, nullptr, &g_postRTV)) || g_postRTV == nullptr)
    {
        Log::Warn("retro", "post chain create failed %ux%u fmt=%u", bd.Width, bd.Height, bd.Format);
        ReleasePostChain();
        return false;
    }
    g_postBB = target;
    target->AddRef();
    g_postW = bd.Width; g_postH = bd.Height;
    g_postSamples = bd.SampleDesc.Count;
    g_postFmt = bd.Format;
    Log::Info("retro", "post chain ready %ux%u fmt=%u", bd.Width, bd.Height, bd.Format);
    return true;
}

ID3D11PixelShader* PsFor(retro::Fx fx) noexcept
{
    switch (fx)
    {
    case retro::Fx::RetroFX:  return g_psRetro;
    case retro::Fx::Ascii:    return g_psAscii;
    case retro::Fx::TermBase: return g_psTermBase;
    case retro::Fx::TermLens: return g_psLens;
    default:                  return nullptr;
    }
}

// 48-byte params cbuffer for a full-target pass of `fx` (each shader declares its own view).
void FillParams(retro::Fx fx, float w, float h, float offX, float offY, float out[12]) noexcept
{
    const float effPixel = g_pixelPx > 1.0f ? g_pixelPx : 1.0f;
    const float effCell  = g_cellPx  > 4.0f ? g_cellPx  : 4.0f;
    float p[12] = { w, h, effPixel, effCell, g_crt, g_vignette, g_palette, 0.0f, offX, offY, 0.0f, 0.0f };
    if (fx == retro::Fx::TermBase)
    {
        p[2] = g_termGain; p[3] = 0.0f;
        p[4] = g_termTint[0]; p[5] = g_termTint[1]; p[6] = g_termTint[2]; p[7] = 0.0f;
    }
    else if (fx == retro::Fx::TermLens)
    {
        p[2] = 0.0f; p[3] = 0.0f;
        p[4] = g_lensCrt; p[5] = g_lensVig; p[6] = g_lensScan; p[7] = 0.0f;
    }
    memcpy(out, p, sizeof(p));
}

// copy (or resolve) src -> sampling texture, then fullscreen `fx` pass into rtv.
void RunFx(ID3D11DeviceContext* ctx, retro::Fx fx, ID3D11Texture2D* src, UINT samples, DXGI_FORMAT fmt,
           ID3D11Texture2D* copy, ID3D11ShaderResourceView* copySRV, ID3D11RenderTargetView* rtv,
           UINT w, UINT h) noexcept
{
    ID3D11PixelShader* ps = PsFor(fx);
    if (ps == nullptr)
        return;
    StateBlock saved;
    saved.Capture(ctx);

    float params[12];
    FillParams(fx, static_cast<float>(w), static_cast<float>(h), 0.0f, 0.0f, params);
    ctx->UpdateSubresource(g_params, 0, nullptr, params, 0, 0);
    if (samples > 1)
        ctx->ResolveSubresource(copy, 0, src, 0, fmt);
    else
        ctx->CopyResource(copy, src);

    retro::BindFullscreenState(ctx);
    ID3D11ShaderResourceView* srvs[2] = { copySRV, g_atlasSRV };
    ctx->PSSetShaderResources(0, 2, srvs);
    ID3D11SamplerState* samps[2] = { g_samp, g_sampLin };
    ctx->PSSetSamplers(0, 2, samps);
    ctx->PSSetShader(ps, nullptr, 0);
    ctx->PSSetConstantBuffers(0, 1, &g_params);
    ctx->OMSetRenderTargets(1, &rtv, nullptr);
    const D3D11_VIEWPORT vp = { 0.0f, 0.0f, static_cast<float>(w), static_cast<float>(h), 0.0f, 1.0f };
    ctx->RSSetViewports(1, &vp);
    ctx->Draw(3, 0);

    // drop our SRVs (next frame's CopyResource writes the copy), then hand the
    // game back exactly the pipeline it had.
    ID3D11ShaderResourceView* nullSrv[2] = { nullptr, nullptr };
    ctx->PSSetShaderResources(0, 2, nullSrv);
    saved.Restore(ctx);
}

} // namespace

ID3DBlob* retro::CompileShader(const char* src, const char* entry, const char* target) noexcept
{
    return Compile(src, entry, target);
}

ID3D11VertexShader* retro::FullscreenVS() noexcept
{
    return g_failed ? nullptr : g_vs;
}

void retro::BindFullscreenState(ID3D11DeviceContext* ctx) noexcept
{
    ctx->OMSetBlendState(g_blend, nullptr, 0xFFFFFFFF);
    ctx->OMSetDepthStencilState(g_ds, 0);
    ctx->RSSetState(g_rs);
    ctx->IASetInputLayout(nullptr);
    ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ctx->VSSetShader(g_vs, nullptr, 0);
    ctx->GSSetShader(nullptr, nullptr, 0);
    ctx->HSSetShader(nullptr, nullptr, 0);
    ctx->DSSetShader(nullptr, nullptr, 0);
}

void retro::SetTunables(float pixelPx, float cellPx) noexcept
{
    g_pixelPx = pixelPx;
    g_cellPx  = cellPx;
}

void retro::SetCrt(float crt, float vignette) noexcept
{
    g_crt      = crt;
    g_vignette = vignette;
}

void retro::SetTermLook(float edgeGain, const float tint[3]) noexcept
{
    g_termGain = edgeGain;
    for (int i = 0; i < 3; ++i)
        g_termTint[i] = tint[i];
}

void retro::SetLens(float crt, float vignette, float scanlines) noexcept
{
    g_lensCrt  = crt;
    g_lensVig  = vignette;
    g_lensScan = scanlines;
}

ID3D11ShaderResourceView* retro::GlyphAtlas() noexcept
{
    return g_atlasSRV;
}

int retro::GlyphIndex(char c) noexcept
{
    if (c >= 'a' && c <= 'z')
        c = static_cast<char>(c - 'a' + 'A');
    for (UINT i = 0; kTextChars[i] != 0; ++i)
        if (kTextChars[i] == c)
            return static_cast<int>(kTextBase + i);
    switch (c) // density-ramp glyphs double as punctuation
    {
    case '.': return 1;
    case ':': return 2;
    case '-': return 3;
    case '=': return 4;
    case '+': return 5;
    case '*': return 6;
    case '#': return 7;
    case '%': return 8;
    default:  return -1; // space / unsupported: no glyph
    }
}

void retro::SetPalette(int palette) noexcept
{
    g_palette = static_cast<float>(palette < 0 ? 0 : (palette > 4 ? 4 : palette));
}

int retro::GetPalette() noexcept
{
    return static_cast<int>(g_palette);
}

void retro::SetRenderScale(int scale) noexcept
{
    g_scalePx = static_cast<float>(scale < 1 ? 1 : scale);
}

void retro::SetGpuBurn(int passes) noexcept
{
    g_burnPasses = passes < 0 ? 0 : passes;
}

void retro::BurnGpu(ID3D11DeviceContext* ctx, IDXGISwapChain* sc) noexcept
{
    if (g_burnPasses <= 0 || ctx == nullptr || sc == nullptr || g_dev == nullptr)
        return;

    // Lazily build the burn shader once (VS/RS/blend/ds come from Ensure).
    if (g_psBurn == nullptr && g_vs != nullptr && !g_failed)
    {
        ID3DBlob* psb = Compile(kBurnPsSrc, "main", "ps_4_0");
        if (psb != nullptr)
        {
            const HRESULT hr = g_dev->CreatePixelShader(psb->GetBufferPointer(), psb->GetBufferSize(), nullptr, &g_psBurn);
            psb->Release();
            if (FAILED(hr))
                Log::Warn("retro", "burn PS create failed hr=0x%08lX", static_cast<unsigned long>(hr));
        }
    }
    if (g_psBurn == nullptr)
        return;

    // Scratch RT sized to the actual swap-chain backbuffer (bound RTs at Present
    // time may be scaled or null — never trust them for dims).
    DXGI_SWAP_CHAIN_DESC sd = {};
    if (FAILED(sc->GetDesc(&sd)) || sd.BufferDesc.Width == 0)
        return;
    const UINT w = sd.BufferDesc.Width, h = sd.BufferDesc.Height;

    if (g_burnTex == nullptr || g_burnW != w || g_burnH != h)
    {
        if (g_burnRTV != nullptr) { g_burnRTV->Release(); g_burnRTV = nullptr; }
        if (g_burnTex != nullptr) { g_burnTex->Release(); g_burnTex = nullptr; }
        D3D11_TEXTURE2D_DESC td = {};
        td.Width = w; td.Height = h;
        td.MipLevels = 1; td.ArraySize = 1;
        td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_DEFAULT;
        td.BindFlags = D3D11_BIND_RENDER_TARGET;
        if (FAILED(g_dev->CreateTexture2D(&td, nullptr, &g_burnTex)) ||
            FAILED(g_dev->CreateRenderTargetView(g_burnTex, nullptr, &g_burnRTV)))
        {
            Log::Warn("retro", "burn scratch RT %ux%u create failed", w, h);
            return;
        }
        g_burnW = w; g_burnH = h;
    }

    StateBlock saved;
    saved.Capture(ctx);

    ID3D11ShaderResourceView* nullSRVs[2] = { nullptr, nullptr };
    ctx->PSSetShaderResources(0, 2, nullSRVs);
    ctx->IASetInputLayout(nullptr);
    ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ctx->VSSetShader(g_vs, nullptr, 0);
    ctx->PSSetShader(g_psBurn, nullptr, 0);
    ctx->OMSetRenderTargets(1, &g_burnRTV, nullptr);
    D3D11_VIEWPORT vp = {};
    vp.Width = static_cast<float>(w); vp.Height = static_cast<float>(h); vp.MaxDepth = 1.0f;
    ctx->RSSetViewports(1, &vp);
    for (int i = 0; i < g_burnPasses; ++i)
        ctx->Draw(3, 0);

    saved.Restore(ctx);
}

void retro::Ensure(ID3D11Device* dev) noexcept
{
    if (dev == nullptr)
        return;
    if (g_dev == dev && (g_vs != nullptr || g_failed))
        return;
    // Presenting device changed (multi-device engines, device-lost recreation):
    // every shader/state/texture/view we own belongs to the old device, and
    // binding them on the new device's context faults inside the driver.
    if (g_dev != nullptr)
    {
        Log::Info("retro", "device changed %p -> %p — rebuilding retro resources", g_dev, dev);
        retro::Reset(); // releases g_dev too
    }
    g_dev = dev;
    dev->AddRef(); // own the device so a game-side Release can't dangle g_dev

    ID3DBlob* vsb = Compile(kVsSrc, "main", "vs_4_0");
    if (vsb == nullptr) { g_failed = true; return; }
    HRESULT hr = dev->CreateVertexShader(vsb->GetBufferPointer(), vsb->GetBufferSize(), nullptr, &g_vs);
    vsb->Release();
    if (FAILED(hr)) { g_failed = true; return; }

    ID3DBlob* rb = Compile(kPsRetroSrc, "main", "ps_4_0");
    if (rb == nullptr) { g_failed = true; return; }
    hr = dev->CreatePixelShader(rb->GetBufferPointer(), rb->GetBufferSize(), nullptr, &g_psRetro);
    rb->Release();
    if (FAILED(hr)) { g_failed = true; return; }

    ID3DBlob* ab = Compile(kPsAsciiSrc, "main", "ps_4_0");
    if (ab == nullptr) { g_failed = true; return; }
    hr = dev->CreatePixelShader(ab->GetBufferPointer(), ab->GetBufferSize(), nullptr, &g_psAscii);
    ab->Release();
    if (FAILED(hr)) { g_failed = true; return; }

    // Terminator shaders are optional: a failure only disables that layer.
    ID3DBlob* tb = Compile(kPsTermBaseSrc, "main", "ps_4_0");
    if (tb == nullptr || FAILED(dev->CreatePixelShader(tb->GetBufferPointer(), tb->GetBufferSize(), nullptr, &g_psTermBase)))
        Log::Warn("retro", "Terminator base PS unavailable — HLSL fallback disabled");
    if (tb != nullptr) tb->Release();
    ID3DBlob* lb = Compile(kPsLensSrc, "main", "ps_4_0");
    if (lb == nullptr || FAILED(dev->CreatePixelShader(lb->GetBufferPointer(), lb->GetBufferSize(), nullptr, &g_psLens)))
        Log::Warn("retro", "Terminator lens PS unavailable — lens disabled");
    if (lb != nullptr) lb->Release();

    D3D11_SAMPLER_DESC sd = {};
    sd.Filter   = D3D11_FILTER_MIN_MAG_MIP_POINT;
    sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    sd.MaxLOD   = D3D11_FLOAT32_MAX;
    dev->CreateSamplerState(&sd, &g_samp);
    sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    dev->CreateSamplerState(&sd, &g_sampLin);

    D3D11_RASTERIZER_DESC rd = {};
    rd.FillMode = D3D11_FILL_SOLID;
    rd.CullMode = D3D11_CULL_NONE;
    rd.DepthClipEnable = TRUE;
    dev->CreateRasterizerState(&rd, &g_rs);

    D3D11_BLEND_DESC bd = {};
    bd.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    dev->CreateBlendState(&bd, &g_blend);

    D3D11_DEPTH_STENCIL_DESC dd = {};
    dd.DepthEnable    = FALSE;
    dd.StencilEnable  = FALSE;
    dev->CreateDepthStencilState(&dd, &g_ds);

    D3D11_BUFFER_DESC pd = {};
    pd.ByteWidth = 48;
    pd.Usage     = D3D11_USAGE_DEFAULT;
    pd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    dev->CreateBuffer(&pd, nullptr, &g_params);

    if (!BuildAtlas(dev))
        Log::Warn("retro", "glyph atlas build failed — Ascii mode degraded");

    Log::Info("retro", "retro pass resources built (vs/ps/sampler/atlas)");
}

void retro::Apply(ID3D11DeviceContext* ctx, IDXGISwapChain* sc, Fx fx) noexcept
{
    if (fx == Fx::None || ctx == nullptr || sc == nullptr || g_failed)
        return;
    if (g_dev == nullptr || g_vs == nullptr)
        return; // Ensure() must run first

    if (!EnsureCopyChain(sc))
        return;

    // Engines with a state cache (Unigine) never re-bind "unchanged" state, so
    // RunFx restores everything it touches or the next frame renders into a null
    // RTV (black backbuffer). The pass runs on the real (shrunken) backbuffer, so
    // 1.0 = one real buffer pixel — no grid-alignment floor needed.
    RunFx(ctx, fx, g_bbPtr, g_bbSamples, g_bbFmt, g_copy, g_copySRV, g_bbRTV, g_bbW, g_bbH);
}

bool retro::ApplyTo(ID3D11DeviceContext* ctx, ID3D11Texture2D* target, Fx fx) noexcept
{
    if (fx == Fx::None || ctx == nullptr || target == nullptr || g_failed || g_vs == nullptr ||
        PsFor(fx) == nullptr || !EnsurePostChain(target))
        return false;
    RunFx(ctx, fx, g_postBB, g_postSamples, g_postFmt, g_postCopy, g_postSRV, g_postRTV, g_postW, g_postH);
    return true;
}

ID3D11RenderTargetView* retro::PostRTV(ID3D11Texture2D* target, UINT* w, UINT* h) noexcept
{
    if (g_failed || !EnsurePostChain(target))
        return nullptr;
    if (w != nullptr) *w = g_postW;
    if (h != nullptr) *h = g_postH;
    return g_postRTV;
}

namespace
{

ID3D11Texture2D*          g_proxySrc = nullptr;
ID3D11ShaderResourceView* g_proxySRV = nullptr;

bool UpscaleInto(ID3D11DeviceContext* ctx, ID3D11ShaderResourceView* srcSRV, UINT srcW, UINT srcH,
                 IDXGISwapChain* dst, retro::Fx fx) noexcept
{
    using retro::Fx;
    if (g_psBlit == nullptr)
    {
        ID3DBlob* b = Compile(kBlitPsSrc, "main", "ps_4_0");
        if (b != nullptr)
        {
            const HRESULT hr = g_dev->CreatePixelShader(b->GetBufferPointer(), b->GetBufferSize(), nullptr, &g_psBlit);
            b->Release();
            if (FAILED(hr)) { Log::Warn("retro", "blit PS failed"); return false; }
        }
        else return false;
    }
    if (g_upCB == nullptr)
    {
        D3D11_BUFFER_DESC cd = {};
        cd.ByteWidth = 16; cd.Usage = D3D11_USAGE_DEFAULT; cd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        if (FAILED(g_dev->CreateBuffer(&cd, nullptr, &g_upCB)))
            return false;
    }
    if (srcSRV == nullptr || srcW == 0 || srcH == 0)
        return false;

    // dst chain RTV — rebuilt if the chain was recreated by HookManager.
    if (g_upSC != dst || g_upRTV == nullptr)
    {
        if (g_upRTV != nullptr) { g_upRTV->Release(); g_upRTV = nullptr; }
        ID3D11Texture2D* dbb = nullptr;
        if (FAILED(dst->GetBuffer(0, __uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&dbb))) || dbb == nullptr)
            return false;
        D3D11_TEXTURE2D_DESC dtd = {};
        dbb->GetDesc(&dtd);
        g_upW = dtd.Width; g_upH = dtd.Height;
        const HRESULT hr = g_dev->CreateRenderTargetView(dbb, nullptr, &g_upRTV);
        dbb->Release();
        if (FAILED(hr) || g_upRTV == nullptr)
            return false;
        g_upSC = dst;
        Log::Info("retro", "upscale present target %ux%u from %ux%u", g_upW, g_upH, srcW, srcH);
    }

    // Integer factor + centered letterbox rect.
    // Effects resample freely, so they fill the whole output instead.
    UINT f = g_upW / srcW;
    if (g_upH / srcH < f) f = g_upH / srcH;
    if (f == 0) f = 1;
    const bool withFx = (fx == Fx::RetroFX || fx == Fx::Ascii) && g_psRetro != nullptr && g_psAscii != nullptr;
    const UINT dw = withFx ? g_upW : srcW * f, dh = withFx ? g_upH : srcH * f;
    const float offX = (g_upW - dw) * 0.5f, offY = (g_upH - dh) * 0.5f;
    const float cb[4] = { offX, offY, static_cast<float>(dw), static_cast<float>(dh) };
    ctx->UpdateSubresource(g_upCB, 0, nullptr, cb, 0, 0);
    if (withFx)
    {
        const float effPixel = g_pixelPx > 1.0f ? g_pixelPx : 1.0f;
        const float effCell  = g_cellPx  > 4.0f ? g_cellPx  : 4.0f;
        const float params[12] = { static_cast<float>(dw), static_cast<float>(dh),
                                   effPixel, effCell, g_crt, g_vignette, g_palette, 0.0f,
                                   offX, offY, 0.0f, 0.0f };
        ctx->UpdateSubresource(g_params, 0, nullptr, params, 0, 0);
    }

    StateBlock saved;
    saved.Capture(ctx);

    const float black[4] = { 0.f, 0.f, 0.f, 1.f };
    ctx->ClearRenderTargetView(g_upRTV, black);
    ctx->OMSetRenderTargets(1, &g_upRTV, nullptr);
    ctx->OMSetBlendState(g_blend, nullptr, 0xFFFFFFFF);
    ctx->OMSetDepthStencilState(g_ds, 0);
    ctx->RSSetState(g_rs);
    D3D11_VIEWPORT vp = { offX, offY, static_cast<float>(dw), static_cast<float>(dh), 0.0f, 1.0f };
    ctx->RSSetViewports(1, &vp);
    ctx->IASetInputLayout(nullptr);
    ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ctx->VSSetShader(g_vs, nullptr, 0);
    ctx->GSSetShader(nullptr, nullptr, 0);
    if (withFx)
    {
        ID3D11ShaderResourceView* srvs[2] = { srcSRV, g_atlasSRV };
        ID3D11SamplerState* samps[2] = { g_samp, g_sampLin };
        ctx->PSSetShader(fx == Fx::Ascii ? g_psAscii : g_psRetro, nullptr, 0);
        ctx->PSSetConstantBuffers(0, 1, &g_params);
        ctx->PSSetShaderResources(0, 2, srvs);
        ctx->PSSetSamplers(0, 2, samps);
    }
    else
    {
        ctx->PSSetShader(g_psBlit, nullptr, 0);
        ctx->PSSetConstantBuffers(0, 1, &g_upCB);
        ctx->PSSetShaderResources(0, 1, &srcSRV);
        ctx->PSSetSamplers(0, 1, &g_samp);   // point — true integer scaling
    }
    ctx->Draw(3, 0);

    ID3D11ShaderResourceView* nullSrv[2] = { nullptr, nullptr };
    ctx->PSSetShaderResources(0, 2, nullSrv);
    saved.Restore(ctx);
    return true;
}

} // namespace

void retro::UpscaleBlit(ID3D11DeviceContext* ctx, IDXGISwapChain* src, IDXGISwapChain* dst, Fx fx) noexcept
{
    if (ctx == nullptr || src == nullptr || dst == nullptr ||
        g_dev == nullptr || g_vs == nullptr || g_failed)
        return;

    // Source frame: copy the (shrunken) src backbuffer into the SRV-able copy tex.
    if (!EnsureCopyChain(src))
        return;
    if (g_bbSamples > 1)
        ctx->ResolveSubresource(g_copy, 0, g_bbPtr, 0, g_bbFmt);
    else
        ctx->CopyResource(g_copy, g_bbPtr);
    UpscaleInto(ctx, g_copySRV, g_bbW, g_bbH, dst, fx);
}

bool retro::UpscaleTexture(ID3D11DeviceContext* ctx, ID3D11Texture2D* src, IDXGISwapChain* dst, Fx fx) noexcept
{
    if (ctx == nullptr || src == nullptr || dst == nullptr ||
        g_dev == nullptr || g_vs == nullptr || g_failed)
        return false;
    if (g_proxySrc != src || g_proxySRV == nullptr)
    {
        if (g_proxySRV != nullptr) { g_proxySRV->Release(); g_proxySRV = nullptr; }
        g_proxySrc = nullptr;
        if (FAILED(g_dev->CreateShaderResourceView(src, nullptr, &g_proxySRV)) || g_proxySRV == nullptr)
        {
            Log::Warn("retro", "proxy backbuffer SRV create failed");
            return false;
        }
        g_proxySrc = src;
    }
    D3D11_TEXTURE2D_DESC td = {};
    src->GetDesc(&td);
    return UpscaleInto(ctx, g_proxySRV, td.Width, td.Height, dst, fx);
}

void retro::DropUpscaleChain() noexcept
{
    if (g_upRTV != nullptr) { g_upRTV->Release(); g_upRTV = nullptr; }
    if (g_proxySRV != nullptr) { g_proxySRV->Release(); g_proxySRV = nullptr; }
    g_proxySrc = nullptr;
    g_upSC = nullptr;
    g_upW = g_upH = 0;
}

void retro::ReleaseBackbuffer() noexcept
{
    auto rel = [](IUnknown*& p) { if (p != nullptr) { p->Release(); p = nullptr; } };
    rel(reinterpret_cast<IUnknown*&>(g_copySRV));
    rel(reinterpret_cast<IUnknown*&>(g_copy));
    rel(reinterpret_cast<IUnknown*&>(g_bbRTV));
    if (g_bbPtr != nullptr) { g_bbPtr->Release(); g_bbPtr = nullptr; }
    g_bbW = g_bbH = 0;
    ReleasePostChain();
}

void retro::Reset() noexcept
{
    ReleasePostChain();
    auto rel = [](IUnknown*& p) { if (p != nullptr) { p->Release(); p = nullptr; } };
    rel(reinterpret_cast<IUnknown*&>(g_psTermBase));
    rel(reinterpret_cast<IUnknown*&>(g_psLens));
    rel(reinterpret_cast<IUnknown*&>(g_copy));
    rel(reinterpret_cast<IUnknown*&>(g_copySRV));
    rel(reinterpret_cast<IUnknown*&>(g_bbRTV));
    if (g_bbPtr != nullptr) { g_bbPtr->Release(); g_bbPtr = nullptr; }
    rel(reinterpret_cast<IUnknown*&>(g_atlas));
    rel(reinterpret_cast<IUnknown*&>(g_atlasSRV));
    rel(reinterpret_cast<IUnknown*&>(g_burnTex));
    rel(reinterpret_cast<IUnknown*&>(g_burnRTV));
    rel(reinterpret_cast<IUnknown*&>(g_psBurn));
    rel(reinterpret_cast<IUnknown*&>(g_upRTV));
    rel(reinterpret_cast<IUnknown*&>(g_psBlit));
    rel(reinterpret_cast<IUnknown*&>(g_upCB));
    rel(reinterpret_cast<IUnknown*&>(g_proxySRV));
    g_proxySrc = nullptr;
    g_upSC = nullptr;
    g_upW = g_upH = 0;
    g_burnW = g_burnH = 0;
    rel(reinterpret_cast<IUnknown*&>(g_params));
    rel(reinterpret_cast<IUnknown*&>(g_ds));
    rel(reinterpret_cast<IUnknown*&>(g_blend));
    rel(reinterpret_cast<IUnknown*&>(g_rs));
    rel(reinterpret_cast<IUnknown*&>(g_samp));
    rel(reinterpret_cast<IUnknown*&>(g_sampLin));
    rel(reinterpret_cast<IUnknown*&>(g_psAscii));
    rel(reinterpret_cast<IUnknown*&>(g_psRetro));
    rel(reinterpret_cast<IUnknown*&>(g_vs));
    rel(reinterpret_cast<IUnknown*&>(g_dev)); // owned since Ensure AddRefs
    g_bbW = g_bbH = 0;
    g_failed = false;
    g_compile = nullptr;
    g_compileTried = false;
}
