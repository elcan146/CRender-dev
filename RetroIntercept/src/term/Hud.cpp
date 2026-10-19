// Hud — see header. One instanced draw: each instance is a pixel-space quad that is either
// an atlas glyph, a solid rect or an anti-aliased ring. Instance data lives in a dynamic
// Buffer<float4> read by the VS (3 float4 per instance), so no input layout/VB is bound.
#include "Hud.h"
#include "../d3d11/RetroPass.h"
#include "../d3d11/StateBlock.h"
#include "../core/Log.h"

#include <Windows.h>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace
{

constexpr int kMaxInst = 4096;

struct Inst
{
    float r[4]; // x0, y0, x1, y1 (px)
    float k[4]; // kind (0 glyph, 1 rect, 2 ring), glyph index, ring thickness (0..1 of radius), 0
    float c[4]; // rgba
};

Inst g_inst[kMaxInst];
int  g_n = 0;

ID3D11Device*             g_dev     = nullptr; // owning
ID3D11VertexShader*       g_vs      = nullptr;
ID3D11PixelShader*        g_ps      = nullptr;
ID3D11Buffer*             g_instBuf = nullptr;
ID3D11ShaderResourceView* g_instSRV = nullptr;
ID3D11Buffer*             g_cb      = nullptr;
ID3D11BlendState*         g_blend   = nullptr;
bool                      g_failed  = false;

hud::Config   g_cfg;
SRWLOCK       g_cfgLock = SRWLOCK_INIT;
LARGE_INTEGER g_t0 = {};
LARGE_INTEGER g_freq = {};

const char kVsSrc[] =
    "Buffer<float4> inst : register(t0);"
    "cbuffer H : register(b0) { float2 rtDim; float2 pad; };"
    "struct VO { float4 pos : SV_Position; float2 uv : TEXCOORD0;"
    "            nointerpolation float4 k : TEXCOORD1; nointerpolation float4 col : COLOR0; };"
    "VO main(uint vid : SV_VertexID, uint iid : SV_InstanceID) {"
    "  float4 r = inst.Load(iid * 3u);"
    "  float4 k = inst.Load(iid * 3u + 1u);"
    "  float4 c = inst.Load(iid * 3u + 2u);"
    "  uint v = vid % 6u;"
    "  float2 t = float2((v == 1u || v == 4u || v == 5u) ? 1.0 : 0.0,"
    "                    (v == 2u || v == 3u || v == 5u) ? 1.0 : 0.0);"
    "  float2 p = lerp(r.xy, r.zw, t);"
    "  VO o;"
    "  o.pos = float4(p.x / rtDim.x * 2.0 - 1.0, 1.0 - p.y / rtDim.y * 2.0, 0.0, 1.0);"
    "  o.uv = t; o.k = k; o.col = c;"
    "  return o;"
    "}";

const char kPsSrc[] =
    "Texture2D atlas : register(t0);"
    "struct VO { float4 pos : SV_Position; float2 uv : TEXCOORD0;"
    "            nointerpolation float4 k : TEXCOORD1; nointerpolation float4 col : COLOR0; };"
    "float4 main(VO i) : SV_Target {"
    "  float a = 1.0;"
    "  if (i.k.x < 0.5) {"
    "    int2 tx = (int2)min(floor(i.uv * 8.0), 7.0);"
    "    a = atlas.Load(int3((int)i.k.y * 8 + tx.x, tx.y, 0)).r;"
    "  } else if (i.k.x > 1.5) {"
    "    float r = length(i.uv * 2.0 - 1.0);"
    "    float aa = max(fwidth(r), 1e-4);"
    "    a = saturate((r - (1.0 - i.k.z)) / aa + 0.5) * saturate((1.0 - r) / aa + 0.5);"
    "  }"
    "  clip(a - 0.01);"
    "  return float4(i.col.rgb, i.col.a * a);"
    "}";

template <class T>
void Rel(T*& p) noexcept
{
    if (p != nullptr) { p->Release(); p = nullptr; }
}

bool Build(ID3D11Device* dev) noexcept
{
    ID3DBlob* vb = retro::CompileShader(kVsSrc, "main", "vs_4_0");
    ID3DBlob* pb = retro::CompileShader(kPsSrc, "main", "ps_4_0");
    bool ok = vb != nullptr && pb != nullptr &&
              SUCCEEDED(dev->CreateVertexShader(vb->GetBufferPointer(), vb->GetBufferSize(), nullptr, &g_vs)) &&
              SUCCEEDED(dev->CreatePixelShader(pb->GetBufferPointer(), pb->GetBufferSize(), nullptr, &g_ps));
    if (vb != nullptr) vb->Release();
    if (pb != nullptr) pb->Release();
    if (!ok)
        return false;

    D3D11_BUFFER_DESC bd = {};
    bd.ByteWidth      = static_cast<UINT>(sizeof(g_inst));
    bd.Usage          = D3D11_USAGE_DYNAMIC;
    bd.BindFlags      = D3D11_BIND_SHADER_RESOURCE;
    bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    D3D11_SHADER_RESOURCE_VIEW_DESC sd = {};
    sd.Format              = DXGI_FORMAT_R32G32B32A32_FLOAT;
    sd.ViewDimension       = D3D11_SRV_DIMENSION_BUFFER;
    sd.Buffer.FirstElement = 0;
    sd.Buffer.NumElements  = kMaxInst * 3;
    if (FAILED(dev->CreateBuffer(&bd, nullptr, &g_instBuf)) ||
        FAILED(dev->CreateShaderResourceView(g_instBuf, &sd, &g_instSRV)))
        return false;

    D3D11_BUFFER_DESC cd = {};
    cd.ByteWidth = 16;
    cd.Usage     = D3D11_USAGE_DEFAULT;
    cd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    if (FAILED(dev->CreateBuffer(&cd, nullptr, &g_cb)))
        return false;

    D3D11_BLEND_DESC bl = {};
    bl.RenderTarget[0].BlendEnable           = TRUE;
    bl.RenderTarget[0].SrcBlend              = D3D11_BLEND_SRC_ALPHA;
    bl.RenderTarget[0].DestBlend             = D3D11_BLEND_INV_SRC_ALPHA;
    bl.RenderTarget[0].BlendOp               = D3D11_BLEND_OP_ADD;
    bl.RenderTarget[0].SrcBlendAlpha         = D3D11_BLEND_ZERO;
    bl.RenderTarget[0].DestBlendAlpha        = D3D11_BLEND_ONE;
    bl.RenderTarget[0].BlendOpAlpha          = D3D11_BLEND_OP_ADD;
    bl.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    return SUCCEEDED(dev->CreateBlendState(&bl, &g_blend));
}

bool Ensure(ID3D11Device* dev) noexcept
{
    if (dev == nullptr)
        return false;
    if (dev == g_dev)
        return !g_failed;
    hud::Reset();
    g_dev = dev;
    dev->AddRef();
    if (!Build(dev))
    {
        Log::Warn("hud", "HUD resources unavailable — HUD disabled");
        g_failed = true;
        return false;
    }
    Log::Info("hud", "HUD resources built");
    return true;
}

// ---- instance builders ------------------------------------------------------------

void Push(float x0, float y0, float x1, float y1, float kind, float glyph, float param, const float col[4]) noexcept
{
    if (g_n >= kMaxInst)
        return;
    Inst& i = g_inst[g_n++];
    i.r[0] = x0; i.r[1] = y0; i.r[2] = x1; i.r[3] = y1;
    i.k[0] = kind; i.k[1] = glyph; i.k[2] = param; i.k[3] = 0.0f;
    memcpy(i.c, col, sizeof(i.c));
}

void Rect(float x0, float y0, float x1, float y1, const float col[4]) noexcept
{
    Push(x0, y0, x1, y1, 1.0f, 0.0f, 0.0f, col);
}

void Ring(float cx, float cy, float r, float thickPx, const float col[4]) noexcept
{
    if (r <= 1.0f)
        return;
    Push(cx - r, cy - r, cx + r, cy + r, 2.0f, 0.0f, thickPx / r, col);
}

// 8x8 glyph cell drawn at s px per texel; 6*s advance (glyphs are 5 columns + gap).
float Text(float x, float y, float s, const char* str, const float col[4]) noexcept
{
    for (const char* p = str; *p != 0; ++p, x += 6.0f * s)
    {
        const int g = retro::GlyphIndex(*p);
        if (g >= 0)
            Push(x, y, x + 8.0f * s, y + 8.0f * s, 0.0f, static_cast<float>(g), 0.0f, col);
    }
    return x;
}

float TextWidth(const char* str, float s) noexcept
{
    return static_cast<float>(strlen(str)) * 6.0f * s;
}

unsigned Hash(unsigned k, unsigned seed) noexcept
{
    unsigned h = k * 2654435761u ^ seed;
    h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
    return h;
}

void GenLeft(unsigned k, char* out, size_t n) noexcept
{
    static const char* kWords[] = { "STRUCTURE", "THERMAL", "MOTION", "PROFILE", "CONTOUR", "MASS", "VECTOR", "SIGNAL" };
    static const char* kThreat[] = { "LOW", "MED", "HIGH", "NONE" };
    const unsigned r = Hash(k, 0x9E3779B9u);
    switch (r % 9u)
    {
    case 0:  _snprintf_s(out, n, _TRUNCATE, "0x%06X %04X", (r >> 4) & 0xFFFFFFu, (r >> 8) & 0xFFFFu); break;
    case 1:  _snprintf_s(out, n, _TRUNCATE, "ANALYSIS: %s", kWords[(r >> 5) % 8u]); break;
    case 2:  _snprintf_s(out, n, _TRUNCATE, "SCAN %02u%%", (r >> 3) % 100u); break;
    case 3:  _snprintf_s(out, n, _TRUNCATE, "%s", ((r >> 7) % 3u) == 0 ? "MATCH FOUND" : "NO MATCH"); break;
    case 4:  _snprintf_s(out, n, _TRUNCATE, "ID %04X-%02X", (r >> 6) & 0xFFFFu, (r >> 2) & 0xFFu); break;
    case 5:  _snprintf_s(out, n, _TRUNCATE, "THREAT ASSESS: %s", kThreat[(r >> 9) % 4u]); break;
    case 6:  _snprintf_s(out, n, _TRUNCATE, "VECTOR %+04d %+04d", static_cast<int>((r >> 4) % 999u) - 499,
                         static_cast<int>((r >> 14) % 999u) - 499); break;
    case 7:  _snprintf_s(out, n, _TRUNCATE, "%08X", r); break;
    default: _snprintf_s(out, n, _TRUNCATE, "PRIORITY %u", (r >> 11) % 10u); break;
    }
}

void GenRight(unsigned k, char* out, size_t n) noexcept
{
    const unsigned r = Hash(k, 0x7F4A7C15u);
    const float f = static_cast<float>((r >> 8) % 100000u) / 100.0f;
    switch (r % 8u)
    {
    case 0:  _snprintf_s(out, n, _TRUNCATE, "X %07.2f", f); break;
    case 1:  _snprintf_s(out, n, _TRUNCATE, "Y %07.2f", f); break;
    case 2:  _snprintf_s(out, n, _TRUNCATE, "Z %07.2f", f); break;
    case 3:  _snprintf_s(out, n, _TRUNCATE, "HDG %03u", (r >> 5) % 360u); break;
    case 4:  _snprintf_s(out, n, _TRUNCATE, "RNG %05.1fM", f * 0.1f); break;
    case 5:  _snprintf_s(out, n, _TRUNCATE, "PWR %02u.%u%%", (r >> 4) % 100u, (r >> 12) % 10u); break;
    case 6:  _snprintf_s(out, n, _TRUNCATE, "MEM %04X:%04X", (r >> 3) & 0xFFFFu, (r >> 16) & 0xFFFFu); break;
    default: _snprintf_s(out, n, _TRUNCATE, "%s", ((r >> 6) & 1u) ? "SYNC OK" : "TRACKING"); break;
    }
}

float Clamp01(float v) noexcept
{
    return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

// One scrolling column. right=true aligns each line to xEdge from the right.
void Column(bool right, float xEdge, float top, int lines, float lineH, float s, float scroll,
            const float col[4]) noexcept
{
    const float fl = floorf(scroll);
    const float fr = scroll - fl;
    char buf[48];
    for (int i = 0; i <= lines; ++i)
    {
        const float row = static_cast<float>(i) - fr;
        const float edge = row < (lines - 1 - row) ? row : (lines - 1 - row);
        const float a = Clamp01(edge / 2.0f);
        if (a <= 0.0f)
            continue;
        const unsigned k = static_cast<unsigned>(fl) + static_cast<unsigned>(i);
        if (right) GenRight(k, buf, sizeof(buf));
        else       GenLeft(k, buf, sizeof(buf));
        const float c[4] = { col[0], col[1], col[2], col[3] * a };
        const float x = right ? xEdge - TextWidth(buf, s) : xEdge;
        Text(x, top + row * lineH, s, buf, c);
    }
}

void Reticle(float w, float h, float s, float t, float lineH) noexcept
{
    static const float kCol[4] = { 1.0f, 0.85f, 0.85f, 0.8f };
    const float cx = w * 0.5f, cy = h * 0.5f, R = h * 0.055f;
    Ring(cx, cy, R, 1.5f * s, kCol);
    const float th = s;
    Rect(cx + R * 1.15f, cy - th * 0.5f, cx + R * 1.7f, cy + th * 0.5f, kCol);
    Rect(cx - R * 1.7f, cy - th * 0.5f, cx - R * 1.15f, cy + th * 0.5f, kCol);
    Rect(cx - th * 0.5f, cy + R * 1.15f, cx + th * 0.5f, cy + R * 1.7f, kCol);
    Rect(cx - th * 0.5f, cy - R * 1.7f, cx + th * 0.5f, cy - R * 1.15f, kCol);
    Rect(cx - s, cy - s, cx + s, cy + s, kCol);
    for (int i = 0; i < 2; ++i) // two orbiting ticks inside the ring
    {
        const float a = t * 1.2f + static_cast<float>(i) * 3.14159265f;
        const float px = cx + cosf(a) * R * 0.78f, py = cy + sinf(a) * R * 0.78f;
        Rect(px - s, py - s, px + s, py + s, kCol);
    }
    char buf[48];
    const float tx = cx + R * 1.9f, ty = cy - R * 0.9f;
    _snprintf_s(buf, _TRUNCATE, "X %06.1f Y %06.1f", cx + sinf(t * 0.7f) * 37.3f, cy + cosf(t * 0.53f) * 21.9f);
    Text(tx, ty, s, buf, kCol);
    _snprintf_s(buf, _TRUNCATE, "RNG %06.1f", 120.0f + 80.0f * sinf(t * 0.37f) + fmodf(t * 13.7f, 1.0f));
    Text(tx, ty + lineH, s, buf, kCol);
}

// Dotted normalized segments (sight lines / tethers). Dots chase along the line.
void SightLines(const hud::Line* segs, int n, float w, float h, float s, float t) noexcept
{
    static const float kCol[4] = { 1.0f, 0.25f, 0.2f, 0.75f };
    for (int i = 0; i < n && i < 32; ++i)
    {
        const float x0 = segs[i].x0 * w, y0 = segs[i].y0 * h;
        const float dx = segs[i].x1 * w - x0, dy = segs[i].y1 * h - y0;
        const float len = sqrtf(dx * dx + dy * dy);
        if (len < 8.0f)
            continue;
        const int dots = static_cast<int>(len / (4.0f * s));
        const float phase = fmodf(t * 24.0f, 4.0f * s);
        for (int d = 0; d < dots && d < 96; ++d)
        {
            const float u = static_cast<float>(d) * 4.0f * s + phase;
            if (u > len)
                break;
            const float px = x0 + dx * (u / len), py = y0 + dy * (u / len);
            Rect(px - s, py - s, px + s, py + s, kCol);
        }
    }
}

void Brackets(const term::Track* tr, int n, const char* const* labels, float w, float h,
              float s, float lineH) noexcept
{
    static const float kNorm[4] = { 1.0f, 0.82f, 0.82f, 0.95f };
    static const float kRed[4]  = { 1.0f, 0.12f, 0.08f, 1.0f };
    for (int i = 0; i < n; ++i)
    {
        const term::Track& t = tr[i];
        float x0 = t.x0 * w, y0 = t.y0 * h, x1 = t.x1 * w, y1 = t.y1 * h;
        const float k = Clamp01(t.age / 0.25f);
        const float ease = 1.0f - (1.0f - k) * (1.0f - k) * (1.0f - k);
        const float sc = 1.0f + 0.6f * (1.0f - ease); // snap-in from 1.6x
        const float cx = (x0 + x1) * 0.5f, cy = (y0 + y1) * 0.5f;
        const float hw = (x1 - x0) * 0.5f * sc, hh = (y1 - y0) * 0.5f * sc;
        x0 = cx - hw; x1 = cx + hw; y0 = cy - hh; y1 = cy + hh;
        const bool flash = t.locked && t.lockAge < 0.6f && fmodf(t.lockAge * 6.0f, 1.0f) < 0.5f;
        float col[4];
        memcpy(col, flash ? kRed : kNorm, sizeof(col));
        col[3] *= ease * (t.unseen > 0.0f ? 0.5f : 1.0f);
        const float m = (hw < hh ? hw : hh) * 2.0f;
        float L = m * 0.25f;
        if (L < 4.0f * s) L = 4.0f * s;
        const float th = s;
        Rect(x0, y0, x0 + L, y0 + th, col); Rect(x0, y0, x0 + th, y0 + L, col);
        Rect(x1 - L, y0, x1, y0 + th, col); Rect(x1 - th, y0, x1, y0 + L, col);
        Rect(x0, y1 - th, x0 + L, y1, col); Rect(x0, y1 - L, x0 + th, y1, col);
        Rect(x1 - L, y1 - th, x1, y1, col); Rect(x1 - th, y1 - L, x1, y1, col);
        char buf[32];
        const char* lbl = labels != nullptr ? labels[i] : nullptr;
        if (lbl != nullptr && lbl[0] != 0)
            _snprintf_s(buf, _TRUNCATE, t.locked ? "%s %02d%% LOCK" : "%s %02d%%", lbl,
                        static_cast<int>(t.score * 100.0f + 0.5f));
        else
            _snprintf_s(buf, _TRUNCATE, t.locked ? "TGT %02d LOCK" : "TGT %02d", t.id);
        const float ly = y0 - lineH >= 0.0f ? y0 - lineH : y1 + 2.0f * s;
        Text(x0, ly, s, buf, t.locked ? kRed : col);
    }
}

} // namespace

void hud::SetConfig(const Config& c) noexcept
{
    AcquireSRWLockExclusive(&g_cfgLock);
    g_cfg = c;
    ReleaseSRWLockExclusive(&g_cfgLock);
}

hud::Config hud::GetConfig() noexcept
{
    AcquireSRWLockShared(&g_cfgLock);
    const Config c = g_cfg;
    ReleaseSRWLockShared(&g_cfgLock);
    return c;
}

void hud::Draw(ID3D11DeviceContext* ctx, ID3D11RenderTargetView* rtv, UINT w, UINT h,
               const term::Track* tracks, int nTracks,
               const char* const* status, int nStatus,
               const char* const* labels,
               const Line* lines, int nLines) noexcept
{
    const Config cfg = GetConfig();
    if (!cfg.enabled || ctx == nullptr || rtv == nullptr || w < 64 || h < 64)
        return;
    ID3D11ShaderResourceView* atlas = retro::GlyphAtlas();
    if (atlas == nullptr)
        return;
    ID3D11Device* dev = nullptr;
    ctx->GetDevice(&dev);
    const bool ready = Ensure(dev);
    if (dev != nullptr)
        dev->Release();
    if (!ready)
        return;

    if (g_freq.QuadPart == 0)
    {
        QueryPerformanceFrequency(&g_freq);
        QueryPerformanceCounter(&g_t0);
    }
    LARGE_INTEGER now = {};
    QueryPerformanceCounter(&now);
    const float t = static_cast<float>(static_cast<double>(now.QuadPart - g_t0.QuadPart) /
                                       static_cast<double>(g_freq.QuadPart)) * cfg.speed;

    const float fw = static_cast<float>(w), fh = static_cast<float>(h);
    const float s = static_cast<float>((h + 240u) / 480u < 1u ? 1u : (h + 240u) / 480u);
    const float m = 12.0f * s, lineH = 10.0f * s;
    static const float kText[4] = { 1.0f, 0.82f, 0.82f, 0.85f };
    static const float kHead[4] = { 1.0f, 0.95f, 0.95f, 0.95f };

    g_n = 0;
    if (cfg.text)
    {
        const int colLines = static_cast<int>((fh * 0.62f) / lineH) < 28 ? static_cast<int>((fh * 0.62f) / lineH) : 28;
        Text(m, m, s, "SYS ANALYSIS", kHead);
        Column(false, m, m + lineH * 2.0f, colLines, lineH, s, t * 1.6f, kText);
        float y = m;
        for (int i = 0; status != nullptr && i < nStatus && i < 8; ++i, y += lineH)
            if (status[i] != nullptr)
                Text(fw - m - TextWidth(status[i], s), y, s, status[i], kHead);
        Column(true, fw - m, y + lineH, colLines - nStatus, lineH, s, t * 1.1f + 37.0f, kText);
    }
    if (cfg.reticle)
        Reticle(fw, fh, s, t, lineH);
    if (lines != nullptr && nLines > 0)
        SightLines(lines, nLines, fw, fh, s, t);
    if (cfg.brackets && tracks != nullptr)
        Brackets(tracks, nTracks, labels, fw, fh, s, lineH);
    if (g_n == 0)
        return;

    retro::StateBlock saved;
    saved.Capture(ctx);

    D3D11_MAPPED_SUBRESOURCE ms = {};
    if (SUCCEEDED(ctx->Map(g_instBuf, 0, D3D11_MAP_WRITE_DISCARD, 0, &ms)) && ms.pData != nullptr)
    {
        memcpy(ms.pData, g_inst, static_cast<size_t>(g_n) * sizeof(Inst));
        ctx->Unmap(g_instBuf, 0);
        const float cb[4] = { fw, fh, 0.0f, 0.0f };
        ctx->UpdateSubresource(g_cb, 0, nullptr, cb, 0, 0);

        retro::BindFullscreenState(ctx);
        ctx->OMSetBlendState(g_blend, nullptr, 0xFFFFFFFF);
        ctx->VSSetShader(g_vs, nullptr, 0);
        ctx->VSSetShaderResources(0, 1, &g_instSRV);
        ctx->VSSetConstantBuffers(0, 1, &g_cb);
        ctx->PSSetShader(g_ps, nullptr, 0);
        ctx->PSSetShaderResources(0, 1, &atlas);
        ctx->OMSetRenderTargets(1, &rtv, nullptr);
        const D3D11_VIEWPORT vp = { 0.0f, 0.0f, fw, fh, 0.0f, 1.0f };
        ctx->RSSetViewports(1, &vp);
        ctx->DrawInstanced(6, static_cast<UINT>(g_n), 0, 0);

        ID3D11ShaderResourceView* nullSrv = nullptr;
        ctx->VSSetShaderResources(0, 1, &nullSrv);
        ctx->PSSetShaderResources(0, 1, &nullSrv);
    }
    saved.Restore(ctx);
}

void hud::Reset() noexcept
{
    Rel(g_blend);
    Rel(g_cb);
    Rel(g_instSRV);
    Rel(g_instBuf);
    Rel(g_ps);
    Rel(g_vs);
    Rel(g_dev);
    g_failed = false;
    g_n = 0;
}
