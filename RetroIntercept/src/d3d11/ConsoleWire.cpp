// ConsoleWire — CPU ASCII wireframe output (secondary render path).
//
// hkDrawIndexed (ConsoleWire mode) submits budgeted VB/IB snapshots; hkPresent
// calls EndFrame which projects verts (tracked slot-0 proj CB when available,
// else a fixed ortho), Bresenham-rasterizes triangle edges into a char grid,
// and blits to an AllocConsole window at ConsoleHz.
#include "ConsoleWire.h"
#include "../core/Log.h"

#include <Windows.h>
#include <vector>
#include <cmath>
#include <cstring>

namespace
{

// ---- tunables (bound to the externs below) ------------------------------------
int  s_maxDraws = 4;
int  s_w = 120;
int  s_h = 40;
int  s_hz = 30;

// ---- collected frame geometry --------------------------------------------------
struct Vert { float x, y, z; };
std::vector<Vert>   s_verts;   // positions only
std::vector<UINT>   s_idx;     // triangle indices into s_verts
int                 s_submits = 0;

constexpr size_t kMaxVerts = 32768;
constexpr size_t kMaxIdx   = 49152; // ~16k tris/frame ceiling

// ---- console --------------------------------------------------------------------
HANDLE g_con = nullptr;
UINT   g_conW = 0, g_conH = 0;
std::vector<wchar_t> s_grid;

void OpenConsole() noexcept
{
    if (g_con != nullptr)
        return;
    if (AllocConsole() == 0 && GetConsoleWindow() == nullptr)
        return;
    g_con = GetStdHandle(STD_OUTPUT_HANDLE);
    if (g_con == nullptr || g_con == INVALID_HANDLE_VALUE)
    {
        g_con = nullptr;
        return;
    }
    // size buffer first (must be >= window), then the window rect
    COORD sz = { static_cast<SHORT>(s_w), static_cast<SHORT>(s_h + 1) };
    SetConsoleScreenBufferSize(g_con, sz);
    SMALL_RECT rc = { 0, 0, static_cast<SHORT>(s_w - 1), static_cast<SHORT>(s_h - 1) };
    if (g_con != nullptr)
    {
        HWND cw = GetConsoleWindow();
        if (cw != nullptr)
            SetWindowPos(cw, nullptr, 40, 40, s_w * 8 + 16, s_h * 16 + 40, SWP_NOZORDER);
        SetConsoleWindowInfo(g_con, TRUE, &rc);
        SetConsoleTitleA("RetroIntercept — console wireframe");
    }
    g_conW = static_cast<UINT>(s_w);
    g_conH = static_cast<UINT>(s_h);
    s_grid.assign(g_conW * g_conH, L' ');
    Log::Info("wire", "console open %ux%u @%dHz", g_conW, g_conH, s_hz);
}

// ---- raster ----------------------------------------------------------------------

void SetPx(int x, int y, wchar_t c) noexcept
{
    if (x >= 0 && y >= 0 && x < static_cast<int>(g_conW) && y < static_cast<int>(g_conH))
        s_grid[static_cast<size_t>(y) * g_conW + x] = c;
}

// Box-drawing char for a segment by slope: horiz/vert/diag.
// (y-down screen space: rising screen row = negative y delta)
wchar_t EdgeChar(int x0, int y0, int x1, int y1) noexcept
{
    const int dx = abs(x1 - x0), dy = abs(y1 - y0);
    if (dy * 4 <= dx)      return 0x2500; // ─
    if (dx * 4 <= dy)      return 0x2502; // │
    const bool same = ((x1 - x0) > 0) == ((y1 - y0) > 0);
    return same ? 0x2572 : 0x2571;        // \ or / (╲╱)
}

// Bresenham into s_grid.
void Line(int x0, int y0, int x1, int y1, wchar_t c) noexcept
{
    const int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    const int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    for (;;)
    {
        if (x0 >= 0 && x0 < s_w && y0 >= 0 && y0 < s_h)
            s_grid[y0 * s_w + x0] = c;
        if (x0 == x1 && y0 == y1)
            break;
        const int e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

// row-major 4x4 multiply: out = v * M
void MulMV(const float v[4], const float m[16], float out[4]) noexcept
{
    for (int c = 0; c < 4; ++c)
        out[c] = v[0] * m[c] + v[1] * m[4 + c] + v[2] * m[8 + c] + v[3] * m[12 + c];
}

// simple staging readback for the proj CB (mirror of HookManager's, kept local)
void* ReadCB(ID3D11DeviceContext* ctx, ID3D11Device* dev, ID3D11Buffer* buf, UINT* bytes) noexcept
{
    if (ctx == nullptr || dev == nullptr || buf == nullptr)
        return nullptr;
    D3D11_BUFFER_DESC bd = {};
    buf->GetDesc(&bd);
    if (bd.ByteWidth < 64 || bd.ByteWidth > 2048)
        return nullptr;
    D3D11_BUFFER_DESC sd = bd;
    sd.Usage = D3D11_USAGE_STAGING;
    sd.BindFlags = 0;
    sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    sd.MiscFlags = 0;
    sd.StructureByteStride = 0;
    ID3D11Buffer* staging = nullptr;
    if (FAILED(dev->CreateBuffer(&sd, nullptr, &staging)) || staging == nullptr)
        return nullptr;
    ctx->CopyResource(staging, buf);
    D3D11_MAPPED_SUBRESOURCE m = {};
    void* out = nullptr;
    if (SUCCEEDED(ctx->Map(staging, 0, D3D11_MAP_READ, 0, &m)) && m.pData != nullptr)
    {
        out = HeapAlloc(GetProcessHeap(), 0, bd.ByteWidth);
        if (out != nullptr)
        {
            memcpy(out, m.pData, bd.ByteWidth);
            *bytes = bd.ByteWidth;
        }
        ctx->Unmap(staging, 0);
    }
    staging->Release();
    return out;
}

} // namespace

int wire::g_maxDrawsPerFrame = 4;
int wire::g_consoleW = 120;
int wire::g_consoleH = 40;
int wire::g_consoleHz = 30;
int wire::g_wireCull = 1;

void wire::EnsureConsole() noexcept
{
    s_w  = g_consoleW;
    s_h  = g_consoleH;
    s_hz = g_consoleHz;
    OpenConsole();
}

void wire::BeginFrame() noexcept
{
    s_submits = 0;
    s_verts.clear();
    s_idx.clear();
}

bool wire::WantsMore() noexcept
{
    return s_submits < g_maxDrawsPerFrame;
}

void wire::Submit(const void* verts, UINT vbBytes, UINT stride,
                  const void* indices, UINT startIndex, UINT indexCount, INT baseVertex,
                  DXGI_FORMAT indexFmt, const float* world) noexcept
{
    if (verts == nullptr || indices == nullptr || stride < 12 || indexCount == 0)
        return;
    if (s_verts.size() >= kMaxVerts || s_idx.size() >= kMaxIdx)
        return;

    const size_t vbase = s_verts.size();
    const UINT nverts = vbBytes / stride;
    const char* vb = static_cast<const char*>(verts);
    const size_t take = (nverts < kMaxVerts - vbase) ? nverts : (kMaxVerts - vbase);
    for (UINT i = 0; i < take; ++i)
    {
        const float* p = reinterpret_cast<const float*>(vb + i * stride);
        Vert v = { p[0], p[1], p[2] };
        if (world != nullptr)
        {
            // v' = v * world (row-major; translation in row 3)
            Vert w;
            w.x = v.x * world[0] + v.y * world[4] + v.z * world[8]  + world[12];
            w.y = v.x * world[1] + v.y * world[5] + v.z * world[9]  + world[13];
            w.z = v.x * world[2] + v.y * world[6] + v.z * world[10] + world[14];
            v = w;
        }
        s_verts.push_back(v);
    }

    const bool is16 = (indexFmt == DXGI_FORMAT_R16_UINT);
    const UINT takeIdx = (indexCount < kMaxIdx - s_idx.size()) ? indexCount
                                                              : static_cast<UINT>(kMaxIdx - s_idx.size());
    for (UINT i = 0; i < takeIdx; ++i)
    {
        const UINT j = startIndex + i;
        const UINT raw = is16 ? static_cast<const unsigned short*>(indices)[j]
                              : static_cast<const UINT*>(indices)[j];
        const INT adjusted = static_cast<INT>(raw) + baseVertex;
        if (adjusted >= 0 && adjusted < static_cast<INT>(take))
            s_idx.push_back(static_cast<UINT>(adjusted + vbase));
    }
    ++s_submits;
}

void wire::EndFrame(ID3D11DeviceContext* ctx, ID3D11Device* dev,
                    ID3D11Buffer* projCB) noexcept
{
    if (g_con == nullptr || s_verts.empty() || s_idx.empty())
        return;

    static ULONGLONG last = 0;
    const ULONGLONG now = GetTickCount64();
    if (s_hz > 0 && now - last < static_cast<ULONGLONG>(1000 / s_hz))
        return;
    last = now;

    // transforms from the tracked 144B slot-0 CB: proj at bytes 0-63,
    // view at bytes 64-127 (scanner-found, Phase 9). Fallback keeps
    // model-space verts on screen with a fixed ortho-ish scale.
    float proj[16] = {};
    float view[16] = {};
    bool haveVP = false;
    if (projCB != nullptr)
    {
        UINT cbBytes = 0;
        void* cb = ReadCB(ctx, dev, projCB, &cbBytes);
        if (cb != nullptr && cbBytes >= 128)
        {
            memcpy(proj, cb, 64);
            memcpy(view, static_cast<const char*>(cb) + 64, 64);
            haveVP = true;
        }
        if (cb != nullptr)
            HeapFree(GetProcessHeap(), 0, cb);
    }
    if (!haveVP)
    {
        memset(proj, 0, sizeof(proj));
        proj[0] = 0.08f; proj[5] = 0.08f; proj[10] = 0.001f; proj[14] = 0.5f; proj[15] = 1.0f;
        memset(view, 0, sizeof(view));
        view[0] = view[5] = view[10] = view[15] = 1.0f; // identity
    }

    memset(s_grid.data(), L' ', s_grid.size());

    const size_t ntri = s_idx.size() / 3;
    for (size_t t = 0; t < ntri; ++t)
    {
        const UINT ia = s_idx[t * 3], ib = s_idx[t * 3 + 1], ic = s_idx[t * 3 + 2];
        if (ia >= s_verts.size() || ib >= s_verts.size() || ic >= s_verts.size())
            continue;
        int px[3], py[3];
        float zv[3];
        bool ok = true;
        const UINT ids[3] = { ia, ib, ic };
        for (int v = 0; v < 3; ++v)
        {
            const Vert& vt = s_verts[ids[v]];
            const float hv[4] = { vt.x, vt.y, vt.z, 1.0f };
            float vw[4], cl[4];
            MulMV(hv, view, vw);          // world -> view
            MulMV(vw, proj, cl);          // view  -> clip
            if (cl[3] < 0.01f) { ok = false; break; }  // behind camera
            const float ndcX = cl[0] / cl[3];
            const float ndcY = cl[1] / cl[3];
            px[v] = static_cast<int>((ndcX * 0.5f + 0.5f) * static_cast<float>(g_conW));
            py[v] = static_cast<int>((0.5f - ndcY * 0.5f) * static_cast<float>(g_conH));
            zv[v] = cl[2] / cl[3];        // ndc depth for fade
        }
        if (!ok)
            continue;
        // backface cull: signed 2D area (screen space, y down). D3D default
        // front = CW in NDC -> CCW on screen -> negative signed area.
        if (wire::g_wireCull != 0)
        {
            const int area = (px[1]-px[0])*(py[2]-py[0]) - (px[2]-px[0])*(py[1]-py[0]);
            if (area > 0)
                continue;
        }
        const float z = (zv[0] + zv[1] + zv[2]) * (1.0f / 3.0f);
        // depth cue: near edges = proper line glyphs, mid = '+', far = '.'
        for (int e = 0; e < 3; ++e)
        {
            const int a = e, b = (e + 1) % 3;
            const wchar_t c = (z < 0.4f) ? EdgeChar(px[a], py[a], px[b], py[b])
                                       : (z < 0.8f ? L'+' : L'.');
            Line(px[a], py[a], px[b], py[b], c);
        }
    }

    COORD origin = { 0, 0 };
    DWORD written = 0;
    SetConsoleCursorPosition(g_con, origin);
    WriteConsoleW(g_con, s_grid.data(), static_cast<DWORD>(s_grid.size()), &written, nullptr);
}
