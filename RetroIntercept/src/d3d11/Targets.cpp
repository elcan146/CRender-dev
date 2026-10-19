#include "Targets.h"

#include <Windows.h>
#include <cmath>
#include <cstring>

namespace
{

constexpr int  kStrideMin  = 12;
constexpr int  kStrideMax  = 64;
constexpr int  kStrides    = (kStrideMax - kStrideMin) / 4 + 1; // 14 candidates
constexpr int  kAabbSlots  = 8192;                              // open addressing, power of 2
constexpr int  kProbe      = 16;
constexpr UINT kMaxScanBytes = 16u * 1024u * 1024u;             // bound one-time load cost
constexpr int  kWorldSlots = 512;
constexpr int  kMaxCand    = 64;

struct AabbRec
{
    ID3D11Buffer* buf;
    unsigned      validMask;       // bit i = stride kStrideMin + 4*i has a usable box
    float         box[kStrides][6]; // min xyz, max xyz
};

struct WorldRec
{
    ID3D11Buffer* buf;
    float         m[16];
};

AabbRec  g_aabb[kAabbSlots] = {};
SRWLOCK  g_aabbLock = SRWLOCK_INIT;
WorldRec g_world[kWorldSlots] = {};

ID3D11Buffer* g_camBuf = nullptr;
float         g_proj[16] = {};
float         g_view[16] = {};
bool          g_camValid = false;

term::Det     g_cand[kMaxCand] = {};
int           g_candN = 0;
term::Tracker g_tracker;

size_t Home(const void* p, size_t mask) noexcept
{
    UINT_PTR v = reinterpret_cast<UINT_PTR>(p) >> 4;
    v ^= v >> 13;
    return static_cast<size_t>(v) & mask;
}

bool Finite(float v) noexcept
{
    return v == v && v < 1e6f && v > -1e6f;
}

void ComputeBoxes(const unsigned char* data, UINT bytes, AabbRec& rec) noexcept
{
    rec.validMask = 0;
    if (bytes > kMaxScanBytes)
        bytes = kMaxScanBytes;
    for (int si = 0; si < kStrides; ++si)
    {
        const UINT stride = static_cast<UINT>(kStrideMin + 4 * si);
        const UINT n = bytes / stride;
        if (n < 3)
            continue;
        float mn[3] = { 1e30f, 1e30f, 1e30f }, mx[3] = { -1e30f, -1e30f, -1e30f };
        UINT good = 0;
        for (UINT i = 0; i < n; ++i)
        {
            float p[3];
            memcpy(p, data + static_cast<size_t>(i) * stride, sizeof(p));
            if (!Finite(p[0]) || !Finite(p[1]) || !Finite(p[2]))
                continue;
            ++good;
            for (int a = 0; a < 3; ++a)
            {
                if (p[a] < mn[a]) mn[a] = p[a];
                if (p[a] > mx[a]) mx[a] = p[a];
            }
        }
        // a wrong stride reads normals/uvs/garbage as positions: demand mostly-finite
        // data and a non-degenerate extent; the bind-time stride picks the real one.
        if (good * 4 < n * 3 || (mx[0] - mn[0]) + (mx[1] - mn[1]) + (mx[2] - mn[2]) <= 1e-5f)
            continue;
        for (int a = 0; a < 3; ++a)
        {
            rec.box[si][a] = mn[a];
            rec.box[si][3 + a] = mx[a];
        }
        rec.validMask |= 1u << si;
    }
}

// row-major, row-vector: out = v * M (ConsoleWire convention, verified on Heaven)
void MulMV(const float v[4], const float m[16], float out[4]) noexcept
{
    for (int c = 0; c < 4; ++c)
        out[c] = v[0] * m[c] + v[1] * m[4 + c] + v[2] * m[8 + c] + v[3] * m[12 + c];
}

bool LooksOrtho(const float* m) noexcept
{
    const float m00 = fabsf(m[0]), m11 = fabsf(m[5]);
    return m00 > 1e-6f && m00 < 0.05f && m11 > 1e-6f && m11 < 0.05f &&
           fabsf(m[1]) < 1e-4f && fabsf(m[2]) < 1e-4f &&
           fabsf(m[4]) < 1e-4f && fabsf(m[6]) < 1e-4f &&
           fabsf(m[15] - 1.0f) < 0.01f;
}

float Iou(const term::Det& a, const term::Det& b) noexcept
{
    const float ix0 = a.x0 > b.x0 ? a.x0 : b.x0, iy0 = a.y0 > b.y0 ? a.y0 : b.y0;
    const float ix1 = a.x1 < b.x1 ? a.x1 : b.x1, iy1 = a.y1 < b.y1 ? a.y1 : b.y1;
    if (ix1 <= ix0 || iy1 <= iy0)
        return 0.0f;
    const float inter = (ix1 - ix0) * (iy1 - iy0);
    const float u = (a.x1 - a.x0) * (a.y1 - a.y0) + (b.x1 - b.x0) * (b.y1 - b.y0) - inter;
    return u > 0.0f ? inter / u : 0.0f;
}

void AddCandidate(const term::Det& d) noexcept
{
    int smallest = -1;
    for (int i = 0; i < g_candN; ++i)
    {
        if (Iou(d, g_cand[i]) > 0.8f) // same mesh drawn by another pass
            return;
        if (smallest < 0 || g_cand[i].score < g_cand[smallest].score)
            smallest = i;
    }
    if (g_candN < kMaxCand)
        g_cand[g_candN++] = d;
    else if (smallest >= 0 && g_cand[smallest].score < d.score)
        g_cand[smallest] = d;
}

} // namespace

void targets::OnCreateBuffer(const D3D11_BUFFER_DESC* desc, const D3D11_SUBRESOURCE_DATA* init,
                             ID3D11Buffer* buf) noexcept
{
    if (desc == nullptr || buf == nullptr)
        return;
    const bool isVB = (desc->BindFlags & D3D11_BIND_VERTEX_BUFFER) != 0;
    const bool hasData = init != nullptr && init->pSysMem != nullptr && desc->ByteWidth >= 36;
    AabbRec rec = {};
    rec.buf = buf;
    if (isVB && hasData)
        ComputeBoxes(static_cast<const unsigned char*>(init->pSysMem), desc->ByteWidth, rec);

    // Always overwrite a recycled pointer's stale entry; only store new ones with data.
    AcquireSRWLockExclusive(&g_aabbLock);
    const size_t mask = kAabbSlots - 1;
    const size_t h = Home(buf, mask);
    size_t slot = h;
    bool found = false;
    for (int i = 0; i < kProbe; ++i)
    {
        const size_t s = (h + static_cast<size_t>(i)) & mask;
        if (g_aabb[s].buf == buf) { slot = s; found = true; break; }
    }
    if (!found && rec.validMask != 0)
    {
        for (int i = 0; i < kProbe; ++i)
        {
            const size_t s = (h + static_cast<size_t>(i)) & mask;
            if (g_aabb[s].buf == nullptr) { slot = s; break; }
        }
        found = true; // table crowded: evict the home slot
    }
    if (found)
        g_aabb[slot] = rec.validMask != 0 ? rec : AabbRec{};
    ReleaseSRWLockExclusive(&g_aabbLock);
}

void targets::OnUpdateCB(ID3D11Buffer* buf, UINT byteWidth, const void* data) noexcept
{
    if (buf == nullptr || data == nullptr)
        return;
    if (byteWidth == 64)
    {
        WorldRec& w = g_world[Home(buf, kWorldSlots - 1)];
        w.buf = buf;
        memcpy(w.m, data, sizeof(w.m));
    }
    else if (byteWidth == 144)
    {
        const float* f = static_cast<const float*>(data);
        memcpy(g_proj, f, sizeof(g_proj));
        memcpy(g_view, f + 16, sizeof(g_view));
        g_camBuf = buf;
        // perspective: w = -z (m[11] = -1) or +z — reject ortho/garbage
        g_camValid = Finite(g_proj[0]) && Finite(g_proj[5]) && fabsf(g_proj[11]) > 0.5f;
    }
}

void targets::OnDraw(ID3D11Buffer* vb, UINT stride, UINT vbOffset, ID3D11Buffer* vsCB0) noexcept
{
    if (!g_camValid || vb == nullptr || stride < static_cast<UINT>(kStrideMin) ||
        stride > static_cast<UINT>(kStrideMax) || (stride & 3) != 0 || (vbOffset % stride) != 0)
        return;
    const int si = static_cast<int>(stride - kStrideMin) / 4;

    float box[6];
    bool have = false;
    AcquireSRWLockShared(&g_aabbLock);
    const size_t mask = kAabbSlots - 1;
    const size_t h = Home(vb, mask);
    for (int i = 0; i < kProbe; ++i)
    {
        const AabbRec& r = g_aabb[(h + static_cast<size_t>(i)) & mask];
        if (r.buf == vb)
        {
            if ((r.validMask & (1u << si)) != 0)
            {
                memcpy(box, r.box[si], sizeof(box));
                have = true;
            }
            break;
        }
    }
    ReleaseSRWLockShared(&g_aabbLock);
    if (!have)
        return;

    static const float kIdentity[16] = { 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 };
    const float* world = nullptr;
    if (vsCB0 == g_camBuf)
        world = kIdentity; // positions are already world-space
    else if (vsCB0 != nullptr)
    {
        const WorldRec& w = g_world[Home(vsCB0, kWorldSlots - 1)];
        if (w.buf == vsCB0)
            world = w.m;
    }
    if (world == nullptr || LooksOrtho(world))
        return; // unknown transform or UI quad

    float nx0 = 1e30f, ny0 = 1e30f, nx1 = -1e30f, ny1 = -1e30f;
    for (int c = 0; c < 8; ++c)
    {
        const float v[4] = { box[(c & 1) ? 3 : 0], box[(c & 2) ? 4 : 1], box[(c & 4) ? 5 : 2], 1.0f };
        float wv[4], vv[4], cl[4];
        MulMV(v, world, wv);
        MulMV(wv, g_view, vv);
        MulMV(vv, g_proj, cl);
        if (!(cl[3] > 0.05f))
            return; // crosses the near plane — box would be unbounded
        const float x = cl[0] / cl[3], y = cl[1] / cl[3];
        if (x < nx0) nx0 = x;
        if (x > nx1) nx1 = x;
        if (y < ny0) ny0 = y;
        if (y > ny1) ny1 = y;
    }
    term::Det d;
    d.x0 = (nx0 + 1.0f) * 0.5f;
    d.x1 = (nx1 + 1.0f) * 0.5f;
    d.y0 = (1.0f - ny1) * 0.5f;
    d.y1 = (1.0f - ny0) * 0.5f;
    if (d.x1 <= 0.0f || d.y1 <= 0.0f || d.x0 >= 1.0f || d.y0 >= 1.0f)
        return; // off screen
    if (d.x0 < 0.0f) d.x0 = 0.0f;
    if (d.y0 < 0.0f) d.y0 = 0.0f;
    if (d.x1 > 1.0f) d.x1 = 1.0f;
    if (d.y1 > 1.0f) d.y1 = 1.0f;
    const float bw = d.x1 - d.x0, bh = d.y1 - d.y0;
    const float area = bw * bh;
    if (bw < 0.02f || bh < 0.02f || area < 0.002f || area > 0.30f)
        return; // specks, terrain, sky domes
    d.score = area;
    AddCandidate(d);
}

const term::Track* targets::EndFrame(float dt, int maxTargets, int* count) noexcept
{
    g_tracker.Update(g_cand, g_candN, dt, maxTargets);
    g_candN = 0;
    if (count != nullptr)
        *count = g_tracker.Count();
    return g_tracker.Tracks();
}

void targets::Reset() noexcept
{
    g_tracker.Clear();
    g_candN = 0;
    g_camValid = false;
    g_camBuf = nullptr;
    memset(g_world, 0, sizeof(g_world));
}
