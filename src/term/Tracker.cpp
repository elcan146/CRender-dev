#include "Tracker.h"

namespace
{

float g_iouMatch  = 0.25f; // primary association
float g_distMatch = 0.08f; // fallback: centroid distance (normalized)
float g_ema       = 0.35f; // weight of the new detection
float g_coast     = 0.35f; // seconds a track survives without detections
float g_lockAfter = 1.2f;  // seconds tracked before "LOCK"

float Clamp(float v, float lo, float hi) noexcept
{
    return v < lo ? lo : (v > hi ? hi : v);
}

float Iou(const term::Det& a, const term::Track& b) noexcept
{
    const float ix0 = a.x0 > b.x0 ? a.x0 : b.x0, iy0 = a.y0 > b.y0 ? a.y0 : b.y0;
    const float ix1 = a.x1 < b.x1 ? a.x1 : b.x1, iy1 = a.y1 < b.y1 ? a.y1 : b.y1;
    if (ix1 <= ix0 || iy1 <= iy0)
        return 0.0f;
    const float inter = (ix1 - ix0) * (iy1 - iy0);
    const float ua = (a.x1 - a.x0) * (a.y1 - a.y0) + (b.x1 - b.x0) * (b.y1 - b.y0) - inter;
    return ua > 0.0f ? inter / ua : 0.0f;
}

float Dist2(const term::Det& a, const term::Track& b) noexcept
{
    const float dx = (a.x0 + a.x1 - b.x0 - b.x1) * 0.5f;
    const float dy = (a.y0 + a.y1 - b.y0 - b.y1) * 0.5f;
    return dx * dx + dy * dy;
}

} // namespace

void term::Tracker::Update(const Det* dets, int n, float dt, int maxTracks) noexcept
{
    if (maxTracks > kMax) maxTracks = kMax;
    if (maxTracks < 0)    maxTracks = 0;
    if (n > 64)           n = 64;
    if (dets == nullptr)  n = 0;
    if (dt < 0.0f) dt = 0.0f;
    if (dt > 1.0f) dt = 1.0f; // long gap (mode switch, hitch): unmatched tracks expire

    // Best-score-first order (n <= 64: selection sort on indices is fine).
    int order[64];
    for (int i = 0; i < n; ++i)
        order[i] = i;
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            if (dets[order[j]].score > dets[order[i]].score)
            {
                const int t = order[i]; order[i] = order[j]; order[j] = t;
            }

    bool trackUsed[kMax] = {};
    for (int oi = 0; oi < n; ++oi)
    {
        const Det& d = dets[order[oi]];
        int best = -1;
        float bestIou = g_iouMatch;
        for (int t = 0; t < m_n; ++t)
        {
            // Same-cls only: sess-1 classes (kSess1Base+) share geometry with
            // sess-0 boxes, so an overlapping gun det would steal a person
            // track and flip tr.cls every frame without this gate.
            if (trackUsed[t] || m_tracks[t].cls != d.cls)
                continue;
            const float iou = Iou(d, m_tracks[t]);
            if (iou > bestIou) { bestIou = iou; best = t; }
        }
        if (best < 0)
        {
            float bestD = g_distMatch * g_distMatch;
            for (int t = 0; t < m_n; ++t)
            {
                if (trackUsed[t] || m_tracks[t].cls != d.cls)
                    continue;
                const float d2 = Dist2(d, m_tracks[t]);
                if (d2 < bestD) { bestD = d2; best = t; }
            }
        }
        if (best >= 0)
        {
            Track& tr = m_tracks[best];
            tr.x0 += (d.x0 - tr.x0) * g_ema; tr.y0 += (d.y0 - tr.y0) * g_ema;
            tr.x1 += (d.x1 - tr.x1) * g_ema; tr.y1 += (d.y1 - tr.y1) * g_ema;
            tr.unseen = 0.0f;
            tr.cls = d.cls;
            tr.score = d.score;
            ++tr.hits;
            tr.ang = d.ang;
            trackUsed[best] = true;
        }
        else if (m_n < maxTracks)
        {
            Track& tr = m_tracks[m_n];
            tr = { d.x0, d.y0, d.x1, d.y1, m_nextId++, 0.0f, 0.0f, 0.0f, false, d.cls, d.score, d.ang, 1 };
            if (m_nextId > 99) m_nextId = 1; // "TGT nn" display range
            trackUsed[m_n] = true;
            ++m_n;
        }
    }

    // Age, lock, coast, expire (compact in place).
    int w = 0;
    for (int t = 0; t < m_n; ++t)
    {
        Track tr = m_tracks[t];
        tr.age += dt;
        if (!trackUsed[t])
            tr.unseen += dt;
        if (!tr.locked && tr.age >= g_lockAfter)
        {
            tr.locked = true;
            tr.lockAge = 0.0f;
        }
        else if (tr.locked)
            tr.lockAge += dt;
        if (tr.unseen <= g_coast && w < maxTracks)
            m_tracks[w++] = tr;
    }
    m_n = w;
}

void term::SetTuning(const Tuning& t) noexcept
{
    g_iouMatch  = Clamp(t.iouMatch,  0.05f, 0.95f);
    g_distMatch = Clamp(t.distMatch, 0.01f, 0.50f);
    g_ema       = Clamp(t.ema,       0.05f, 1.00f);
    g_coast     = Clamp(t.coast,     0.0f,  2.00f);
    g_lockAfter = Clamp(t.lockAfter, 0.0f, 30.0f);
}
