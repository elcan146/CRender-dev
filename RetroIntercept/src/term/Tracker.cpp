#include "Tracker.h"

namespace
{

constexpr float kIouMatch   = 0.25f; // primary association
constexpr float kDistMatch  = 0.08f; // fallback: centroid distance (normalized)
constexpr float kEma        = 0.35f; // weight of the new detection
constexpr float kCoast      = 0.35f; // seconds a track survives without detections
constexpr float kLockAfter  = 1.2f;  // seconds tracked before "LOCK"

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
        float bestIou = kIouMatch;
        for (int t = 0; t < m_n; ++t)
        {
            if (trackUsed[t])
                continue;
            const float iou = Iou(d, m_tracks[t]);
            if (iou > bestIou) { bestIou = iou; best = t; }
        }
        if (best < 0)
        {
            float bestD = kDistMatch * kDistMatch;
            for (int t = 0; t < m_n; ++t)
            {
                if (trackUsed[t])
                    continue;
                const float d2 = Dist2(d, m_tracks[t]);
                if (d2 < bestD) { bestD = d2; best = t; }
            }
        }
        if (best >= 0)
        {
            Track& tr = m_tracks[best];
            tr.x0 += (d.x0 - tr.x0) * kEma; tr.y0 += (d.y0 - tr.y0) * kEma;
            tr.x1 += (d.x1 - tr.x1) * kEma; tr.y1 += (d.y1 - tr.y1) * kEma;
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
        if (!tr.locked && tr.age >= kLockAfter)
        {
            tr.locked = true;
            tr.lockAge = 0.0f;
        }
        else if (tr.locked)
            tr.lockAge += dt;
        if (tr.unseen <= kCoast && w < maxTracks)
            m_tracks[w++] = tr;
    }
    m_n = w;
}
