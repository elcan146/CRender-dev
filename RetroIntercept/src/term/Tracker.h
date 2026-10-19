// Tracker — stable IDs + EMA smoothing for per-frame target boxes (normalized 0..1 coords).
// Game mode feeds projected mesh bounds; stream mode feeds motion/detector boxes.
// Not thread-safe: one instance per producer thread.
#pragma once

namespace term
{

struct Det
{
    float x0, y0, x1, y1; // normalized, x0<x1, y0<y1
    float score;          // larger = more important (kept first when over budget)
    int   cls = -1;       // detector class id; -1 = unlabeled (mesh AABB scans)
};

struct Track
{
    float x0, y0, x1, y1; // smoothed box
    int   id;             // stable, monotonically assigned
    float age;            // seconds since first seen
    float unseen;         // seconds since last matched detection (coasting)
    float lockAge;        // seconds since lock (valid when locked)
    bool  locked;         // age passed the lock threshold
    int   cls = -1;       // last matched detection's class (-1 = unlabeled)
    float score = 0.0f;   // last matched detection's score
};

class Tracker
{
public:
    static constexpr int kMax = 32;

    // Matches dets to tracks (IoU, then centroid distance), smooths, ages, expires.
    // maxTracks caps the live set (best-scoring dets win). dt in seconds.
    void Update(const Det* dets, int n, float dt, int maxTracks) noexcept;
    void Clear() noexcept { m_n = 0; }

    int          Count() const noexcept { return m_n; }
    const Track* Tracks() const noexcept { return m_tracks; }

private:
    Track m_tracks[kMax] = {};
    int   m_n = 0;
    int   m_nextId = 1;
};

} // namespace term
