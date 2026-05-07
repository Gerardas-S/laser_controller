#define NOMINMAX
#include "HeliosOutput.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <chrono>

// -----------------------------------------------------------------------------
// Construction / destruction
// -----------------------------------------------------------------------------

HeliosOutput::HeliosOutput() {}

HeliosOutput::~HeliosOutput() {
    Close();
}

// -----------------------------------------------------------------------------
// Public interface
// -----------------------------------------------------------------------------

bool HeliosOutput::Initialize(HeliosConfig config)
{
    config_ = config;
    running_ = true;
    dacThread_ = std::thread(&HeliosOutput::DacThreadFunc, this);
    return Connect();
}

bool HeliosOutput::Connect()
{
    if (connected_) return true;
    int count = helios_.OpenDevices();
    if (count < 1) {
        std::cerr << "HeliosOutput: No DAC found\n";
        return false;
    }
    std::cout << "HeliosOutput: Found " << count << " DAC(s)\n";
    connected_ = true;
    return true;
}

void HeliosOutput::Disconnect()
{
    connected_ = false;
    helios_.CloseDevices();
    std::cout << "HeliosOutput: DAC disconnected\n";
}

void HeliosOutput::SetConfig(const HeliosConfig& config) {
    config_ = config;
}

void HeliosOutput::Close()
{
    running_ = false;
    queueCV_.notify_all();
    if (dacThread_.joinable())
        dacThread_.join();
    if (connected_)
        Disconnect();
}

bool HeliosOutput::IsConnected() const {
    return connected_;
}

void HeliosOutput::SendPolyline(const std::vector<LaserPoint>& points)
{
    SendFrame({ points });
}

void HeliosOutput::SendFrame(const std::vector<std::vector<LaserPoint>>& polylines)
{
    if (!connected_ || polylines.empty()) return;

    // Scale to ILDA
    std::vector<std::vector<HeliosPoint>> scaled;
    scaled.reserve(polylines.size());
    for (const auto& poly : polylines)
        scaled.push_back(ScaleToILDA(poly));

    // Optional reorder
    if (config_.enable_reorder)
        ReorderPath(scaled);

    // --- Budget distribution ---
    int totalBudget = config_.max_pps / config_.target_fps;

    // Estimate transition overhead — seed from actual tracked galvo position
    // so budget allocation is accurate even when the galvo isn't at centre.
    int overhead = 0;
    HeliosPoint trackedPos{};
    trackedPos.x = (uint16_t)lastEndX_.load(std::memory_order_relaxed);
    trackedPos.y = (uint16_t)lastEndY_.load(std::memory_order_relaxed);
    trackedPos.r = trackedPos.g = trackedPos.b = trackedPos.i = 0;
    for (size_t i = 0; i < scaled.size(); ++i) {
        if (scaled[i].empty()) continue;
        HeliosPoint from = (i == 0) ? trackedPos : scaled[i - 1].back();
        overhead += EstimateTransitionCost(from, scaled[i].front());
    }

    int drawBudget = std::max(totalBudget - overhead,
        (int)scaled.size() * 2);

    // Compute lengths
    std::vector<float> lengths(scaled.size());
    float totalLength = 0.0f;
    for (size_t i = 0; i < scaled.size(); ++i) {
        lengths[i] = PolylineLength(scaled[i]);
        totalLength += lengths[i];
    }

    // Resample each polyline to its budget share
    for (size_t i = 0; i < scaled.size(); ++i) {
        if (scaled[i].size() < 2) continue;
        int allocated = (totalLength > 0.0f)
            ? (int)(drawBudget * (lengths[i] / totalLength))
            : (drawBudget / (int)scaled.size());
        allocated = std::max(allocated, 2);
        scaled[i] = ResampleToCount(scaled[i], allocated);
    }

    // Build and send
    auto frame = BuildFrame(scaled);
    if (frame.empty()) return;

    {
        std::lock_guard<std::mutex> lock(queueMutex_);
        while (frameQueue_.size() > 2)
            frameQueue_.pop();
        frameQueue_.push(std::move(frame));
    }
    queueCV_.notify_one();
}

void HeliosOutput::SendPhysical(const std::vector<HeliosPoint>& points)
{
    if (!connected_ || points.size() < 2) return;

    // Already-baked: every point in here is a point the DAC will play.
    // Just queue.  Phase correction happens in DacThreadFunc.
    std::vector<HeliosPoint> frame = points;

    {
        std::lock_guard<std::mutex> lock(queueMutex_);
        while (frameQueue_.size() > 2)
            frameQueue_.pop();
        frameQueue_.push(std::move(frame));
    }
    queueCV_.notify_one();
}

void HeliosOutput::SendPointCloud(
    const std::vector<LaserPoint>& points, float gapThreshold)
{
    if (points.empty()) return;

    std::vector<std::vector<LaserPoint>> polylines;
    std::vector<LaserPoint> current;
    current.push_back(points[0]);

    for (size_t i = 1; i < points.size(); ++i) {
        float dx = points[i].x - points[i - 1].x;
        float dy = points[i].y - points[i - 1].y;
        float dist = std::sqrt(dx * dx + dy * dy);

        if (dist > gapThreshold) {
            // gap detected — end current polyline, start new one
            if (!current.empty())
                polylines.push_back(std::move(current));
            current.clear();
        }
        current.push_back(points[i]);
    }

    if (!current.empty())
        polylines.push_back(std::move(current));

    SendFrame(polylines);
}

// -----------------------------------------------------------------------------
// BuildFrame — core assembly
// -----------------------------------------------------------------------------

std::vector<HeliosPoint> HeliosOutput::BuildFrame(
    const std::vector<std::vector<HeliosPoint>>& polylines)
{
    std::vector<HeliosPoint> frame;
    frame.reserve(512);

    // Start from where the galvo actually is (tracked by DacThreadFunc).
    // On the very first frame this is the centre (2048, 2048).  On every
    // subsequent frame it is the last point of the previously sent frame,
    // so repeated frames (e.g. the alignment circle) produce zero travel
    // overhead at the join — eliminating the per-frame lurch that caused
    // galvo chirping at the frame repeat boundary.
    HeliosPoint currentPos{};
    currentPos.x = (uint16_t)lastEndX_.load(std::memory_order_relaxed);
    currentPos.y = (uint16_t)lastEndY_.load(std::memory_order_relaxed);
    currentPos.r = currentPos.g = currentPos.b = currentPos.i = 0;

    for (size_t pi = 0; pi < polylines.size(); ++pi)
    {
        const auto& poly = polylines[pi];
        if (poly.empty()) continue;

        const HeliosPoint& polyStart = poly.front();
        const HeliosPoint& polyEnd = poly.back();

        // 1. Blanked eased move from current position to this polyline's start
        int travelPts = CalcTravelPoints(currentPos, polyStart);
        InsertEasedMove(frame, currentPos, polyStart, travelPts);

        // 2. Blank dwell at start — galvo settles, laser still off
        InsertBlankAtPoint(frame, polyStart, config_.blank_points);

        // 3. Pre-on dwell — laser turns on, galvo already at position
        InsertDwellAtPoint(frame, polyStart, config_.pre_on_points);

        // After ScaleToILDA, before sending to DAC thread:
        // 1. Fill gaps in each polyline
        // 2. Insert corner dwell proportionally

        for (size_t i = 0; i < poly.size(); ++i)
        {
            frame.push_back(poly[i]);

            if (i > 0 && i < poly.size() - 1)
            {
                int dwell = CalcCornerDwell(poly[i - 1], poly[i], poly[i + 1]);
                if (dwell > 0)
                    InsertDwellAtPoint(frame, poly[i], dwell);
            }
        }

        // 5. Post-on dwell at end — laser stays on briefly
        InsertDwellAtPoint(frame, polyEnd, config_.post_on_points);

        // 6. Blank dwell at end — laser turns off before moving
        InsertBlankAtPoint(frame, polyEnd, config_.blank_points);

        // Update current position
        currentPos = polyEnd;
        currentPos.r = currentPos.g = currentPos.b = currentPos.i = 0;
    }

    return frame;
}

// -----------------------------------------------------------------------------
// ScaleToILDA
// -----------------------------------------------------------------------------

std::vector<HeliosPoint> HeliosOutput::ScaleToILDA(
    const std::vector<LaserPoint>& points)
{
    std::vector<HeliosPoint> out;
    out.reserve(points.size());
    for (const auto& p : points) {
        HeliosPoint hp;
        hp.x = (uint16_t)std::clamp((int)((p.x + 1.0f) * 0.5f * 4095), 0, 4095);
        hp.y = (uint16_t)std::clamp((int)((p.y + 1.0f) * 0.5f * 4095), 0, 4095);
        hp.r = (uint8_t)std::clamp((int)(p.r * 255), 0, 255);
        hp.g = (uint8_t)std::clamp((int)(p.g * 255), 0, 255);
        hp.b = (uint8_t)std::clamp((int)(p.b * 255), 0, 255);
        hp.i = (uint8_t)std::max(hp.r, std::max(hp.g, hp.b));
        out.push_back(hp);
    }
    return out;
}

// -----------------------------------------------------------------------------
// ReorderPath — nearest neighbor sort on polylines
// -----------------------------------------------------------------------------

void HeliosOutput::ReorderPath(std::vector<std::vector<HeliosPoint>>& polylines)
{
    if (polylines.size() < 2) return;

    const size_t N = polylines.size();
    const float  W = config_.reorder_angle_weight;  // ILDA units per (1 - cos θ)

    // ── Pre-compute exit directions for forward and reversed traversal ────────
    // fwd_ex/ey: unit direction of the LAST step when traversed forward
    // rev_ex/ey: unit direction of the LAST step when traversed reversed
    //            = negated unit direction of the FIRST step traversed forward
    struct ExitDir { float fwd_ex, fwd_ey, rev_ex, rev_ey; };
    std::vector<ExitDir> exits(N);

    for (size_t i = 0; i < N; ++i)
    {
        const auto& p = polylines[i];
        auto unit = [](float dx, float dy, float fb_ex, float fb_ey)
            -> std::pair<float, float>
        {
            float len = std::sqrt(dx * dx + dy * dy);
            return (len > 0.5f) ? std::make_pair(dx / len, dy / len)
                                : std::make_pair(fb_ex,     fb_ey);
        };

        if (p.size() >= 2) {
            // forward exit: last step direction
            auto [fex, fey] = unit((float)(p.back().x    - p[p.size()-2].x),
                                   (float)(p.back().y    - p[p.size()-2].y),  1.f, 0.f);
            // reversed exit: opposite of the forward entry (first step direction reversed)
            auto [rex, rey] = unit((float)(p.front().x   - p[1].x),
                                   (float)(p.front().y   - p[1].y), -1.f, 0.f);
            exits[i] = { fex, fey, rex, rey };
        }
        else {
            exits[i] = { 1.f, 0.f, -1.f, 0.f };
        }
    }

    // ── Nearest-neighbour TSP with angle penalty ──────────────────────────────
    //
    // cost(cur → seg_endpoint) = Euclidean distance
    //                           + W · (1 − dot(cur_dir, travel_dir))
    //
    // The second term penalises blank jumps that require the galvo to reverse
    // direction: going straight ahead costs 0 extra; a full reversal costs 2·W.
    // W = 100 means a 180° reversal adds 200 ILDA-unit equivalent penalty,
    // roughly the cost of a small additional travel segment.

    std::vector<std::vector<HeliosPoint>> sorted;
    sorted.reserve(N);
    std::vector<bool> visited(N, false);

    // Seed from actual tracked galvo position so reorder reflects real state
    HeliosPoint cur{};
    cur.x = (uint16_t)lastEndX_.load(std::memory_order_relaxed);
    cur.y = (uint16_t)lastEndY_.load(std::memory_order_relaxed);
    float cur_dx = 1.0f, cur_dy = 0.0f;  // unknown initial direction → no penalty on first pick

    for (size_t n = 0; n < N; ++n)
    {
        float bestCost    = std::numeric_limits<float>::max();
        int   bestIdx     = -1;
        bool  bestReversed = false;

        for (size_t i = 0; i < N; ++i)
        {
            if (visited[i]) continue;

            // Helper: cost from cur to a candidate endpoint
            auto segCost = [&](const HeliosPoint& endpoint,
                               bool firstSeg) -> float
            {
                float dist  = Distance(cur, endpoint);
                if (firstSeg || dist < 0.5f)
                    return dist;   // no angle penalty on the very first pick

                float tdx = (float)(endpoint.x - cur.x);
                float tdy = (float)(endpoint.y - cur.y);
                float tlen = std::sqrt(tdx * tdx + tdy * tdy);
                if (tlen < 0.5f)
                    return dist;

                float ndx  = tdx / tlen;
                float ndy  = tdy / tlen;
                // (1 − cos θ) is 0 when cur_dir and travel_dir are aligned, 2 when opposed
                float penalty = W * (1.0f - (cur_dx * ndx + cur_dy * ndy));
                return dist + penalty;
            };

            bool  isFirst = (n == 0);
            float costFwd = segCost(polylines[i].front(), isFirst);
            float costRev = segCost(polylines[i].back(),  isFirst);

            if (costFwd < bestCost) { bestCost = costFwd; bestIdx = (int)i; bestReversed = false; }
            if (costRev < bestCost) { bestCost = costRev; bestIdx = (int)i; bestReversed = true;  }
        }

        if (bestIdx < 0) break;

        visited[bestIdx] = true;
        if (bestReversed) {
            std::reverse(polylines[bestIdx].begin(), polylines[bestIdx].end());
            cur_dx = exits[bestIdx].rev_ex;
            cur_dy = exits[bestIdx].rev_ey;
        }
        else {
            cur_dx = exits[bestIdx].fwd_ex;
            cur_dy = exits[bestIdx].fwd_ey;
        }
        cur = polylines[bestIdx].back();
        sorted.push_back(std::move(polylines[bestIdx]));
    }

    polylines = std::move(sorted);
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

void HeliosOutput::InsertEasedMove(
    std::vector<HeliosPoint>& frame,
    HeliosPoint from, HeliosPoint to,
    int numPoints)
{
    if (numPoints <= 0) return;
    for (int i = 0; i < numPoints; ++i) {
        float t = (float)i / (float)(numPoints - 1);
        float ease = QuintEaseInOut(t);
        HeliosPoint p;
        p.x = (uint16_t)(from.x + (to.x - from.x) * ease);
        p.y = (uint16_t)(from.y + (to.y - from.y) * ease);
        p.r = p.g = p.b = p.i = 0;
        frame.push_back(p);
    }
}

void HeliosOutput::InsertBlankAtPoint(
    std::vector<HeliosPoint>& frame,
    HeliosPoint p, int count)
{
    p.r = p.g = p.b = p.i = 0;
    for (int i = 0; i < count; ++i)
        frame.push_back(p);
}

void HeliosOutput::InsertDwellAtPoint(
    std::vector<HeliosPoint>& frame,
    HeliosPoint p, int count)
{
    // color preserved from the point — laser stays on
    for (int i = 0; i < count; ++i)
        frame.push_back(p);
}

int HeliosOutput::CalcCornerDwell(HeliosPoint a, HeliosPoint b, HeliosPoint c)
{
    // Segment vectors A→B and B→C
    float ax = (float)(b.x - a.x),  ay = (float)(b.y - a.y);
    float bx = (float)(c.x - b.x),  by = (float)(c.y - b.y);

    float lenA = std::sqrt(ax * ax + ay * ay);
    float lenB = std::sqrt(bx * bx + by * by);
    if (lenA < 1.0f || lenB < 1.0f) return config_.min_vertex_hold;

    // ── Scale-independent angle check (curve_threshold acts as a smoothness gate)
    float dot = std::clamp((ax * bx + ay * by) / (lenA * lenB), -1.0f, 1.0f);
    float angleDeg = std::acos(dot) * (180.0f / 3.14159265f);
    if (angleDeg < config_.curve_threshold)
        return 0;

    // ── Menger curvature  κ = 2·|cross(AB,BC)| / (|AB|·|BC|·|AC|)
    //
    // Unlike the old angle-only formula, κ also accounts for segment length:
    // the same turn angle on a SHORT segment yields higher κ (the galvo must
    // change direction faster), so it correctly demands more dwell time.
    //
    // For equal-length segments of length L: κ = 2·sin(θ/2) / L
    //   • straight line  (θ = 0°)  → κ = 0
    //   • 90° corner     (θ = 90°) → κ = √2 / L  ≈ 1.414 / L
    //   • hairpin        (θ = 180°)→ κ = 2 / L
    float cross_z = ax * by - ay * bx;        // 2-D cross product (z-component)
    float lenAC   = Distance(a, c);

    float kappa;
    if (lenAC > 0.5f) {
        kappa = 2.0f * std::abs(cross_z) / (lenA * lenB * lenAC);
    }
    else {
        // Degenerate hairpin: A and C nearly coincide → κ → 2 / max(segment)
        kappa = 2.0f / std::max(lenA, lenB);
    }

    // Scale to a point count and clamp to configured range
    int dwell = (int)(kappa * config_.kappa_scale);
    return std::clamp(dwell, config_.min_vertex_hold, config_.max_vertex_hold);
}

float HeliosOutput::Distance(HeliosPoint a, HeliosPoint b)
{
    float dx = (float)(a.x - b.x);
    float dy = (float)(a.y - b.y);
    return std::sqrt(dx * dx + dy * dy);
}

float HeliosOutput::QuintEaseInOut(float t)
{
    if (t < 0.5f) {
        return 16.0f * t * t * t * t * t;
    }
    else {
        float f = (2.0f * t) - 2.0f;
        return 0.5f * f * f * f * f * f + 1.0f;
    }
}

int HeliosOutput::CalcTravelPoints(HeliosPoint from, HeliosPoint to)
{
    float dist = Distance(from, to);
    int pts = (int)(dist / config_.move_speed);
    return std::clamp(pts, config_.min_travel_points, config_.max_travel_points);
}

int HeliosOutput::CalculatePPS(int numPoints)
{
    int pps = numPoints * config_.target_fps;
    return std::clamp(pps, config_.min_pps, config_.max_pps);
}

float HeliosOutput::PolylineLength(const std::vector<HeliosPoint>& poly)
{
    float len = 0.0f;
    for (size_t i = 1; i < poly.size(); ++i)
        len += Distance(poly[i - 1], poly[i]);
    return len;
}

int HeliosOutput::EstimateTransitionCost(HeliosPoint from, HeliosPoint to)
{
    return config_.blank_points
        + config_.pre_on_points
        + config_.post_on_points
        + CalcTravelPoints(from, to);
}

std::vector<HeliosPoint> HeliosOutput::ResampleToCount(
    const std::vector<HeliosPoint>& poly, int targetCount)
{
    if (poly.size() < 2 || targetCount < 2) return poly;

    // cumulative arc lengths
    std::vector<float> cumLen(poly.size(), 0.0f);
    for (size_t i = 1; i < poly.size(); ++i)
        cumLen[i] = cumLen[i - 1] + Distance(poly[i - 1], poly[i]);

    float totalLength = cumLen.back();
    if (totalLength < 0.001f) return poly;

    std::vector<HeliosPoint> out;
    out.reserve(targetCount);

    float  interval = totalLength / (targetCount - 1);
    size_t seg = 0;

    for (int i = 0; i < targetCount; ++i)
    {
        float targetLen = interval * i;

        while (seg < poly.size() - 2 && cumLen[seg + 1] < targetLen)
            ++seg;

        float segLen = cumLen[seg + 1] - cumLen[seg];
        float t = (segLen < 0.001f) ? 0.0f :
            (targetLen - cumLen[seg]) / segLen;
        t = std::clamp(t, 0.0f, 1.0f);

        HeliosPoint p;
        p.x = (uint16_t)(poly[seg].x + (poly[seg + 1].x - poly[seg].x) * t);
        p.y = (uint16_t)(poly[seg].y + (poly[seg + 1].y - poly[seg].y) * t);
        p.r = (uint8_t)(poly[seg].r + (poly[seg + 1].r - poly[seg].r) * t);
        p.g = (uint8_t)(poly[seg].g + (poly[seg + 1].g - poly[seg].g) * t);
        p.b = (uint8_t)(poly[seg].b + (poly[seg + 1].b - poly[seg].b) * t);
        p.i = (uint8_t)std::max(p.r, std::max(p.g, p.b));
        out.push_back(p);
    }

    return out;
}


// -----------------------------------------------------------------------------
// ApplyPhaseCorrection
// -----------------------------------------------------------------------------

void HeliosOutput::ApplyPhaseCorrection(std::vector<HeliosPoint>& frame)
{
    // The Y galvo servo typically has a slightly different phase response to X.
    // This manifests as diagonal lines bowing: the Y position lags behind the
    // commanded value relative to X by a fraction of a point interval.
    //
    // Correction: advance Y by xy_phase_shift samples using linear interpolation.
    // Y_corrected[n] = lerp(Y_input[n + floor(shift)], Y_input[n + ceil(shift)], frac)
    //
    // A shift of 0.3 at 30 kpps corresponds to a 10 µs Y lag — typical for
    // modest-bandwidth galvo drivers.  Tune by projecting a 45° diagonal and
    // adjusting until the line appears straight at full scan speed.

    const float shift = config_.xy_phase_shift;
    if (shift <= 0.0f || frame.size() < 2)
        return;

    const size_t n   = frame.size();
    const int    lo  = (int)shift;           // floor of shift
    const float  frac = shift - (float)lo;   // fractional remainder

    // Snapshot the original Y values before overwriting
    std::vector<uint16_t> y0(n);
    for (size_t i = 0; i < n; ++i)
        y0[i] = frame[i].y;

    for (size_t i = 0; i < n; ++i)
    {
        // Source indices, clamped to frame boundary
        size_t j = std::min(i + (size_t)lo,     n - 1);
        size_t k = std::min(i + (size_t)lo + 1, n - 1);
        frame[i].y = (uint16_t)((float)y0[j] + ((float)y0[k] - (float)y0[j]) * frac);
    }
}

// -----------------------------------------------------------------------------
// DAC thread
// -----------------------------------------------------------------------------

void HeliosOutput::DacThreadFunc()
{
    std::vector<HeliosPoint> currentFrame;

    while (running_)
    {
        // If not connected, wait — don't spin on USB errors.
        if (!connected_) {
            // Drain stale queue so we don't play old frames on reconnect.
            { std::lock_guard<std::mutex> lk(queueMutex_);
              while (!frameQueue_.empty()) frameQueue_.pop(); }
            currentFrame.clear();
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }

        // Poll until DAC is ready.
        int status = helios_.GetStatus(0);
        if (status < 0) {
            std::cerr << "HeliosOutput: DAC error " << status << " — going offline\n";
            connected_ = false;
            continue;
        }
        if (status != 1) continue; // not ready yet, poll again

        // Grab latest frame (re-send last if nothing new)
        {
            std::lock_guard<std::mutex> lock(queueMutex_);
            if (!frameQueue_.empty()) {
                currentFrame = std::move(frameQueue_.front());
                frameQueue_.pop();
            }
        }

        if (currentFrame.empty()) continue;

        // Apply Y-axis phase correction before the frame goes to hardware.
        // This is the last processing step — done here so the correction is
        // always applied regardless of which Send* path built the frame.
        ApplyPhaseCorrection(currentFrame);

        // Record where the galvo will be after this frame completes so the
        // next BuildFrame call can start its travel from the right position.
        lastEndX_.store((int)currentFrame.back().x, std::memory_order_relaxed);
        lastEndY_.store((int)currentFrame.back().y, std::memory_order_relaxed);

        int pps = CalculatePPS((int)currentFrame.size());
        helios_.WriteFrame(0, pps, HELIOS_FLAGS_SINGLE_MODE,
                           currentFrame.data(), (uint32_t)currentFrame.size());
    }
}