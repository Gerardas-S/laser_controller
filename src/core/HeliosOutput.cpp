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

    // Estimate transition overhead
    int overhead = 0;
    HeliosPoint center{};
    center.x = 2048; center.y = 2048;
    for (size_t i = 0; i < scaled.size(); ++i) {
        if (scaled[i].empty()) continue;
        HeliosPoint from = (i == 0) ? center : scaled[i - 1].back();
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

    std::vector<std::vector<HeliosPoint>> sorted;
    sorted.reserve(polylines.size());

    std::vector<bool> visited(polylines.size(), false);

    // Start from center
    HeliosPoint current{};
    current.x = 2048;
    current.y = 2048;

    for (size_t n = 0; n < polylines.size(); ++n)
    {
        float bestDist = std::numeric_limits<float>::max();
        int   bestIdx = -1;
        bool  bestReversed = false;

        for (size_t i = 0; i < polylines.size(); ++i)
        {
            if (visited[i]) continue;

            float dStart = Distance(current, polylines[i].front());
            float dEnd = Distance(current, polylines[i].back());

            if (dStart < bestDist) {
                bestDist = dStart;
                bestIdx = (int)i;
                bestReversed = false;
            }
            if (dEnd < bestDist) {
                bestDist = dEnd;
                bestIdx = (int)i;
                bestReversed = true;
            }
        }

        if (bestIdx < 0) break;

        visited[bestIdx] = true;
        if (bestReversed) {
            std::reverse(polylines[bestIdx].begin(), polylines[bestIdx].end());
        }
        current = polylines[bestIdx].back();
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
    float ax = (float)(b.x - a.x);
    float ay = (float)(b.y - a.y);
    float bx = (float)(c.x - b.x);
    float by = (float)(c.y - b.y);

    float lenA = std::sqrt(ax * ax + ay * ay);
    float lenB = std::sqrt(bx * bx + by * by);

    if (lenA < 1.0f || lenB < 1.0f) return config_.min_vertex_hold;

    float dot = (ax * bx + ay * by) / (lenA * lenB);
    dot = std::clamp(dot, -1.0f, 1.0f);
    float angleDeg = std::acos(dot) * 180.0f / 3.14159265f;

    // below curve threshold — galvo handles it via momentum, no dwell needed
    if (angleDeg < config_.curve_threshold)
        return 0;

    // above threshold — scale proportionally between min and max
    float t = (angleDeg - config_.curve_threshold) /
        (180.0f - config_.curve_threshold);
    int   dwell = (int)(t * config_.max_vertex_hold);
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

        // Record where the galvo will be after this frame completes so the
        // next BuildFrame call can start its travel from the right position.
        lastEndX_.store((int)currentFrame.back().x, std::memory_order_relaxed);
        lastEndY_.store((int)currentFrame.back().y, std::memory_order_relaxed);

        int pps = CalculatePPS((int)currentFrame.size());
        helios_.WriteFrame(0, pps, HELIOS_FLAGS_SINGLE_MODE,
                           currentFrame.data(), (uint32_t)currentFrame.size());
    }
}