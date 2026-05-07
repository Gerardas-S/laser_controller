#pragma once

#include <vector>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <atomic>
#include <cstdint>
#include "libs/helios/HeliosDac.h"

// -----------------------------------------------------------------------------
// Public input type — normalized coordinates
// -----------------------------------------------------------------------------

struct LaserPoint {
    float x, y;    // -1.0 to 1.0
    float r, g, b; // 0.0 to 1.0
};

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------

struct HeliosConfig {
    int   target_fps = 60;
    int   max_pps = 30000;
    int   min_pps = 7;
    int   blank_points = 20;
    int   pre_on_points = 8;
    int   post_on_points = 8;
    int   min_travel_points = 8;
    int   max_travel_points = 80;
    float move_speed = 100.0f;
    float max_point_distance = 50.0f;  // kept for SendPointCloud gap detection
    int   min_vertex_hold = 2;
    int   max_vertex_hold = 20;
    float curve_threshold = 20.0f;     // degrees — bends below this get zero dwell
    bool  enable_reorder = false;

    // ── Phase 1 optimisations ────────────────────────────────────────────────

    // Menger-curvature dwell scale.
    // CalcCornerDwell computes κ = 2·|cross| / (|AB|·|BC|·|AC|) (ILDA units⁻¹)
    // then dwell = κ × kappa_scale, clamped to [min_vertex_hold, max_vertex_hold].
    // Increase to dwell longer at sharp corners; decrease to tighten timing.
    float kappa_scale = 500.0f;

    // Angle penalty weight for TSP path reorder (ILDA units per radian ×2).
    // A blank-travel that requires a 180° galvo direction reversal costs an
    // extra  reorder_angle_weight  ILDA units on top of its Euclidean distance.
    // Only active when enable_reorder = true.
    float reorder_angle_weight = 100.0f;

    // Y-axis phase correction (fractional sample count, typically 0.0–1.0).
    // The Y galvo servo often lags X by a fraction of a point interval; this
    // advances Y by xy_phase_shift samples via linear interpolation so that
    // diagonal lines remain straight.  Set to 0 to disable.
    float xy_phase_shift = 0.0f;
};

// -----------------------------------------------------------------------------
// HeliosOutput
// -----------------------------------------------------------------------------

class HeliosOutput {
public:
    HeliosOutput();
    ~HeliosOutput();

    // Start the DAC thread with the given config. Does not require a DAC to be
    // present — returns false if no DAC found but the program can keep running.
    bool Initialize(HeliosConfig config = {});
    void Close();

    // Attempt to open DAC devices. Safe to call while running.
    bool Connect();

    // Close DAC devices. Output silently stops; Connect() can re-open later.
    void Disconnect();

    bool IsConnected() const;
    void SetConfig(const HeliosConfig& config);

    // 1. You know your structure — multiple explicit polylines.
    //    Runtime-generated content (alignment circle, etc.) goes through here:
    //    HeliosOutput synthesises blanking, dwells, and eased travel via
    //    BuildFrame.  Use this when you DON'T have a pre-baked .ild file.
    void SendFrame(const std::vector<std::vector<LaserPoint>>& polylines);

    // 2. You have one continuous stroke
    void SendPolyline(const std::vector<LaserPoint>& points);

    // 3. You have a flat bag of points with no structure
    void SendPointCloud(const std::vector<LaserPoint>& points, float gapThreshold = 0.1f);

    // 4. You already have a physically-baked point sequence (12-bit ILDA
    //    coords, blanking and dwells already inserted at encode time).
    //    Used for .ild files produced by encode.py.  Goes straight into the
    //    DAC queue with no further geometric processing — only hardware
    //    compensation (xy_phase_shift) is applied in DacThreadFunc.
    void SendPhysical(const std::vector<HeliosPoint>& points);

private:
    // --- coordinate conversion ---
    std::vector<HeliosPoint> ScaleToILDA(const std::vector<LaserPoint>& points);

    // --- shape ordering ---
    void ReorderPath(std::vector<std::vector<HeliosPoint>>& polylines);

    // --- frame assembly ---
    std::vector<HeliosPoint> BuildFrame(
        const std::vector<std::vector<HeliosPoint>>& polylines);

    // --- helpers ---
    void InsertEasedMove(std::vector<HeliosPoint>& frame,
        HeliosPoint from, HeliosPoint to,
        int numPoints);
    void InsertBlankAtPoint(std::vector<HeliosPoint>& frame,
        HeliosPoint p, int count);
    void InsertDwellAtPoint(std::vector<HeliosPoint>& frame,
        HeliosPoint p, int count);
    float Distance(HeliosPoint a, HeliosPoint b);
    float QuintEaseInOut(float t);
    int   CalcTravelPoints(HeliosPoint from, HeliosPoint to);
    int   CalculatePPS(int numPoints);
    std::vector<HeliosPoint> ResampleToCount(const std::vector<HeliosPoint>& poly,
        int targetCount);
    float PolylineLength(const std::vector<HeliosPoint>& poly);
    int   EstimateTransitionCost(HeliosPoint from, HeliosPoint to);
    int   CalcCornerDwell(HeliosPoint a, HeliosPoint b, HeliosPoint c);
    void  ApplyPhaseCorrection(std::vector<HeliosPoint>& frame);

    // --- DAC thread ---
    void DacThreadFunc();

    // --- state ---
    HeliosDac    helios_;
    HeliosConfig config_;
    bool         running_ = false;
    bool         connected_ = false;

    // Last known galvo position (ILDA coords 0-4095).
    // Updated by DacThreadFunc after each frame is sent so BuildFrame can
    // begin its travel calculation from where the galvo actually is rather
    // than always assuming it starts at the centre.  Using atomics means no
    // additional locking is required in BuildFrame.
    std::atomic<int> lastEndX_{ 2048 };
    std::atomic<int> lastEndY_{ 2048 };

    // --- threading ---
    std::thread                          dacThread_;
    std::queue<std::vector<HeliosPoint>> frameQueue_;
    std::mutex                           queueMutex_;
    std::condition_variable              queueCV_;
};