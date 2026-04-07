#pragma once

#include <vector>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
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
    float curve_threshold = 20.0f;
    bool  enable_reorder = false;
};

// -----------------------------------------------------------------------------
// HeliosOutput
// -----------------------------------------------------------------------------

class HeliosOutput {
public:
    HeliosOutput();
    ~HeliosOutput();

    bool Initialize(HeliosConfig config = {});
    void Close();
    bool IsConnected() const;
    void SetConfig(const HeliosConfig& config);

    // 1. You know your structure — multiple explicit polylines
    void SendFrame(const std::vector<std::vector<LaserPoint>>& polylines);

    // 2. You have one continuous stroke
    void SendPolyline(const std::vector<LaserPoint>& points);

    // 3. You have a flat bag of points with no structure
    void SendPointCloud(const std::vector<LaserPoint>& points, float gapThreshold = 0.1f);

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

    // --- DAC thread ---
    void DacThreadFunc();

    // --- state ---
    HeliosDac    helios_;
    HeliosConfig config_;
    bool         running_ = false;
    bool         connected_ = false;

    // --- threading ---
    std::thread                          dacThread_;
    std::queue<std::vector<HeliosPoint>> frameQueue_;
    std::mutex                           queueMutex_;
    std::condition_variable              queueCV_;
};