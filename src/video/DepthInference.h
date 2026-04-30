#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <vector>
#include <string>
#include <memory>
#include "HeliosOutput.h"

// -----------------------------------------------------------------------------
// DepthInference
//
// Runs Depth Anything V2 (Small) via ONNX Runtime on BGR frames.
// Produces a monocular depth map, then extracts laser-drawable lines via:
//
//   Mode::Edges    — gradient of depth map  → fires on object boundaries
//   Mode::Isolines — threshold depth at N levels → topographic contour rings
//   Mode::Both     — both layers combined
//
// Expected model: models/depth/model.onnx
//   Input:  "pixel_values"    [1, 3, 518, 518]  float32, RGB, ImageNet-normalized
//   Output: "predicted_depth" [1, H, W]          float32, unnormalized depth
// -----------------------------------------------------------------------------

class DepthInference {
public:
    enum class Backend { CPU, GPU };

    enum class Mode {
        Edges,     // depth gradient edges only
        Isolines,  // topographic isolines only
        Both       // both combined
    };

    DepthInference();
    ~DepthInference();

    bool Initialize(Backend backend = Backend::CPU);
    bool LoadModel(const std::wstring& modelPath);
    bool IsLoaded() const;
    const char* GetBackendName() const;

    // Main pipeline: BGR frame -> depth map -> laser polylines.
    //
    // edgeThreshold      gradient magnitude cutoff for Edges mode  (0.05–0.3 typical)
    // isolineCount       number of evenly-spaced depth levels      (4–10 typical)
    // minContourPoints   discard contours shorter than this
    // smoothEpsilon      polyDP approximation in pixels (0 = off)
    std::vector<std::vector<LaserPoint>> ProcessFrame(
        const cv::Mat& bgrFrame,
        Mode  mode             = Mode::Both,
        float edgeThreshold    = 0.1f,
        int   isolineCount     = 6,
        int   minContourPoints = 10,
        float smoothEpsilon    = 2.0f);

    // Expose the last normalised depth map for debugging / visualisation.
    cv::Mat LastDepthMap() const { return lastDepth_; }

private:
    // Run inference; returns unnormalised float depth map at frame resolution.
    cv::Mat RunInference(const cv::Mat& bgrFrame);

    // Extract depth-gradient edges as laser polylines.
    std::vector<std::vector<LaserPoint>> ExtractEdges(
        const cv::Mat& depthNorm,
        float threshold,
        int   minPoints,
        float smoothEpsilon);

    // Extract isoline contours as laser polylines.
    std::vector<std::vector<LaserPoint>> ExtractIsolines(
        const cv::Mat& depthNorm,
        int   count,
        int   minPoints,
        float smoothEpsilon);

    // Helper: find contours in a binary 8-bit image, convert to LaserPoint polys.
    std::vector<std::vector<LaserPoint>> ContoursToPolylines(
        const cv::Mat& binary,
        int   frameW,
        int   frameH,
        int   minPoints,
        float smoothEpsilon);

    // ONNX Runtime
    std::unique_ptr<Ort::Env>            env_;
    std::unique_ptr<Ort::SessionOptions> sessionOptions_;
    std::unique_ptr<Ort::Session>        session_;

    std::vector<const char*> inputNames_;
    std::vector<const char*> outputNames_;

    Backend backend_  = Backend::CPU;
    bool    isLoaded_ = false;

    // Depth Anything V2 Small fixed input size.
    int inputW_ = 518;
    int inputH_ = 518;

    // ImageNet RGB normalisation: (x/255 - mean) / std
    static constexpr float kMeanR = 0.485f;
    static constexpr float kMeanG = 0.456f;
    static constexpr float kMeanB = 0.406f;
    static constexpr float kStdR  = 0.229f;
    static constexpr float kStdG  = 0.224f;
    static constexpr float kStdB  = 0.225f;

    // Last normalised [0,1] depth map (for LastDepthMap()).
    cv::Mat lastDepth_;
};
