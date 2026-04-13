#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <vector>
#include <string>
#include <memory>
#include "HeliosOutput.h"

// -----------------------------------------------------------------------------
// HEDInference
//
// Runs HED (Holistically-nested Edge Detection) via ONNX Runtime on BGR frames.
// Outputs edge contours as normalized LaserPoint polylines ready for SendFrame().
//
// Expected model: models/hed/model.onnx
//   Input:  "data"  [1, 3, H, W]  float32, BGR, mean-subtracted
//   Output: "fused" [1, 1, H, W]  float32, sigmoid edge probability [0..1]
// -----------------------------------------------------------------------------

class HEDInference {
public:
    enum class Backend { CPU, GPU };

    HEDInference();
    ~HEDInference();

    bool Initialize(Backend backend = Backend::CPU);
    bool LoadModel(const std::wstring& modelPath);
    bool IsLoaded() const;
    const char* GetBackendName() const;

    // Main pipeline: BGR frame -> edge contours -> laser polylines.
    // edgeThreshold:     sigmoid probability cutoff (0.3-0.6 typical)
    // minContourPoints:  discard contours shorter than this (filters noise)
    // smoothEpsilon:     polyDP approximation tolerance in pixels (0 = off)
    // sampleColor:       sample BGR color from source frame at each point
    std::vector<std::vector<LaserPoint>> ProcessFrame(
        const cv::Mat& bgrFrame,
        float edgeThreshold    = 0.4f,
        int   minContourPoints = 15,
        float smoothEpsilon    = 1.5f,
        bool  sampleColor      = true,
        float temporalAlpha    = 1.0f);  // 1.0 = no blending, 0.5 = heavy smoothing

private:
    // Run HED ONNX inference; returns float32 edge probability map [H, W].
    cv::Mat RunInference(const cv::Mat& bgrFrame);

    // Threshold edge map and extract contours as laser polylines.
    std::vector<std::vector<LaserPoint>> ExtractContours(
        const cv::Mat& edgeMap,
        const cv::Mat& srcFrame,
        float threshold,
        int   minContourPoints,
        float smoothEpsilon,
        bool  sampleColor);

    // ONNX Runtime
    std::unique_ptr<Ort::Env>            env_;
    std::unique_ptr<Ort::SessionOptions> sessionOptions_;
    std::unique_ptr<Ort::Session>        session_;

    std::vector<const char*> inputNames_;
    std::vector<const char*> outputNames_;

    Backend backend_  = Backend::CPU;
    bool    isLoaded_ = false;

    // Model runs at this resolution internally; output is resized to match
    // the incoming frame before contour extraction.
    int inputW_ = 480;
    int inputH_ = 480;

    // HED BGR channel means (from the original Caffe training).
    static constexpr float kMeanB = 104.00698793f;
    static constexpr float kMeanG = 116.66876762f;
    static constexpr float kMeanR = 122.67891434f;
};
