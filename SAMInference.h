#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <vector>
#include <string>
#include <memory>
#include "HeliosOutput.h"

// -----------------------------------------------------------------------------
// SAMInference
//
// Runs MobileSAM (Segment Anything Model) via ONNX Runtime on BGR frames.
// Uses automatic grid-based point prompts — no user interaction required.
// Outputs segment boundary contours as normalized LaserPoint polylines.
//
// Expected models:
//   models/sam/encoder.onnx   — image encoder (ViT)
//     Input:  "image"            [1, 3, 1024, 1024]  float32, normalized
//     Output: "image_embeddings" [1, 256, 64, 64]    float32
//
//   models/sam/decoder.onnx   — mask decoder
//     Inputs:  image_embeddings  [1, 256, 64, 64]
//              point_coords      [1, N, 2]            float32
//              point_labels      [1, N]               float32
//              mask_input        [1, 1, 256, 256]     float32
//              has_mask_input    [1]                  float32
//              orig_im_size      [2]                  float32
//     Outputs: masks             [1, 3, H, W]         float32
//              iou_predictions   [1, 3]               float32
//              low_res_masks     [1, 3, 256, 256]     float32
// -----------------------------------------------------------------------------

class SAMInference {
public:
    enum class Backend { CPU, GPU };

    SAMInference();
    ~SAMInference();

    bool Initialize(Backend backend = Backend::CPU);
    bool LoadModels(const std::wstring& encoderPath,
                    const std::wstring& decoderPath);
    bool IsLoaded() const;
    const char* GetBackendName() const;

    // Main pipeline: BGR frame -> segment contours -> laser polylines.
    // gridSize:          NxN grid of point prompts (8 = 64 segments attempted)
    // iouThreshold:      discard masks with predicted IoU below this (0.7-0.9)
    // minContourPoints:  discard contours shorter than this (filters noise)
    // smoothEpsilon:     polyDP approximation tolerance in pixels (0 = off)
    // sampleColor:       sample BGR color from source frame at each point
    std::vector<std::vector<LaserPoint>> ProcessFrame(
        const cv::Mat& bgrFrame,
        int   gridSize         = 8,
        float iouThreshold     = 0.75f,
        int   minContourPoints = 20,
        float smoothEpsilon    = 1.5f,
        bool  sampleColor      = true);

private:
    // Run encoder on frame; returns image embedding.
    cv::Mat RunEncoder(const cv::Mat& bgrFrame);

    // Run decoder for a single point prompt; returns best mask.
    cv::Mat RunDecoder(const cv::Mat& imageEmbedding,
                       float px, float py,
                       float origW, float origH);

    // Extract contour of a binary mask as laser polyline.
    std::vector<std::vector<LaserPoint>> ExtractContours(
        const cv::Mat& mask,
        const cv::Mat& srcFrame,
        float origW, float origH,
        int   minContourPoints,
        float smoothEpsilon,
        bool  sampleColor);

    // ONNX Runtime — two sessions
    std::unique_ptr<Ort::Env>            env_;
    std::unique_ptr<Ort::SessionOptions> encoderOptions_;
    std::unique_ptr<Ort::SessionOptions> decoderOptions_;
    std::unique_ptr<Ort::Session>        encoderSession_;
    std::unique_ptr<Ort::Session>        decoderSession_;

    std::vector<const char*> encInputNames_;
    std::vector<const char*> encOutputNames_;
    std::vector<const char*> decInputNames_;
    std::vector<const char*> decOutputNames_;

    Backend backend_   = Backend::CPU;
    bool    isLoaded_  = false;

    // SAM encoder input resolution
    static constexpr int kEncSize = 1024;

    // ImageNet mean/std for ViT encoder normalisation (RGB order)
    static constexpr float kMeanR = 123.675f / 255.0f;
    static constexpr float kMeanG = 116.280f / 255.0f;
    static constexpr float kMeanB = 103.530f / 255.0f;
    static constexpr float kStdR  =  58.395f / 255.0f;
    static constexpr float kStdG  =  57.120f / 255.0f;
    static constexpr float kStdB  =  57.375f / 255.0f;
};
