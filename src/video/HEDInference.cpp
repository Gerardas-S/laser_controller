#include "HEDInference.h"
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cstring>

HEDInference::HEDInference() = default;

HEDInference::~HEDInference()
{
    for (const char* n : inputNames_)  free(const_cast<char*>(n));
    for (const char* n : outputNames_) free(const_cast<char*>(n));
}

// -----------------------------------------------------------------------------
// Initialize
// -----------------------------------------------------------------------------

bool HEDInference::Initialize(Backend backend)
{
    backend_ = backend;

    try {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "HEDInference");
        sessionOptions_ = std::make_unique<Ort::SessionOptions>();

        sessionOptions_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        if (backend_ == Backend::GPU) {
            std::cout << "[HED] Initializing with GPU (CUDA) backend\n";
            OrtCUDAProviderOptions cuda{};
            cuda.device_id = 0;
            sessionOptions_->AppendExecutionProvider_CUDA(cuda);
        } else {
            std::cout << "[HED] Initializing with CPU backend\n";
            sessionOptions_->SetIntraOpNumThreads(4);
        }
        return true;
    } catch (const Ort::Exception& e) {
        std::cerr << "[HED] Init failed: " << e.what() << "\n";
        return false;
    }
}

// -----------------------------------------------------------------------------
// LoadModel
// -----------------------------------------------------------------------------

bool HEDInference::LoadModel(const std::wstring& modelPath)
{
    if (!env_ || !sessionOptions_) {
        std::cerr << "[HED] Call Initialize() before LoadModel()\n";
        return false;
    }

    try {
        std::wcout << L"[HED] Loading model: " << modelPath << L"\n";
        session_ = std::make_unique<Ort::Session>(*env_, modelPath.c_str(), *sessionOptions_);

        Ort::AllocatorWithDefaultOptions alloc;

        size_t nIn = session_->GetInputCount();
        std::cout << "[HED] Model inputs (" << nIn << "):\n";
        for (size_t i = 0; i < nIn; ++i) {
            auto name = session_->GetInputNameAllocated(i, alloc);
            inputNames_.push_back(_strdup(name.get()));
            std::cout << "  [" << i << "] " << inputNames_.back() << "\n";
        }

        size_t nOut = session_->GetOutputCount();
        std::cout << "[HED] Model outputs (" << nOut << "):\n";
        for (size_t i = 0; i < nOut; ++i) {
            auto name = session_->GetOutputNameAllocated(i, alloc);
            outputNames_.push_back(_strdup(name.get()));
            std::cout << "  [" << i << "] " << outputNames_.back() << "\n";
        }

        // Print which execution providers actually loaded
        auto providers = Ort::GetAvailableProviders();
        std::cout << "[HED] Available providers: ";
        for (auto& p : providers) std::cout << p << " ";
        std::cout << "\n";

        isLoaded_ = true;
        std::cout << "[HED] Model loaded OK on " << GetBackendName() << "\n";
        return true;
    } catch (const Ort::Exception& e) {
        std::cerr << "[HED] LoadModel failed: " << e.what() << "\n";
        return false;
    }
}

bool HEDInference::IsLoaded() const { return isLoaded_; }

const char* HEDInference::GetBackendName() const
{
    return backend_ == Backend::GPU ? "GPU (CUDA)" : "CPU";
}

// -----------------------------------------------------------------------------
// ProcessFrame  (public entry point)
// -----------------------------------------------------------------------------

std::vector<std::vector<LaserPoint>> HEDInference::ProcessFrame(
    const cv::Mat& bgrFrame,
    float edgeThreshold,
    int   minContourPoints,
    float smoothEpsilon,
    bool  sampleColor,
    float temporalAlpha)
{
    if (!isLoaded_ || bgrFrame.empty()) return {};

    cv::Mat edgeMap = RunInference(bgrFrame);
    if (edgeMap.empty()) return {};

    // Temporal blending: mix current edge map with previous frame's.
    if (temporalAlpha < 1.0f && !prevEdgeMap_.empty() &&
        prevEdgeMap_.size() == edgeMap.size())
    {
        cv::addWeighted(edgeMap, temporalAlpha,
                        prevEdgeMap_, 1.0f - temporalAlpha,
                        0.0f, edgeMap);
    }
    prevEdgeMap_ = edgeMap.clone();

    return ExtractContours(edgeMap, bgrFrame, edgeThreshold,
                           minContourPoints, smoothEpsilon, sampleColor);
}

// -----------------------------------------------------------------------------
// RunInference  (private)
// -----------------------------------------------------------------------------

cv::Mat HEDInference::RunInference(const cv::Mat& bgrFrame)
{
    // 1. Resize to model's fixed input resolution.
    cv::Mat resized;
    cv::resize(bgrFrame, resized, cv::Size(inputW_, inputH_));

    // 2. Convert to float32 and subtract per-channel BGR means.
    cv::Mat floatImg;
    resized.convertTo(floatImg, CV_32F);

    std::vector<cv::Mat> ch(3);
    cv::split(floatImg, ch);
    ch[0] -= kMeanB;
    ch[1] -= kMeanG;
    ch[2] -= kMeanR;

    // 3. Pack channels into a contiguous [1, 3, H, W] (NCHW) buffer.
    const int H = inputH_, W = inputW_;
    std::vector<float> blob(3 * H * W);

    for (int c = 0; c < 3; ++c) {
        float* dst = blob.data() + c * H * W;
        for (int y = 0; y < H; ++y)
            std::memcpy(dst + y * W, ch[c].ptr<float>(y), W * sizeof(float));
    }

    // 4. Build input tensor and run inference.
    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> shape = { 1, 3, H, W };

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memInfo, blob.data(), blob.size(), shape.data(), shape.size());

    // HED models typically have multiple side outputs + a fused output.
    // We want the last output (fused edge map).
    std::vector<Ort::Value> outputs;
    try {
        outputs = session_->Run(
            Ort::RunOptions{ nullptr },
            inputNames_.data(), &inputTensor, 1,
            outputNames_.data(), outputNames_.size());
    } catch (const Ort::Exception& e) {
        std::cerr << "[HED] Inference error: " << e.what() << "\n";
        return {};
    }

    // 5. Extract fused edge map from the last output tensor.
    Ort::Value& fusedTensor = outputs.back();
    auto outShape = fusedTensor.GetTensorTypeAndShapeInfo().GetShape();

    // Output may be [1,1,H,W] or [1,H,W] — find spatial dims.
    int64_t outH = outShape[outShape.size() - 2];
    int64_t outW = outShape[outShape.size() - 1];

    float* data = fusedTensor.GetTensorMutableData<float>();
    cv::Mat edgeMap(static_cast<int>(outH), static_cast<int>(outW), CV_32F, data);
    edgeMap = edgeMap.clone();  // own the memory before the tensor is freed

    // 6. Resize edge map back to original frame dimensions for accurate contours.
    if (outH != bgrFrame.rows || outW != bgrFrame.cols)
        cv::resize(edgeMap, edgeMap, bgrFrame.size(), 0, 0, cv::INTER_LINEAR);

    return edgeMap;
}

// -----------------------------------------------------------------------------
// ExtractContours  (private)
// -----------------------------------------------------------------------------

std::vector<std::vector<LaserPoint>> HEDInference::ExtractContours(
    const cv::Mat& edgeMap,
    const cv::Mat& srcFrame,
    float threshold,
    int   minContourPoints,
    float smoothEpsilon,
    bool  sampleColor)
{
    // Threshold float edge map to binary 8-bit.
    cv::Mat binary;
    cv::threshold(edgeMap, binary, threshold, 255.0, cv::THRESH_BINARY);
    binary.convertTo(binary, CV_8U);

    // Find contours.
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_LIST, cv::CHAIN_APPROX_TC89_KCOS);

    const float W = static_cast<float>(edgeMap.cols);
    const float H = static_cast<float>(edgeMap.rows);

    std::vector<std::vector<LaserPoint>> result;
    result.reserve(contours.size());

    for (const auto& contour : contours) {
        if (static_cast<int>(contour.size()) < minContourPoints) continue;

        // Optional smoothing — reduces galvo jitter on noisy edges.
        std::vector<cv::Point> pts;
        if (smoothEpsilon > 0.0f)
            cv::approxPolyDP(contour, pts, smoothEpsilon, /*closed=*/true);
        else
            pts = contour;

        if (static_cast<int>(pts.size()) < minContourPoints) continue;

        std::vector<LaserPoint> poly;
        poly.reserve(pts.size() + 1);

        for (const auto& pt : pts) {
            float x =  2.0f * pt.x / W - 1.0f;
            float y = -(2.0f * pt.y / H - 1.0f);

            float r = 1.0f, g = 1.0f, b = 1.0f;
            if (sampleColor && !srcFrame.empty()) {
                // Clamp to source frame bounds.
                int sx = std::clamp(pt.x, 0, srcFrame.cols - 1);
                int sy = std::clamp(pt.y, 0, srcFrame.rows - 1);
                const cv::Vec3b& bgr = srcFrame.at<cv::Vec3b>(sy, sx);
                b = bgr[0] / 255.0f;
                g = bgr[1] / 255.0f;
                r = bgr[2] / 255.0f;
            }

            poly.push_back({ x, y, r, g, b });
        }

        poly.push_back(poly.front());  // close contour
        result.push_back(std::move(poly));
    }

    return result;
}
