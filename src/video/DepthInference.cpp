#include "DepthInference.h"
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cstring>
#include <algorithm>

DepthInference::DepthInference() = default;

DepthInference::~DepthInference()
{
    for (const char* n : inputNames_)  free(const_cast<char*>(n));
    for (const char* n : outputNames_) free(const_cast<char*>(n));
}

// -----------------------------------------------------------------------------
// Initialize
// -----------------------------------------------------------------------------

bool DepthInference::Initialize(Backend backend)
{
    backend_ = backend;
    try {
        env_            = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "DepthInference");
        sessionOptions_ = std::make_unique<Ort::SessionOptions>();
        sessionOptions_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        if (backend_ == Backend::GPU) {
            std::cout << "[Depth] Initializing with GPU (CUDA) backend\n";
            OrtCUDAProviderOptions cuda{};
            cuda.device_id = 0;
            sessionOptions_->AppendExecutionProvider_CUDA(cuda);
        } else {
            std::cout << "[Depth] Initializing with CPU backend\n";
            sessionOptions_->SetIntraOpNumThreads(4);
        }
        return true;
    } catch (const Ort::Exception& e) {
        std::cerr << "[Depth] Init failed: " << e.what() << "\n";
        return false;
    }
}

// -----------------------------------------------------------------------------
// LoadModel
// -----------------------------------------------------------------------------

bool DepthInference::LoadModel(const std::wstring& modelPath)
{
    if (!env_ || !sessionOptions_) {
        std::cerr << "[Depth] Call Initialize() before LoadModel()\n";
        return false;
    }
    try {
        std::wcout << L"[Depth] Loading model: " << modelPath << L"\n";
        session_ = std::make_unique<Ort::Session>(*env_, modelPath.c_str(), *sessionOptions_);

        Ort::AllocatorWithDefaultOptions alloc;

        size_t nIn = session_->GetInputCount();
        std::cout << "[Depth] Model inputs (" << nIn << "):\n";
        for (size_t i = 0; i < nIn; ++i) {
            auto name = session_->GetInputNameAllocated(i, alloc);
            inputNames_.push_back(_strdup(name.get()));

            auto info  = session_->GetInputTypeInfo(i);
            auto shape = info.GetTensorTypeAndShapeInfo().GetShape();
            std::cout << "  [" << i << "] " << inputNames_.back() << "  shape=[";
            for (size_t d = 0; d < shape.size(); ++d)
                std::cout << (d ? "," : "") << shape[d];
            std::cout << "]\n";

            // If the model has dynamic spatial dims, keep our 518 default.
            // If the model reports fixed dims, adopt them.
            if (shape.size() == 4 && shape[2] > 0 && shape[3] > 0) {
                inputH_ = static_cast<int>(shape[2]);
                inputW_ = static_cast<int>(shape[3]);
            }
        }

        size_t nOut = session_->GetOutputCount();
        std::cout << "[Depth] Model outputs (" << nOut << "):\n";
        for (size_t i = 0; i < nOut; ++i) {
            auto name = session_->GetOutputNameAllocated(i, alloc);
            outputNames_.push_back(_strdup(name.get()));
            std::cout << "  [" << i << "] " << outputNames_.back() << "\n";
        }

        auto providers = Ort::GetAvailableProviders();
        std::cout << "[Depth] Available providers: ";
        for (auto& p : providers) std::cout << p << " ";
        std::cout << "\n";

        isLoaded_ = true;
        std::cout << "[Depth] Model loaded OK on " << GetBackendName()
                  << "  input=" << inputW_ << "x" << inputH_ << "\n";
        return true;
    } catch (const Ort::Exception& e) {
        std::cerr << "[Depth] LoadModel failed: " << e.what() << "\n";
        return false;
    }
}

bool DepthInference::IsLoaded()         const { return isLoaded_; }
const char* DepthInference::GetBackendName() const
{
    return backend_ == Backend::GPU ? "GPU (CUDA)" : "CPU";
}

// -----------------------------------------------------------------------------
// ProcessFrame  (public)
// -----------------------------------------------------------------------------

std::vector<std::vector<LaserPoint>> DepthInference::ProcessFrame(
    const cv::Mat& bgrFrame,
    Mode  mode,
    float edgeThreshold,
    int   isolineCount,
    int   minContourPoints,
    float smoothEpsilon)
{
    if (!isLoaded_ || bgrFrame.empty()) return {};

    cv::Mat depthRaw = RunInference(bgrFrame);
    if (depthRaw.empty()) return {};

    // Normalize to [0, 1] using the min/max of this frame.
    double dMin, dMax;
    cv::minMaxLoc(depthRaw, &dMin, &dMax);
    cv::Mat depthNorm;
    if (dMax - dMin < 1e-6)
        depthNorm = cv::Mat::zeros(depthRaw.size(), CV_32F);
    else
        depthNorm = (depthRaw - static_cast<float>(dMin)) /
                    static_cast<float>(dMax - dMin);

    lastDepth_ = depthNorm.clone();

    std::vector<std::vector<LaserPoint>> result;

    if (mode == Mode::Edges || mode == Mode::Both) {
        auto edges = ExtractEdges(depthNorm, edgeThreshold, minContourPoints, smoothEpsilon);
        result.insert(result.end(), edges.begin(), edges.end());
    }
    if (mode == Mode::Isolines || mode == Mode::Both) {
        auto lines = ExtractIsolines(depthNorm, isolineCount, minContourPoints, smoothEpsilon);
        result.insert(result.end(), lines.begin(), lines.end());
    }
    return result;
}

// -----------------------------------------------------------------------------
// RunInference  (private)
// -----------------------------------------------------------------------------

cv::Mat DepthInference::RunInference(const cv::Mat& bgrFrame)
{
    // 1. Resize to model input.
    cv::Mat resized;
    cv::resize(bgrFrame, resized, cv::Size(inputW_, inputH_));

    // 2. BGR -> RGB, float32 in [0,1], then ImageNet normalise.
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    cv::Mat floatImg;
    rgb.convertTo(floatImg, CV_32F, 1.0f / 255.0f);

    std::vector<cv::Mat> ch(3);
    cv::split(floatImg, ch);
    // ch[0]=R, ch[1]=G, ch[2]=B  (after cvtColor)
    ch[0] = (ch[0] - kMeanR) / kStdR;
    ch[1] = (ch[1] - kMeanG) / kStdG;
    ch[2] = (ch[2] - kMeanB) / kStdB;

    // 3. Pack [1, 3, H, W] NCHW.
    const int H = inputH_, W = inputW_;
    std::vector<float> blob(3 * H * W);
    for (int c = 0; c < 3; ++c) {
        float* dst = blob.data() + c * H * W;
        for (int y = 0; y < H; ++y)
            std::memcpy(dst + y * W, ch[c].ptr<float>(y), W * sizeof(float));
    }

    // 4. Build input tensor.
    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> shape = { 1, 3, H, W };

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memInfo, blob.data(), blob.size(), shape.data(), shape.size());

    // 5. Run.
    std::vector<Ort::Value> outputs;
    try {
        outputs = session_->Run(
            Ort::RunOptions{ nullptr },
            inputNames_.data(), &inputTensor, 1,
            outputNames_.data(), outputNames_.size());
    } catch (const Ort::Exception& e) {
        std::cerr << "[Depth] Inference error: " << e.what() << "\n";
        return {};
    }

    // 6. Extract depth tensor — use first output.
    Ort::Value& depthTensor = outputs.front();
    auto outShape = depthTensor.GetTensorTypeAndShapeInfo().GetShape();

    // Shape is [1, H, W] or [1, 1, H, W] — find spatial dims.
    int64_t outH = outShape[outShape.size() - 2];
    int64_t outW = outShape[outShape.size() - 1];

    float* data = depthTensor.GetTensorMutableData<float>();
    cv::Mat depthMap(static_cast<int>(outH), static_cast<int>(outW), CV_32F, data);
    depthMap = depthMap.clone();

    // 7. Resize back to original frame resolution.
    if (outH != bgrFrame.rows || outW != bgrFrame.cols)
        cv::resize(depthMap, depthMap, bgrFrame.size(), 0, 0, cv::INTER_LINEAR);

    return depthMap;
}

// -----------------------------------------------------------------------------
// ExtractEdges  (private)
// Compute Sobel gradient of the normalised depth map; threshold; find contours.
// -----------------------------------------------------------------------------

std::vector<std::vector<LaserPoint>> DepthInference::ExtractEdges(
    const cv::Mat& depthNorm,
    float threshold,
    int   minPoints,
    float smoothEpsilon)
{
    // Sobel gradient magnitude on the depth map.
    cv::Mat gx, gy;
    cv::Sobel(depthNorm, gx, CV_32F, 1, 0, 3);
    cv::Sobel(depthNorm, gy, CV_32F, 0, 1, 3);

    cv::Mat magnitude;
    cv::magnitude(gx, gy, magnitude);

    // Threshold to binary.
    cv::Mat binary;
    cv::threshold(magnitude, binary, threshold, 255.0, cv::THRESH_BINARY);
    binary.convertTo(binary, CV_8U);

    // Optional: slight dilation to connect near-breaks in the edge map.
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(binary, binary, kernel, cv::Point(-1,-1), 1);

    return ContoursToPolylines(binary, depthNorm.cols, depthNorm.rows, minPoints, smoothEpsilon);
}

// -----------------------------------------------------------------------------
// ExtractIsolines  (private)
// Band-threshold the depth map at N evenly-spaced levels.
// Each level produces a thin ring of contours at that depth plane.
// -----------------------------------------------------------------------------

std::vector<std::vector<LaserPoint>> DepthInference::ExtractIsolines(
    const cv::Mat& depthNorm,
    int   count,
    int   minPoints,
    float smoothEpsilon)
{
    std::vector<std::vector<LaserPoint>> result;

    if (count < 1) return result;

    // Depth 0 = nearest, 1 = farthest.
    // We skip the very edges (0.05 and 0.95) to avoid spurious borders.
    const float step  = 0.9f / static_cast<float>(count + 1);
    const float width = step * 0.5f;   // half-band on each side of the level

    for (int i = 1; i <= count; ++i) {
        float level = 0.05f + i * step;
        float lo    = level - width;
        float hi    = level + width;

        // Binary: pixels within [lo, hi] band = white.
        cv::Mat band;
        cv::inRange(depthNorm, lo, hi, band);  // output is already CV_8U

        auto polys = ContoursToPolylines(band, depthNorm.cols, depthNorm.rows, minPoints, smoothEpsilon);
        result.insert(result.end(), polys.begin(), polys.end());
    }
    return result;
}

// -----------------------------------------------------------------------------
// ContoursToPolylines  (private)
// Shared helper: binary CV_8U -> normalized LaserPoint polylines.
// -----------------------------------------------------------------------------

std::vector<std::vector<LaserPoint>> DepthInference::ContoursToPolylines(
    const cv::Mat& binary,
    int   frameW,
    int   frameH,
    int   minPoints,
    float smoothEpsilon)
{
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_LIST, cv::CHAIN_APPROX_TC89_KCOS);

    const float W = static_cast<float>(frameW);
    const float H = static_cast<float>(frameH);

    std::vector<std::vector<LaserPoint>> result;
    result.reserve(contours.size());

    for (const auto& contour : contours) {
        if (static_cast<int>(contour.size()) < minPoints) continue;

        std::vector<cv::Point> pts;
        if (smoothEpsilon > 0.0f)
            cv::approxPolyDP(contour, pts, smoothEpsilon, /*closed=*/true);
        else
            pts = contour;

        if (static_cast<int>(pts.size()) < minPoints) continue;

        std::vector<LaserPoint> poly;
        poly.reserve(pts.size() + 1);

        for (const auto& pt : pts) {
            float x =  2.0f * pt.x / W - 1.0f;
            float y = -(2.0f * pt.y / H - 1.0f);   // flip Y
            poly.push_back({ x, y, 1.0f, 1.0f, 1.0f });
        }

        poly.push_back(poly.front());   // close contour
        result.push_back(std::move(poly));
    }
    return result;
}
