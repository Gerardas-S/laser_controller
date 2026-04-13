#include "SAMInference.h"
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cstring>

SAMInference::SAMInference() = default;

SAMInference::~SAMInference()
{
    for (const char* n : encInputNames_)  free(const_cast<char*>(n));
    for (const char* n : encOutputNames_) free(const_cast<char*>(n));
    for (const char* n : decInputNames_)  free(const_cast<char*>(n));
    for (const char* n : decOutputNames_) free(const_cast<char*>(n));
}

// -----------------------------------------------------------------------------
// Initialize
// -----------------------------------------------------------------------------

bool SAMInference::Initialize(Backend backend)
{
    backend_ = backend;

    try {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "SAMInference");

        encoderOptions_ = std::make_unique<Ort::SessionOptions>();
        decoderOptions_ = std::make_unique<Ort::SessionOptions>();

        if (backend_ == Backend::GPU) {
            std::cout << "[SAM] Initializing with GPU (CUDA) backend\n";
            OrtCUDAProviderOptions cuda{};
            cuda.device_id = 0;
            encoderOptions_->AppendExecutionProvider_CUDA(cuda);
            decoderOptions_->AppendExecutionProvider_CUDA(cuda);
        } else {
            std::cout << "[SAM] Initializing with CPU backend\n";
            encoderOptions_->SetIntraOpNumThreads(4);
            encoderOptions_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            decoderOptions_->SetIntraOpNumThreads(2);
            decoderOptions_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        }
        return true;
    } catch (const Ort::Exception& e) {
        std::cerr << "[SAM] Init failed: " << e.what() << "\n";
        return false;
    }
}

// -----------------------------------------------------------------------------
// LoadModels
// -----------------------------------------------------------------------------

bool SAMInference::LoadModels(const std::wstring& encoderPath,
                               const std::wstring& decoderPath)
{
    if (!env_) {
        std::cerr << "[SAM] Call Initialize() before LoadModels()\n";
        return false;
    }

    try {
        Ort::AllocatorWithDefaultOptions alloc;

        // --- Encoder ---
        std::wcout << L"[SAM] Loading encoder: " << encoderPath << L"\n";
        encoderSession_ = std::make_unique<Ort::Session>(*env_, encoderPath.c_str(), *encoderOptions_);

        for (size_t i = 0; i < encoderSession_->GetInputCount(); ++i) {
            auto n = encoderSession_->GetInputNameAllocated(i, alloc);
            encInputNames_.push_back(_strdup(n.get()));
            std::cout << "  enc in  [" << i << "] " << encInputNames_.back() << "\n";
        }
        for (size_t i = 0; i < encoderSession_->GetOutputCount(); ++i) {
            auto n = encoderSession_->GetOutputNameAllocated(i, alloc);
            encOutputNames_.push_back(_strdup(n.get()));
            std::cout << "  enc out [" << i << "] " << encOutputNames_.back() << "\n";
        }

        // --- Decoder ---
        std::wcout << L"[SAM] Loading decoder: " << decoderPath << L"\n";
        decoderSession_ = std::make_unique<Ort::Session>(*env_, decoderPath.c_str(), *decoderOptions_);

        for (size_t i = 0; i < decoderSession_->GetInputCount(); ++i) {
            auto n = decoderSession_->GetInputNameAllocated(i, alloc);
            decInputNames_.push_back(_strdup(n.get()));
            std::cout << "  dec in  [" << i << "] " << decInputNames_.back() << "\n";
        }
        for (size_t i = 0; i < decoderSession_->GetOutputCount(); ++i) {
            auto n = decoderSession_->GetOutputNameAllocated(i, alloc);
            decOutputNames_.push_back(_strdup(n.get()));
            std::cout << "  dec out [" << i << "] " << decOutputNames_.back() << "\n";
        }

        isLoaded_ = true;
        std::cout << "[SAM] Models loaded OK\n";
        return true;
    } catch (const Ort::Exception& e) {
        std::cerr << "[SAM] LoadModels failed: " << e.what() << "\n";
        return false;
    }
}

bool SAMInference::IsLoaded() const { return isLoaded_; }

const char* SAMInference::GetBackendName() const
{
    return backend_ == Backend::GPU ? "GPU (CUDA)" : "CPU";
}

// -----------------------------------------------------------------------------
// ProcessFrame  (public entry point)
// -----------------------------------------------------------------------------

std::vector<std::vector<LaserPoint>> SAMInference::ProcessFrame(
    const cv::Mat& bgrFrame,
    int   gridSize,
    float iouThreshold,
    int   minContourPoints,
    float smoothEpsilon,
    bool  sampleColor)
{
    if (!isLoaded_ || bgrFrame.empty()) return {};

    const float origW = static_cast<float>(bgrFrame.cols);
    const float origH = static_cast<float>(bgrFrame.rows);

    // 1. Encode frame once.
    cv::Mat embedding = RunEncoder(bgrFrame);
    if (embedding.empty()) return {};

    // 2. Generate uniform NxN grid of point prompts.
    const float scaleX = kEncSize / origW;
    const float scaleY = kEncSize / origH;

    std::vector<std::vector<LaserPoint>> result;

    cv::Mat claimed = cv::Mat::zeros(static_cast<int>(origH),
                                     static_cast<int>(origW), CV_8U);

    for (int row = 0; row < gridSize; ++row) {
        for (int col = 0; col < gridSize; ++col) {
            float px    = (col + 0.5f) * origW / gridSize;
            float py    = (row + 0.5f) * origH / gridSize;
            float encPx = px * scaleX;
            float encPy = py * scaleY;

            cv::Mat mask = RunDecoder(embedding, encPx, encPy, origW, origH);
            if (mask.empty()) continue;

            cv::Mat overlap;
            cv::bitwise_and(mask, claimed, overlap);
            int overlapPx = cv::countNonZero(overlap);
            int maskPx    = cv::countNonZero(mask);
            if (maskPx == 0) continue;
            if (static_cast<float>(overlapPx) / maskPx > 0.5f) continue;

            cv::bitwise_or(claimed, mask, claimed);

            auto contours = ExtractContours(mask, bgrFrame, origW, origH,
                                            minContourPoints, smoothEpsilon, sampleColor);
            for (auto& c : contours)
                result.push_back(std::move(c));
        }
    }

    return result;
}

// -----------------------------------------------------------------------------
// RunEncoder  (private)
// -----------------------------------------------------------------------------

cv::Mat SAMInference::RunEncoder(const cv::Mat& bgrFrame)
{
    // 1. Resize to 1024x1024.
    cv::Mat resized;
    cv::resize(bgrFrame, resized, cv::Size(kEncSize, kEncSize));

    // 2. BGR -> RGB, float32, normalise with ImageNet mean/std.
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> ch(3);
    cv::split(rgb, ch);
    ch[0] = (ch[0] - kMeanR) / kStdR;
    ch[1] = (ch[1] - kMeanG) / kStdG;
    ch[2] = (ch[2] - kMeanB) / kStdB;

    // 3. Pack into [1, 3, H, W] NCHW.
    const int H = kEncSize, W = kEncSize;
    std::vector<float> blob(3 * H * W);
    for (int c = 0; c < 3; ++c) {
        float* dst = blob.data() + c * H * W;
        for (int y = 0; y < H; ++y)
            std::memcpy(dst + y * W, ch[c].ptr<float>(y), W * sizeof(float));
    }

    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> shape = { 1, 3, H, W };

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memInfo, blob.data(), blob.size(), shape.data(), shape.size());

    std::vector<Ort::Value> outputs;
    try {
        outputs = encoderSession_->Run(
            Ort::RunOptions{ nullptr },
            encInputNames_.data(), &inputTensor, 1,
            encOutputNames_.data(), encOutputNames_.size());
    } catch (const Ort::Exception& e) {
        std::cerr << "[SAM] Encoder error: " << e.what() << "\n";
        return {};
    }

    // Return embedding as a flat CV_32F mat [256 * 64 * 64].
    // We keep it flat; RunDecoder will wrap it in a tensor directly.
    auto& emb = outputs[0];
    auto embShape = emb.GetTensorTypeAndShapeInfo().GetShape();
    size_t embSize = 1;
    for (auto d : embShape) embSize *= d;

    float* embData = emb.GetTensorMutableData<float>();
    cv::Mat embedding(1, static_cast<int>(embSize), CV_32F);
    std::memcpy(embedding.ptr<float>(), embData, embSize * sizeof(float));

    // Store the shape so RunDecoder can reconstruct the tensor.
    // We smuggle it via the mat's step info — just store flat, shape is fixed.
    return embedding;
}

// -----------------------------------------------------------------------------
// RunDecoder  (private)
// -----------------------------------------------------------------------------

cv::Mat SAMInference::RunDecoder(const cv::Mat& imageEmbedding,
                                  float px, float py,
                                  float origW, float origH)
{
    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // image_embeddings: [1, 256, 64, 64]
    std::vector<int64_t> embShape = { 1, 256, 64, 64 };
    Ort::Value embTensor = Ort::Value::CreateTensor<float>(
        memInfo,
        const_cast<float*>(imageEmbedding.ptr<float>()),
        imageEmbedding.total(),
        embShape.data(), embShape.size());

    // point_coords: [1, 1, 2]  — one foreground point
    std::vector<float> coords = { px, py };
    std::vector<int64_t> coordShape = { 1, 1, 2 };
    Ort::Value coordTensor = Ort::Value::CreateTensor<float>(
        memInfo, coords.data(), coords.size(), coordShape.data(), coordShape.size());

    // point_labels: [1, 1]  — 1 = foreground
    std::vector<float> labels = { 1.0f };
    std::vector<int64_t> labelShape = { 1, 1 };
    Ort::Value labelTensor = Ort::Value::CreateTensor<float>(
        memInfo, labels.data(), labels.size(), labelShape.data(), labelShape.size());

    // mask_input: [1, 1, 256, 256]  — zeros (no prior mask)
    std::vector<float> maskInput(1 * 1 * 256 * 256, 0.0f);
    std::vector<int64_t> maskInputShape = { 1, 1, 256, 256 };
    Ort::Value maskInputTensor = Ort::Value::CreateTensor<float>(
        memInfo, maskInput.data(), maskInput.size(),
        maskInputShape.data(), maskInputShape.size());

    // has_mask_input: [1]  — 0 = no prior mask
    std::vector<float> hasMask = { 0.0f };
    std::vector<int64_t> hasMaskShape = { 1 };
    Ort::Value hasMaskTensor = Ort::Value::CreateTensor<float>(
        memInfo, hasMask.data(), hasMask.size(),
        hasMaskShape.data(), hasMaskShape.size());

    // orig_im_size: [2]  — original frame dimensions
    std::vector<float> origSize = { origH, origW };
    std::vector<int64_t> origSizeShape = { 2 };
    Ort::Value origSizeTensor = Ort::Value::CreateTensor<float>(
        memInfo, origSize.data(), origSize.size(),
        origSizeShape.data(), origSizeShape.size());

    // Assemble inputs in SAM decoder order.
    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(embTensor));
    inputs.push_back(std::move(coordTensor));
    inputs.push_back(std::move(labelTensor));
    inputs.push_back(std::move(maskInputTensor));
    inputs.push_back(std::move(hasMaskTensor));
    inputs.push_back(std::move(origSizeTensor));

    std::vector<Ort::Value> outputs;
    try {
        outputs = decoderSession_->Run(
            Ort::RunOptions{ nullptr },
            decInputNames_.data(), inputs.data(), inputs.size(),
            decOutputNames_.data(), decOutputNames_.size());
    } catch (const Ort::Exception& e) {
        std::cerr << "[SAM] Decoder error: " << e.what() << "\n";
        return {};
    }

    // outputs[0] = masks        [1, 3, H, W]
    // outputs[1] = iou_predictions [1, 3]
    // Take the mask with the highest IoU prediction.
    float* iouData = outputs[1].GetTensorMutableData<float>();
    int bestIdx = 0;
    if (iouData[1] > iouData[bestIdx]) bestIdx = 1;
    if (iouData[2] > iouData[bestIdx]) bestIdx = 2;

    auto maskShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    int H = static_cast<int>(maskShape[2]);
    int W = static_cast<int>(maskShape[3]);

    float* maskData = outputs[0].GetTensorMutableData<float>() + bestIdx * H * W;

    // Threshold at 0.0 (SAM outputs logits).
    cv::Mat mask(H, W, CV_8U);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            mask.at<uint8_t>(y, x) = (maskData[y * W + x] > 0.0f) ? 255 : 0;

    return mask;
}

// -----------------------------------------------------------------------------
// ExtractContours  (private)
// -----------------------------------------------------------------------------

std::vector<std::vector<LaserPoint>> SAMInference::ExtractContours(
    const cv::Mat& mask,
    const cv::Mat& srcFrame,
    float origW, float origH,
    int   minContourPoints,
    float smoothEpsilon,
    bool  sampleColor)
{
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_TC89_KCOS);

    std::vector<std::vector<LaserPoint>> result;

    for (const auto& contour : contours) {
        if (static_cast<int>(contour.size()) < minContourPoints) continue;

        // Optional smoothing.
        std::vector<cv::Point> pts;
        if (smoothEpsilon > 0.0f)
            cv::approxPolyDP(contour, pts, smoothEpsilon, /*closed=*/true);
        else
            pts = contour;

        if (static_cast<int>(pts.size()) < minContourPoints) continue;

        std::vector<LaserPoint> poly;
        poly.reserve(pts.size() + 1);

        for (const auto& pt : pts) {
            float x =  2.0f * pt.x / origW - 1.0f;
            float y = -(2.0f * pt.y / origH - 1.0f);

            float r = 1.0f, g = 1.0f, b = 1.0f;
            if (sampleColor && !srcFrame.empty()) {
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
