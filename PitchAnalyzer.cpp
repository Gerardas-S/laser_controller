#include "PitchAnalyzer.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

// ---------------------------------------------------------------------------
// CREPE pitch bins: equally spaced in cents, 20 cents apart
// Bin 0  = 32.70 Hz (C1),  bin 359 = 1975.5 Hz (B6)
// Formula: hz(b) = 10.0 * 2^((b * 20 + 1997.3796) / 1200)
// ---------------------------------------------------------------------------

static constexpr float kCentsPerBin = 20.0f;
static constexpr float kBin0Cents   = 1997.3796f;  // cents from A0 for bin 0

float PitchAnalyzer::BinToHz(int bin)
{
    float cents = kBin0Cents + static_cast<float>(bin) * kCentsPerBin;
    return 10.0f * std::pow(2.0f, cents / 1200.0f);
}

float PitchAnalyzer::HzToNorm(float hz)
{
    if (hz <= 0.0f) return 0.0f;
    // Log scale: C1 (32.70 Hz) = 0, C8 (4186 Hz) = 1
    static const float logC1 = std::log2(32.70f);
    static const float logC8 = std::log2(4186.0f);
    float v = (std::log2(hz) - logC1) / (logC8 - logC1);
    return std::clamp(v, 0.0f, 1.0f);
}

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

PitchAnalyzer::PitchAnalyzer()
    : memInfo_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
    // Pre-size ring buffer to hold ~0.5 s at 48 kHz source
    inputRing_.resize(48000, 0.0f);
}

PitchAnalyzer::~PitchAnalyzer() = default;

// ---------------------------------------------------------------------------
// Initialize — load ONNX model
// ---------------------------------------------------------------------------

bool PitchAnalyzer::Initialize(const std::string& modelPath, bool useGpu)
{
    try {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "PitchAnalyzer");

        sessionOpts_ = std::make_unique<Ort::SessionOptions>();
        sessionOpts_->SetIntraOpNumThreads(1);
        sessionOpts_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        if (useGpu) {
            // CUDA EP — optional, falls back to CPU if unavailable
            OrtCUDAProviderOptions cuda{};
            cuda.device_id = 0;
            sessionOpts_->AppendExecutionProvider_CUDA(cuda);
        }

        // Convert to wide string for Windows ORT API
        std::wstring wpath(modelPath.begin(), modelPath.end());
        session_ = std::make_unique<Ort::Session>(*env_, wpath.c_str(), *sessionOpts_);

        // Cache input/output names
        Ort::AllocatorWithDefaultOptions alloc;
        auto inNamePtr  = session_->GetInputNameAllocated(0, alloc);
        auto outNamePtr = session_->GetOutputNameAllocated(0, alloc);
        inputName_  = inNamePtr.get();
        outputName_ = outNamePtr.get();

        // Verify input shape: [batch, 1024]
        auto inInfo  = session_->GetInputTypeInfo(0);
        auto inShape = inInfo.GetTensorTypeAndShapeInfo().GetShape();
        if (inShape.size() != 2 || (inShape[1] != kWindowSamples && inShape[1] != -1)) {
            std::cerr << "[PitchAnalyzer] Unexpected input shape (expected [N, 1024])\n";
            return false;
        }

        loaded_ = true;
        std::cout << "[PitchAnalyzer] CREPE model loaded: " << modelPath << "\n";
        return true;
    }
    catch (const Ort::Exception& ex) {
        std::cerr << "[PitchAnalyzer] ORT error: " << ex.what() << "\n";
        return false;
    }
}

// ---------------------------------------------------------------------------
// PushSamples — buffer audio, fire inference when a full window is ready
// ---------------------------------------------------------------------------

void PitchAnalyzer::PushSamples(const float* samples, int count)
{
    if (!loaded_) return;

    // Resample to 16 kHz
    int dstLen = static_cast<int>(std::round(
        static_cast<double>(count) * kModelRate / sourceSampleRate_));
    if (dstLen <= 0) return;

    std::vector<float> resampled(dstLen);
    ResampleLinear(samples, count, resampled.data(), dstLen);

    {
        std::lock_guard<std::mutex> lock(ringMutex_);
        for (float s : resampled) {
            inputRing_[ringWrite_] = s;
            ringWrite_ = (ringWrite_ + 1) % static_cast<int>(inputRing_.size());
            if (ringFill_ < static_cast<int>(inputRing_.size()))
                ++ringFill_;
            else
                ringRead_ = ringWrite_;  // overwrite oldest
        }
    }

    // Fire inference if we have at least kWindowSamples
    if (ringFill_ >= kWindowSamples) {
        std::vector<float> window(kWindowSamples);
        {
            std::lock_guard<std::mutex> lock(ringMutex_);
            int rd = ringRead_;
            for (int i = 0; i < kWindowSamples; ++i) {
                window[i] = inputRing_[rd];
                rd = (rd + 1) % static_cast<int>(inputRing_.size());
            }
            // Advance read by hopSize (512 at 16 kHz = 32 ms)
            int hop = 512;
            ringRead_ = (ringRead_ + hop) % static_cast<int>(inputRing_.size());
            ringFill_ = std::max(0, ringFill_ - hop);
        }
        ProcessWindow(window.data(), kWindowSamples);
    }
}

// ---------------------------------------------------------------------------
// ProcessWindow — run CREPE on a normalised 1024-sample @16kHz frame
// ---------------------------------------------------------------------------

bool PitchAnalyzer::ProcessWindow(const float* samples16k, int count)
{
    if (!loaded_ || count < kWindowSamples) return false;

    // Mean-normalise (CREPE expects zero-mean unit-variance input)
    std::vector<float> normed(samples16k, samples16k + kWindowSamples);
    float mean = std::accumulate(normed.begin(), normed.end(), 0.0f) / kWindowSamples;
    float var  = 0.0f;
    for (float v : normed) var += (v - mean) * (v - mean);
    var = std::max(var / kWindowSamples, 1e-8f);
    float invStd = 1.0f / std::sqrt(var);
    for (float& v : normed) v = (v - mean) * invStd;

    return RunInference(normed.data());
}

// ---------------------------------------------------------------------------
// RunInference — private ONNX call
// ---------------------------------------------------------------------------

bool PitchAnalyzer::RunInference(const float* window16k)
{
    try {
        std::vector<int64_t> inShape = { 1, kWindowSamples };
        std::vector<float>   inData(window16k, window16k + kWindowSamples);

        auto inTensor = Ort::Value::CreateTensor<float>(
            memInfo_, inData.data(), inData.size(),
            inShape.data(), inShape.size());

        const char* inNames[]  = { inputName_.c_str() };
        const char* outNames[] = { outputName_.c_str() };

        auto results = session_->Run(
            Ort::RunOptions{nullptr},
            inNames,  &inTensor,  1,
            outNames, 1);

        // Output: [1, 360] salience
        auto& outTensor = results[0];
        float* salience = outTensor.GetTensorMutableData<float>();
        int64_t numBins = outTensor.GetTensorTypeAndShapeInfo().GetElementCount();

        if (numBins < kPitchBins) {
            std::cerr << "[PitchAnalyzer] Unexpected output bins: " << numBins << "\n";
            return false;
        }

        // Apply softmax for well-calibrated confidence
        float maxVal = *std::max_element(salience, salience + kPitchBins);
        float expSum = 0.0f;
        std::vector<float> probs(kPitchBins);
        for (int i = 0; i < kPitchBins; ++i) {
            probs[i] = std::exp(salience[i] - maxVal);
            expSum  += probs[i];
        }
        for (float& p : probs) p /= expSum;

        // Weighted mean pitch estimate (more stable than argmax)
        float weightedBin = 0.0f;
        float totalWeight = 0.0f;
        int   peakBin     = 0;
        float peakVal     = 0.0f;
        for (int i = 0; i < kPitchBins; ++i) {
            weightedBin += static_cast<float>(i) * probs[i];
            totalWeight += probs[i];
            if (probs[i] > peakVal) { peakVal = probs[i]; peakBin = i; }
        }
        float estBin = (totalWeight > 0.0f) ? weightedBin / totalWeight
                                             : static_cast<float>(peakBin);

        float hz   = BinToHz(static_cast<int>(std::round(estBin)));
        float conf = peakVal * kPitchBins;   // peak probability (0-1 rescaled)
        conf = std::clamp(conf, 0.0f, 1.0f);

        // Voiced / unvoiced threshold — below 0.5 confidence treat as silence
        if (conf < 0.5f) hz = 0.0f;

        float norm = HzToNorm(hz);

        {
            std::lock_guard<std::mutex> lock(resultMutex_);
            pitchHz_    = hz;
            confidence_ = conf;
            pitchNorm_  = norm;
        }
        return true;
    }
    catch (const Ort::Exception& ex) {
        std::cerr << "[PitchAnalyzer] Inference error: " << ex.what() << "\n";
        return false;
    }
}

// ---------------------------------------------------------------------------
// Getters
// ---------------------------------------------------------------------------

float PitchAnalyzer::GetPitchHz() const
{
    std::lock_guard<std::mutex> lock(resultMutex_);
    return pitchHz_;
}

float PitchAnalyzer::GetConfidence() const
{
    std::lock_guard<std::mutex> lock(resultMutex_);
    return confidence_;
}

float PitchAnalyzer::GetPitchNorm() const
{
    std::lock_guard<std::mutex> lock(resultMutex_);
    return pitchNorm_;
}

// ---------------------------------------------------------------------------
// FillFeatures — write latest pitch estimate into AudioFeatures
// ---------------------------------------------------------------------------

void PitchAnalyzer::FillFeatures(AudioFeatures& feat, float smoothAlpha)
{
    float hz, conf, norm;
    {
        std::lock_guard<std::mutex> lock(resultMutex_);
        hz   = pitchHz_;
        conf = confidence_;
        norm = pitchNorm_;
    }

    feat.pitch           = hz;
    feat.pitchConfidence = conf;
    feat.pitchNorm       = norm;

    // Exponential smooth — only when voiced
    if (hz > 0.0f)
        smoothedPitchNorm_ = smoothedPitchNorm_ * (1.0f - smoothAlpha) + norm * smoothAlpha;

    feat.pitchSmooth = smoothedPitchNorm_;
}

// ---------------------------------------------------------------------------
// ResampleLinear — simple linear interpolation
// ---------------------------------------------------------------------------

void PitchAnalyzer::ResampleLinear(const float* src, int srcLen,
                                    float* dst,  int dstLen) const
{
    if (dstLen <= 0 || srcLen <= 0) return;
    if (dstLen == 1) { dst[0] = src[0]; return; }

    float ratio = static_cast<float>(srcLen - 1) / static_cast<float>(dstLen - 1);
    for (int i = 0; i < dstLen; ++i) {
        float pos = i * ratio;
        int   lo  = static_cast<int>(pos);
        int   hi  = std::min(lo + 1, srcLen - 1);
        float frac = pos - lo;
        dst[i] = src[lo] * (1.0f - frac) + src[hi] * frac;
    }
}
