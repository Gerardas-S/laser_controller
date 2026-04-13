#pragma once

#include "AudioFeatures.h"
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <memory>
#include <mutex>
#include <deque>

// -----------------------------------------------------------------------------
// PitchAnalyzer
//
// Real-time fundamental-frequency estimator powered by CREPE (ONNX).
// CREPE (Convolutional REpresentation for Pitch Estimation) takes a 1024-sample
// audio window at 16 kHz and outputs a 360-bin pitch salience distribution
// covering 32.7–1975.5 Hz (20 cents/bin).
//
// Usage:
//   PitchAnalyzer pa;
//   if (pa.Initialize("models/crepe/model.onnx")) {
//       pa.ProcessWindow(samples16k, numSamples);
//       float hz  = pa.GetPitchHz();
//       float conf = pa.GetConfidence();
//   }
//
// Integration:
//   MusicPipeline calls FillFeatures(feat) after each AudioAnalyzer callback
//   to populate feat.pitch / feat.pitchConfidence / feat.pitchNorm.
//
// Resampling:
//   If the host sample rate is not 16000, call SetSourceSampleRate() before
//   the first ProcessSamples() call. Linear interpolation is used.
// -----------------------------------------------------------------------------

class PitchAnalyzer {
public:
    // Number of pitch bins in the CREPE output (32.7–1975.5 Hz, 20 cents/bin)
    static constexpr int   kPitchBins      = 360;
    static constexpr int   kWindowSamples  = 1024;   // at 16 kHz
    static constexpr float kMinHz          = 32.70f; // C1
    static constexpr float kMaxHz          = 1975.5f; // B6
    static constexpr int   kModelRate      = 16000;

    PitchAnalyzer();
    ~PitchAnalyzer();

    // Load CREPE ONNX model.  Returns false if file not found or model mismatch.
    bool Initialize(const std::string& modelPath, bool useGpu = false);
    bool IsLoaded() const { return loaded_; }

    // Tell the analyzer the sample rate of incoming audio.
    // Internally resamples to 16 kHz before inference.
    void SetSourceSampleRate(int sr) { sourceSampleRate_ = sr; }

    // Push raw mono audio samples (any count).  Thread-safe.
    // Internally buffers until a full 1024-sample window is available.
    void PushSamples(const float* samples, int count);

    // Run inference synchronously on a pre-assembled 1024-sample @16kHz window.
    // Returns true if a new estimate is available.
    bool ProcessWindow(const float* samples16k, int count);

    // Latest results (thread-safe).
    float GetPitchHz()     const;
    float GetConfidence()  const;
    float GetPitchNorm()   const;   // [0,1] log scale C1→C8

    // Fill AudioFeatures pitch fields from latest estimate.
    void FillFeatures(AudioFeatures& feat, float smoothAlpha = 0.15f);

    // Convert a CREPE bin index to Hz.
    static float BinToHz(int bin);

    // Convert Hz to normalised [0,1] (log scale C1–C8).
    static float HzToNorm(float hz);

private:
    bool RunInference(const float* window16k);
    void ResampleLinear(const float* src, int srcLen, float* dst, int dstLen) const;

    // ONNX Runtime
    std::unique_ptr<Ort::Env>            env_;
    std::unique_ptr<Ort::SessionOptions> sessionOpts_;
    std::unique_ptr<Ort::Session>        session_;
    Ort::MemoryInfo                      memInfo_;

    bool loaded_ = false;
    int  sourceSampleRate_ = 44100;

    // Input/output names cached after model load
    std::string inputName_;
    std::string outputName_;

    // Ring buffer for incoming audio (at source sample rate)
    std::vector<float>  inputRing_;
    int                 ringWrite_ = 0;
    int                 ringRead_  = 0;
    int                 ringFill_  = 0;
    mutable std::mutex  ringMutex_;

    // Latest results
    mutable std::mutex resultMutex_;
    float pitchHz_    = 0.0f;
    float confidence_ = 0.0f;
    float pitchNorm_  = 0.0f;
    float pitchSmooth_= 0.0f;

    // Smoothed pitch for FillFeatures
    float smoothedPitchNorm_ = 0.0f;
};
