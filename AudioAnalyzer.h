#pragma once

#include "AudioFeatures.h"
#include "AudioPipeline.h"   // reuse AudioCapture + AudioPreprocessor
#include "pocketfft.h"
#include <vector>
#include <array>
#include <deque>
#include <memory>
#include <mutex>
#include <atomic>
#include <thread>
#include <functional>

// -----------------------------------------------------------------------------
// AudioAnalyzer
//
// Wraps AudioCapture (PortAudio) and computes AudioFeatures in real time.
// Runs its own analysis thread; delivers results via callback.
//
// Usage:
//   AudioAnalyzer analyzer;
//   analyzer.SetCallback([](const AudioFeatures& f) { /* use f */ });
//   analyzer.Start();
//   ...
//   analyzer.Stop();
// -----------------------------------------------------------------------------

class AudioAnalyzer {
public:
    using FeatureCallback = std::function<void(const AudioFeatures&)>;
    // Raw audio callback: fired with the latest hop of mono float samples.
    // Useful for feeding PitchAnalyzer or other raw-signal processors.
    using RawAudioCallback = std::function<void(const float* samples, int count, int sampleRate)>;

    AudioAnalyzer();
    ~AudioAnalyzer();

    // Register callbacks — both called from the analysis thread each hop.
    void SetCallback(FeatureCallback cb);
    void SetRawCallback(RawAudioCallback cb);

    // Start/stop audio capture and analysis.
    bool Start(int sampleRate = 44100, int hopSize = 512);
    void Stop();
    bool IsRunning() const;

    // Get latest features (thread-safe, non-blocking).
    AudioFeatures GetLatestFeatures() const;

    // Analysis parameters (set before Start).
    int   fftSize       = 2048;
    int   melBands      = AudioFeatures::kMelBands;
    float smoothingAlpha = 0.2f;   // exponential smoothing [0=slow, 1=instant]
    float beatThreshold  = 0.3f;   // onset flux threshold for beat detection
    float bpmEstWindow   = 4.0f;   // seconds of history for BPM estimation

private:
    void AnalysisLoop();

    // DSP helpers
    void ComputeFFT(const std::vector<float>& window,
                    std::vector<float>& magnitudes);
    void ComputeMelBands(const std::vector<float>& magnitudes,
                         AudioFeatures& out);
    void ComputeSpectralFeatures(const std::vector<float>& magnitudes,
                                  AudioFeatures& out);
    void ComputeBandEnergies(const std::vector<float>& magnitudes,
                              AudioFeatures& out);
    void ComputeChroma(const std::vector<float>& magnitudes,
                       AudioFeatures& out);
    void DetectBeat(const AudioFeatures& current, AudioFeatures& out);
    void SmoothFeatures(AudioFeatures& current);

    // Mel filterbank builder
    void BuildMelFilterbank(int numFilters, int fftSize, int sampleRate,
                             float fMin, float fMax);
    static float HzToMel(float hz);
    static float MelToHz(float mel);

    // Audio capture
    std::unique_ptr<AudioCapture> capture_;
    int sampleRate_ = 44100;
    int hopSize_    = 512;

    // Ring buffer for incoming audio
    std::vector<float>  ringBuffer_;
    std::atomic<int>    writePos_{ 0 };
    int                 readPos_ = 0;
    mutable std::mutex  ringMutex_;

    // Mel filterbank [numFilters][fftBins]
    std::vector<std::vector<float>> melFilterbank_;

    // Beat detection state
    std::deque<float>   onsetHistory_;   // recent onset strengths
    std::deque<double>  beatTimes_;      // timestamps of recent beats
    float               prevFlux_ = 0.0f;
    double              lastBeatTime_ = 0.0;
    double              elapsedSec_   = 0.0;

    // Previous mel spectrum for flux
    std::array<float, AudioFeatures::kMelBands> prevMel_{};

    // Previous features for smoothing
    AudioFeatures smoothed_;

    // Analysis thread
    std::thread          analysisThread_;
    std::atomic<bool>    running_{ false };
    FeatureCallback      callback_;
    RawAudioCallback     rawCallback_;
    mutable std::mutex   featureMutex_;
    AudioFeatures        latestFeatures_;
};
