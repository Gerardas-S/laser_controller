#pragma once
#include "portaudio.h"
#include <sndfile.h>
#include <functional>
#include <vector>
#include <string>
#include <memory>

std::vector<float> LoadAudioFile(const std::string& path, int& outSampleRate);

// -----------------------------------------------------------------------------
// PreprocessorConfig
// ---

class PreprocessorConfig {
public:
    int sampling_rate = 48000;
    int n_fft = 1024;
    int hop_length = 480;
    int feature_size = 64;  // n_mels
    int frequency_min = 50;
    int frequency_max = 14000;
    int nb_max_samples = 480000;
    int chunk_length_s = 10;

    static PreprocessorConfig LoadFromFile(const std::string& path);

};

// -----------------------------------------------------------------------------
// AudioCapture
// -----------------------------------------------------------------------------


class AudioCapture
{
public:
    // Callback type for processing audio data
    using AudioCallback = std::function<void(const float* buffer, unsigned long frameCount)>;

    AudioCapture();
    ~AudioCapture();

    // Initialize audio system
    bool Initialize();

    // Start capturing audio with a callback
    bool StartCapture(AudioCallback callback, int sampleRate = 44100, int framesPerBuffer = 512);

    // Stop capturing
    void StopCapture();

    // Get device info
    const char* GetDeviceName() const;
    double GetSampleRate() const;

    // Check if currently capturing
    bool IsCapturing() const;

private:
    PaStream* stream_;
    PaDeviceIndex deviceIndex_;
    const PaDeviceInfo* deviceInfo_;
    AudioCallback userCallback_;
    bool isInitialized_;
    bool isCapturing_;

    // Internal PortAudio callback (static)
    static int paCallback(const void* inputBuffer, void* outputBuffer,
        unsigned long framesPerBuffer,
        const PaStreamCallbackTimeInfo* timeInfo,
        PaStreamCallbackFlags statusFlags,
        void* userData);
};



namespace pocketfft {
    namespace detail {
        template<typename T> class pocketfft_r;
    }
}

// -----------------------------------------------------------------------------
// AudioPreprocessor
// -----------------------------------------------------------------------------


class AudioPreprocessor {
public:
    explicit AudioPreprocessor(const PreprocessorConfig& config);
    ~AudioPreprocessor();

    std::vector<float> Process(const float* audio, size_t sample_count, int original_sample_rate);
    size_t GetNumMelBins() const { return config_.feature_size; }
    size_t GetNumTimeFrames(size_t num_samples) const;

private:
    PreprocessorConfig config_;
    std::vector<float> mel_filterbank_;
    std::vector<float> hann_window_;
    std::vector<float> fft_buffer_;  // Reusable buffer for FFT operations

    std::unique_ptr<pocketfft::detail::pocketfft_r<float>> rfft_plan_;

    void InitMelFilterbank();
    void InitHannWindow();

    std::vector<float> Resample(const float* audio, size_t sample_count,
        int original_sr, int target_sr);
    std::vector<float> ResampleFast_44100_to_48000(const float* audio, size_t sample_count);

    std::vector<std::vector<float>> ComputeSTFT(const std::vector<float>& audio);
    std::vector<float> ApplyMelFilterbank(const std::vector<std::vector<float>>& spectrogram);

    float HzToMel(float hz) const;
    float MelToHz(float mel) const;
};