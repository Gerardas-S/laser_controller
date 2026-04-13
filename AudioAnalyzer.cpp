#include "AudioAnalyzer.h"
#include <iostream>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <chrono>

// AudioCapture delivers samples via lambda callback — no static PaCallback needed.

// ---------------------------------------------------------------------------

AudioAnalyzer::AudioAnalyzer() = default;

AudioAnalyzer::~AudioAnalyzer() { Stop(); }

void AudioAnalyzer::SetCallback(FeatureCallback cb)
{
    callback_ = std::move(cb);
}

void AudioAnalyzer::SetRawCallback(RawAudioCallback cb)
{
    rawCallback_ = std::move(cb);
}

// ---------------------------------------------------------------------------

bool AudioAnalyzer::Start(int sampleRate, int hopSize)
{
    if (running_) return true;

    sampleRate_ = sampleRate;
    hopSize_    = hopSize;

    // Ring buffer holds 2 seconds of audio
    ringBuffer_.assign(sampleRate_ * 2, 0.0f);
    writePos_ = 0;
    readPos_  = 0;

    // Build mel filterbank
    BuildMelFilterbank(melBands, fftSize, sampleRate_, 20.0f, 20000.0f);

    // Start PortAudio capture via AudioCapture lambda callback
    capture_ = std::make_unique<AudioCapture>();
    if (!capture_->Initialize()) {
        std::cerr << "[AudioAnalyzer] Failed to initialize audio capture\n";
        return false;
    }

    // Lambda pushes incoming samples into our ring buffer
    auto audioCb = [this](const float* buf, unsigned long count) {
        std::lock_guard<std::mutex> lock(ringMutex_);
        int wp = writePos_.load();
        int sz = static_cast<int>(ringBuffer_.size());
        for (unsigned long i = 0; i < count; ++i) {
            ringBuffer_[wp % sz] = buf[i];
            ++wp;
        }
        writePos_.store(wp);
    };

    if (!capture_->StartCapture(audioCb, sampleRate_, hopSize_)) {
        std::cerr << "[AudioAnalyzer] Failed to start audio capture\n";
        return false;
    }

    running_ = true;
    analysisThread_ = std::thread(&AudioAnalyzer::AnalysisLoop, this);
    std::cout << "[AudioAnalyzer] Started ("
              << sampleRate_ << " Hz, hop " << hopSize_ << ")\n";
    return true;
}

void AudioAnalyzer::Stop()
{
    running_ = false;
    if (analysisThread_.joinable()) analysisThread_.join();
    if (capture_) { capture_->StopCapture(); capture_.reset(); }
}

bool AudioAnalyzer::IsRunning() const { return running_; }

AudioFeatures AudioAnalyzer::GetLatestFeatures() const
{
    std::lock_guard<std::mutex> lock(featureMutex_);
    return latestFeatures_;
}

// ---------------------------------------------------------------------------
// Analysis loop
// ---------------------------------------------------------------------------

void AudioAnalyzer::AnalysisLoop()
{
    std::vector<float> window(fftSize, 0.0f);
    std::vector<float> magnitudes(fftSize / 2 + 1, 0.0f);

    // Hann window coefficients
    std::vector<float> hann(fftSize);
    for (int i = 0; i < fftSize; ++i)
        hann[i] = 0.5f * (1.0f - std::cos(2.0f * 3.14159265f * i / fftSize));

    while (running_) {
        // Wait until enough new samples are available
        int available = 0;
        while (running_) {
            int wp = writePos_.load();
            available = wp - readPos_;
            if (available >= hopSize_) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (!running_) break;

        elapsedSec_ += static_cast<double>(hopSize_) / sampleRate_;

        // Copy hop into analysis window (overlap-add style)
        {
            std::lock_guard<std::mutex> lock(ringMutex_);
            int sz = static_cast<int>(ringBuffer_.size());

            // Shift window left by hopSize
            for (int i = 0; i < fftSize - hopSize_; ++i)
                window[i] = window[i + hopSize_];

            // Fill new samples at end
            for (int i = 0; i < hopSize_; ++i)
                window[fftSize - hopSize_ + i] = ringBuffer_[(readPos_ + i) % sz];

            readPos_ += hopSize_;
        }

        // Apply Hann window
        std::vector<float> windowed(fftSize);
        for (int i = 0; i < fftSize; ++i)
            windowed[i] = window[i] * hann[i];

        // Compute FFT magnitude spectrum
        ComputeFFT(windowed, magnitudes);

        // Build features
        AudioFeatures feat;
        feat.timestampSec = elapsedSec_;

        // Time domain
        float sumSq = 0, peak = 0, zc = 0;
        for (int i = 0; i < fftSize; ++i) {
            sumSq += window[i] * window[i];
            if (std::abs(window[i]) > peak) peak = std::abs(window[i]);
            if (i > 0 && (window[i] >= 0) != (window[i-1] >= 0)) zc++;
        }
        feat.rms           = std::sqrt(sumSq / fftSize);
        feat.peakAmplitude = peak;
        feat.zeroCrossRate = zc / fftSize;

        ComputeMelBands(magnitudes, feat);
        ComputeSpectralFeatures(magnitudes, feat);
        ComputeBandEnergies(magnitudes, feat);
        ComputeChroma(magnitudes, feat);
        DetectBeat(feat, feat);
        SmoothFeatures(feat);

        {
            std::lock_guard<std::mutex> lock(featureMutex_);
            latestFeatures_ = feat;
        }
        // Fire raw audio callback (for PitchAnalyzer etc.) with latest hop
        if (rawCallback_) {
            // The current hop lives at the end of `window` (last hopSize_ samples)
            int offset = fftSize - hopSize_;
            rawCallback_(window.data() + offset, hopSize_, sampleRate_);
        }
        if (callback_) callback_(feat);
    }
}

// ---------------------------------------------------------------------------
// FFT
// ---------------------------------------------------------------------------

void AudioAnalyzer::ComputeFFT(const std::vector<float>& windowed,
                                 std::vector<float>& magnitudes)
{
    // Use pocketfft for real FFT
    pocketfft::shape_t shape = { static_cast<size_t>(fftSize) };
    pocketfft::stride_t stridein  = { sizeof(float) };
    pocketfft::stride_t strideout = { sizeof(std::complex<float>) };

    std::vector<std::complex<float>> spectrum(fftSize / 2 + 1);

    pocketfft::r2c<float>(shape, stridein, strideout,
                           { 0 }, true,
                           windowed.data(),
                           spectrum.data(), 1.0f);

    for (int i = 0; i <= fftSize / 2; ++i)
        magnitudes[i] = std::abs(spectrum[i]);
}

// ---------------------------------------------------------------------------
// Mel filterbank
// ---------------------------------------------------------------------------

float AudioAnalyzer::HzToMel(float hz)
{
    return 2595.0f * std::log10(1.0f + hz / 700.0f);
}

float AudioAnalyzer::MelToHz(float mel)
{
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

void AudioAnalyzer::BuildMelFilterbank(int numFilters, int fftSz,
                                         int sr, float fMin, float fMax)
{
    float melMin = HzToMel(fMin);
    float melMax = HzToMel(fMax);

    // numFilters + 2 equally spaced mel points
    std::vector<float> melPoints(numFilters + 2);
    for (int i = 0; i < numFilters + 2; ++i)
        melPoints[i] = melMin + (melMax - melMin) * i / (numFilters + 1);

    // Convert to FFT bin indices
    std::vector<int> binPoints(numFilters + 2);
    for (int i = 0; i < numFilters + 2; ++i) {
        float hz = MelToHz(melPoints[i]);
        binPoints[i] = static_cast<int>(std::floor((fftSz + 1) * hz / sr));
    }

    melFilterbank_.assign(numFilters, std::vector<float>(fftSz / 2 + 1, 0.0f));

    for (int m = 1; m <= numFilters; ++m) {
        int f_m_minus = binPoints[m - 1];
        int f_m       = binPoints[m];
        int f_m_plus  = binPoints[m + 1];

        for (int k = f_m_minus; k < f_m && k <= fftSz / 2; ++k) {
            if (f_m - f_m_minus > 0)
                melFilterbank_[m-1][k] = static_cast<float>(k - f_m_minus)
                                         / (f_m - f_m_minus);
        }
        for (int k = f_m; k <= f_m_plus && k <= fftSz / 2; ++k) {
            if (f_m_plus - f_m > 0)
                melFilterbank_[m-1][k] = static_cast<float>(f_m_plus - k)
                                         / (f_m_plus - f_m);
        }
    }
}

void AudioAnalyzer::ComputeMelBands(const std::vector<float>& magnitudes,
                                      AudioFeatures& out)
{
    float maxEnergy = 1e-10f;
    for (int m = 0; m < melBands; ++m) {
        float energy = 0.0f;
        for (int k = 0; k <= fftSize / 2; ++k)
            energy += melFilterbank_[m][k] * magnitudes[k];
        out.melBands[m] = energy;
        if (energy > maxEnergy) maxEnergy = energy;
    }
    // Normalise
    for (int m = 0; m < melBands; ++m)
        out.melBands[m] /= maxEnergy;
}

// ---------------------------------------------------------------------------
// Spectral features
// ---------------------------------------------------------------------------

void AudioAnalyzer::ComputeSpectralFeatures(const std::vector<float>& magnitudes,
                                              AudioFeatures& out)
{
    int N = fftSize / 2 + 1;
    float totalEnergy = 0, weightedFreq = 0;
    for (int k = 0; k < N; ++k) {
        float mag = magnitudes[k];
        totalEnergy  += mag;
        weightedFreq += mag * k;
    }

    // Centroid
    out.spectralCentroid = (totalEnergy > 1e-10f)
        ? (weightedFreq / totalEnergy) / (N - 1)
        : 0.0f;

    // Rolloff (85% energy)
    float cumEnergy = 0, rolloffThresh = 0.85f * totalEnergy;
    out.spectralRolloff = 0.0f;
    for (int k = 0; k < N; ++k) {
        cumEnergy += magnitudes[k];
        if (cumEnergy >= rolloffThresh) {
            out.spectralRolloff = static_cast<float>(k) / (N - 1);
            break;
        }
    }

    // Flux vs previous mel spectrum
    float flux = 0.0f;
    for (int m = 0; m < melBands; ++m) {
        float diff = out.melBands[m] - prevMel_[m];
        if (diff > 0) flux += diff;
    }
    out.spectralFlux = std::min(flux, 1.0f);
    prevMel_ = out.melBands;

    // Flatness (geometric mean / arithmetic mean of magnitude)
    float logSum = 0, sumMag = 0;
    int nonzero = 0;
    for (int k = 1; k < N; ++k) {
        float m = magnitudes[k];
        if (m > 1e-10f) { logSum += std::log(m); sumMag += m; ++nonzero; }
    }
    if (nonzero > 0 && sumMag > 1e-10f) {
        float geoMean = std::exp(logSum / nonzero);
        float ariMean = sumMag / nonzero;
        out.spectralFlatness = std::min(geoMean / ariMean, 1.0f);
    }
}

// ---------------------------------------------------------------------------
// Band energies
// ---------------------------------------------------------------------------

void AudioAnalyzer::ComputeBandEnergies(const std::vector<float>& magnitudes,
                                          AudioFeatures& out)
{
    auto bandEnergy = [&](float fLow, float fHigh) -> float {
        int lo = static_cast<int>(fLow  * fftSize / sampleRate_);
        int hi = static_cast<int>(fHigh * fftSize / sampleRate_);
        lo = std::clamp(lo, 0, fftSize / 2);
        hi = std::clamp(hi, 0, fftSize / 2);
        float e = 0.0f;
        for (int k = lo; k <= hi; ++k) e += magnitudes[k];
        return (hi > lo) ? e / (hi - lo) : 0.0f;
    };

    float maxE = 1e-10f;
    float raw[6] = {
        bandEnergy(20,   60),
        bandEnergy(60,   250),
        bandEnergy(250,  500),
        bandEnergy(500,  2000),
        bandEnergy(2000, 4000),
        bandEnergy(4000, 20000)
    };
    for (float v : raw) if (v > maxE) maxE = v;

    out.subBass    = raw[0] / maxE;
    out.bass       = raw[1] / maxE;
    out.midLow     = raw[2] / maxE;
    out.midHigh    = raw[3] / maxE;
    out.presence   = raw[4] / maxE;
    out.brilliance = raw[5] / maxE;
}

// ---------------------------------------------------------------------------
// Chroma (pitch-class profile)
// Maps each FFT bin to one of 12 semitone classes (A=0 … G#=11),
// accumulates magnitude, and normalises.
// Useful range: ~80 Hz – 4000 Hz (outside that, chroma is unreliable).
// ---------------------------------------------------------------------------

void AudioAnalyzer::ComputeChroma(const std::vector<float>& magnitudes,
                                    AudioFeatures& out)
{
    out.chroma.fill(0.0f);

    int N = fftSize / 2 + 1;
    for (int k = 1; k < N; ++k) {
        float hz = static_cast<float>(k) * sampleRate_ / fftSize;
        if (hz < 80.0f || hz > 4000.0f) continue;

        // MIDI note: A4 = 69, 440 Hz
        float midi = 12.0f * std::log2(hz / 440.0f) + 69.0f;
        int   pitchClass = static_cast<int>(std::round(midi)) % 12;
        if (pitchClass < 0) pitchClass += 12;

        out.chroma[pitchClass] += magnitudes[k];
    }

    // Normalise to [0,1]
    float maxC = *std::max_element(out.chroma.begin(), out.chroma.end());
    if (maxC > 1e-10f) {
        for (float& c : out.chroma) c /= maxC;
    }

    // Dominant pitch class
    out.dominantChroma = static_cast<int>(
        std::max_element(out.chroma.begin(), out.chroma.end()) - out.chroma.begin());
}

// ---------------------------------------------------------------------------
// Beat detection  (spectral flux onset detection + inter-onset BPM)
// ---------------------------------------------------------------------------

void AudioAnalyzer::DetectBeat(const AudioFeatures& current,
                                AudioFeatures& out)
{
    // Maintain rolling mean + std of flux for adaptive thresholding
    onsetHistory_.push_back(current.spectralFlux);
    if (static_cast<int>(onsetHistory_.size()) > 43) // ~1 sec at 512/44100
        onsetHistory_.pop_front();

    float mean = 0, sq = 0;
    for (float v : onsetHistory_) { mean += v; sq += v * v; }
    mean /= onsetHistory_.size();
    float stddev = std::sqrt(sq / onsetHistory_.size() - mean * mean);

    float threshold = mean + 1.5f * stddev + beatThreshold;

    out.onsetStrength = current.spectralFlux;
    out.isOnset = (current.spectralFlux > threshold &&
                   current.spectralFlux > prevFlux_ &&
                   (elapsedSec_ - lastBeatTime_) > 0.2);  // 300 BPM max

    prevFlux_ = current.spectralFlux;

    if (out.isOnset) {
        out.isBeat        = true;
        out.beatConfidence = std::min((current.spectralFlux - threshold) * 2.0f, 1.0f);
        lastBeatTime_      = elapsedSec_;

        beatTimes_.push_back(elapsedSec_);
        // Keep only last bpmEstWindow seconds
        while (!beatTimes_.empty() &&
               elapsedSec_ - beatTimes_.front() > bpmEstWindow)
            beatTimes_.pop_front();

        // BPM from inter-onset intervals
        if (beatTimes_.size() >= 2) {
            double totalInterval = beatTimes_.back() - beatTimes_.front();
            double avgInterval   = totalInterval / (beatTimes_.size() - 1);
            out.bpm = static_cast<float>(60.0 / avgInterval);
        }
    } else {
        out.isBeat        = false;
        out.beatConfidence = 0.0f;
    }

    // Beat phase (0→1 ramp since last beat)
    if (lastBeatTime_ > 0 && out.bpm > 0) {
        double beatPeriod = 60.0 / out.bpm;
        out.beatPhase = static_cast<float>(
            std::fmod(elapsedSec_ - lastBeatTime_, beatPeriod) / beatPeriod);
    }
}

// ---------------------------------------------------------------------------
// Exponential smoothing
// ---------------------------------------------------------------------------

void AudioAnalyzer::SmoothFeatures(AudioFeatures& f)
{
    float a = smoothingAlpha;
    float b = 1.0f - a;

    f.rmsSmooth              = a * f.rms              + b * smoothed_.rmsSmooth;
    f.bassSmooth             = a * f.bass             + b * smoothed_.bassSmooth;
    f.spectralCentroidSmooth = a * f.spectralCentroid + b * smoothed_.spectralCentroidSmooth;

    // Carry forward BPM — only update when a beat is detected (avoid zeroing out)
    if (f.bpm < 20.0f && smoothed_.bpm > 20.0f)
        f.bpm = smoothed_.bpm;

    // Smooth chroma with slower alpha (it changes slowly with harmony)
    float ca = smoothingAlpha * 0.3f;
    float cb = 1.0f - ca;
    for (int i = 0; i < 12; ++i)
        f.chroma[i] = ca * f.chroma[i] + cb * smoothed_.chroma[i];

    smoothed_ = f;
}
