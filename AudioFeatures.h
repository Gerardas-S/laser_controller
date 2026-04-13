#pragma once
#include <vector>
#include <array>
#include <cstdint>

// -----------------------------------------------------------------------------
// AudioFeatures
//
// Plain data struct carrying all real-time audio analysis results.
// Produced by AudioAnalyzer, consumed by MusicGeometryEngine.
// All values are normalised to [0,1] unless noted otherwise.
// -----------------------------------------------------------------------------

struct AudioFeatures {

    // --- Time domain ---
    float rms           = 0.0f;   // Root mean square energy [0,1]
    float peakAmplitude = 0.0f;   // Peak sample amplitude  [0,1]
    float zeroCrossRate = 0.0f;   // Zero crossing rate     [0,1]

    // --- Frequency domain (32 mel bands, normalised) ---
    static constexpr int kMelBands = 32;
    std::array<float, kMelBands> melBands{};

    // --- Spectral features ---
    float spectralCentroid  = 0.0f;  // "brightness" [0,1], 0=low, 1=high freq
    float spectralFlux      = 0.0f;  // frame-to-frame change [0,1]
    float spectralRolloff   = 0.0f;  // frequency below which 85% energy lies [0,1]
    float spectralFlatness  = 0.0f;  // noise vs tone [0=tonal, 1=noisy]

    // --- Band energies (sub, bass, mid, high, air) ---
    float subBass   = 0.0f;   // 20–60 Hz
    float bass      = 0.0f;   // 60–250 Hz
    float midLow    = 0.0f;   // 250–500 Hz
    float midHigh   = 0.0f;   // 500–2000 Hz
    float presence  = 0.0f;   // 2000–4000 Hz
    float brilliance= 0.0f;   // 4000–20000 Hz

    // --- Beat / rhythm ---
    float beatConfidence = 0.0f;  // [0,1] how confident we are a beat occurred
    float bpm            = 0.0f;  // estimated tempo in BPM (0 = unknown)
    float beatPhase      = 0.0f;  // [0,1] position within current beat period
    bool  isBeat         = false; // true on beat onset frame

    // --- Onset detection ---
    float onsetStrength  = 0.0f;  // [0,1] strength of detected onset
    bool  isOnset        = false; // true when onset detected this frame

    // --- Smoothed versions (for continuous animation) ---
    float rmsSmooth          = 0.0f;
    float bassSmooth         = 0.0f;
    float spectralCentroidSmooth = 0.0f;

    // --- Pitch / melody (filled by PitchAnalyzer when loaded) ---
    // Fundamental frequency in Hz. 0.0 = unvoiced / silence.
    float pitch           = 0.0f;   // Hz, range ~32–2093 Hz (C1–C7)
    float pitchConfidence = 0.0f;   // CREPE confidence [0,1]
    float pitchNorm       = 0.0f;   // Pitch mapped to [0,1] on log scale (C1=0, C8=1)
    float pitchSmooth     = 0.0f;   // Exponentially smoothed pitchNorm

    // --- Chroma (12 semitones, A=0, computed from FFT) ---
    std::array<float, 12> chroma{};   // energy per pitch class [0,1]
    int   dominantChroma  = 0;        // index of strongest chroma bin [0..11]

    // --- CLAP embedding (optional, filled by CLAPInference) ---
    // 512-dim audio embedding — can drive more abstract geometry mappings
    std::vector<float> clapEmbedding;

    // --- Timestamp ---
    double timestampSec = 0.0;
};
