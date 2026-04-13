#pragma once

#include "GeometryPoint3D.h"
#include "GeometryShapes.h"
#include "AudioFeatures.h"
#include <functional>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <cmath>

// -----------------------------------------------------------------------------
// MusicGeometryEngine
//
// Maps AudioFeatures to Frame3D every render tick.
// Supports multiple named "modes" (scenes), each driven differently by audio.
// Modes can be switched at runtime.
//
// Usage:
//   MusicGeometryEngine engine;
//   engine.SetMode("lissajous");
//   Frame3D frame = engine.Generate(features, timeSec);
//   laser.SendFrame(Frame3DToLaser(frame));
// -----------------------------------------------------------------------------

class MusicGeometryEngine {
public:
    MusicGeometryEngine();

    // Generate a frame from current audio features.
    Frame3D Generate(const AudioFeatures& feat, double timeSec);

    // Switch active mode by name.
    void SetMode(const std::string& name);
    std::string GetMode() const { return currentMode_; }
    std::vector<std::string> GetModeNames() const;

    // Point budget hint (engine tries to stay within this).
    int targetPoints = 800;

private:
    // Each mode is a function: (features, time) -> Frame3D
    using ModeFunc = std::function<Frame3D(const AudioFeatures&, double)>;
    std::unordered_map<std::string, ModeFunc> modes_;
    std::string currentMode_ = "lissajous";

    // Per-mode state
    double beatPhaseContinuous_ = 0.0;
    double lastBeatTime_        = 0.0;
    float  lissA_ = 3.0f, lissB_ = 2.0f, lissC_ = 0.0f;
    float  lissATarget_ = 3.0f, lissBTarget_ = 2.0f;
    int    knotP_ = 2, knotQ_ = 3;
    double modeTime_ = 0.0;

    // Melody mode state
    float  melodyPitchSmooth_   = 0.0f;  // smoothed normalised pitch [0,1]
    float  melodyRadiusSmooth_  = 0.5f;
    float  melodyLastHz_        = 0.0f;
    int    melodyOctave_        = 4;

    // Colour helpers
    static void HsvToRgb(float h, float s, float v,
                          float& r, float& g, float& b);
    static float Lerp(float a, float b, float t) { return a + (b - a) * t; }

    // Mode generators
    Frame3D ModeLissajous    (const AudioFeatures& f, double t);
    Frame3D ModeTorusKnot    (const AudioFeatures& f, double t);
    Frame3D ModeSpectrum     (const AudioFeatures& f, double t);
    Frame3D ModeSpiral       (const AudioFeatures& f, double t);
    Frame3D ModeRose         (const AudioFeatures& f, double t);
    Frame3D ModeHypotrochoid (const AudioFeatures& f, double t);
    Frame3D ModePulsar       (const AudioFeatures& f, double t);
    Frame3D ModeWaveform     (const AudioFeatures& f, double t);
    Frame3D ModeTerrain      (const AudioFeatures& f, double t);
    Frame3D ModeMelody       (const AudioFeatures& f, double t);
    Frame3D ModeChroma       (const AudioFeatures& f, double t);
};
