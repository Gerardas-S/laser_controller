#pragma once

#include "GeometryPoint3D.h"
#include "AudioFeatures.h"
#include <cmath>
#include <functional>

// -----------------------------------------------------------------------------
// BeatSyncEngine
//
// Sits between MusicGeometryEngine output and laser output.
// Applies beat-reactive transforms to a Frame3D:
//   - Scale pulse on beat
//   - Continuous rotation driven by BPM
//   - Colour hue rotation
//   - Z breathe (depth oscillation)
//   - Flash/strobe on strong beat
//
// All transforms are smoothed to avoid galvo shock.
// Usage:
//   BeatSyncEngine sync;
//   sync.Update(features, dt);
//   Frame3D out = sync.Apply(rawFrame);
// -----------------------------------------------------------------------------

class BeatSyncEngine {
public:
    BeatSyncEngine() = default;

    // Call once per render tick. dt = seconds since last call.
    void Update(const AudioFeatures& feat, float dt);

    // Apply current transforms to a frame.
    Frame3D Apply(const Frame3D& frame) const;

    // --- Controls (tweak at runtime) ---
    bool  enableRotation   = true;
    bool  enableScale      = true;
    bool  enableZBreathe   = true;
    bool  enableHueShift   = true;
    float rotationSpeed    = 1.0f;   // multiplier on BPM-driven rotation
    float scaleBaseMin     = 0.5f;   // minimum scale
    float scaleBaseMax     = 1.0f;   // scale at max RMS
    float beatScaleBoost   = 0.25f;  // extra scale punch on beat
    float zBreatheDepth    = 0.3f;   // Z oscillation amplitude

    // Read-only current state
    float currentScale     = 1.0f;
    float currentRotAngle  = 0.0f;
    float currentHue       = 0.0f;
    float currentZ         = 0.0f;

private:
    float scaleTarget_  = 1.0f;
    float scaleSmooth_  = 1.0f;
    float rotAngle_     = 0.0f;
    float hueOffset_    = 0.0f;
    float zPhase_       = 0.0f;
    float bpmSmooth_    = 120.0f;

    static void HsvShift(float& r, float& g, float& b, float hueShift);
};
