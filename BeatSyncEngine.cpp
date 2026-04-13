#include "BeatSyncEngine.h"
#include <algorithm>
#include <cmath>

static constexpr float PI  = 3.14159265f;
static constexpr float TAU = 6.28318530f;

// ---------------------------------------------------------------------------
// Update — advance all animated state
// ---------------------------------------------------------------------------

void BeatSyncEngine::Update(const AudioFeatures& feat, float dt)
{
    // Smooth BPM
    if (feat.bpm > 20.0f && feat.bpm < 300.0f)
        bpmSmooth_ = bpmSmooth_ * 0.98f + feat.bpm * 0.02f;

    // --- Scale ---
    float targetScale = scaleBaseMin + feat.rmsSmooth * (scaleBaseMax - scaleBaseMin);
    if (feat.isBeat)
        targetScale = std::min(targetScale + beatScaleBoost * feat.beatConfidence, 1.3f);

    scaleTarget_ = targetScale;
    // Asymmetric smoothing: fast attack, slow decay
    float alpha = (scaleTarget_ > scaleSmooth_) ? 0.4f : 0.05f;
    scaleSmooth_ = scaleSmooth_ + alpha * (scaleTarget_ - scaleSmooth_);
    currentScale = scaleSmooth_;

    // --- Rotation ---
    if (enableRotation) {
        float rpsTarget = (bpmSmooth_ / 60.0f) * rotationSpeed * 0.25f;
        rotAngle_ += TAU * rpsTarget * dt;
        if (rotAngle_ > TAU) rotAngle_ -= TAU;
    }
    currentRotAngle = rotAngle_;

    // --- Z breathe ---
    if (enableZBreathe) {
        float breatheRate = bpmSmooth_ / 60.0f * 0.5f;  // half BPM rate
        zPhase_ += TAU * breatheRate * dt;
        currentZ = std::sin(zPhase_) * zBreatheDepth * feat.rmsSmooth;
    }

    // --- Hue shift ---
    if (enableHueShift) {
        float hueRate = feat.spectralCentroidSmooth * 0.1f + 0.01f;
        hueOffset_ += hueRate * dt;
        if (hueOffset_ > 1.0f) hueOffset_ -= 1.0f;
    }
    currentHue = hueOffset_;
}

// ---------------------------------------------------------------------------
// Apply — transform a frame
// ---------------------------------------------------------------------------

Frame3D BeatSyncEngine::Apply(const Frame3D& frame) const
{
    float cosA = std::cos(currentRotAngle);
    float sinA = std::sin(currentRotAngle);

    Frame3D out;
    out.reserve(frame.size());

    for (const auto& poly : frame) {
        Polyline3D outPoly;
        outPoly.reserve(poly.size());

        for (const auto& p : poly) {
            GeometryPoint3D q = p;

            // Scale
            if (enableScale) {
                q.x *= currentScale;
                q.y *= currentScale;
                q.z *= currentScale;
            }

            // Rotation (around Z axis)
            if (enableRotation) {
                float nx = q.x * cosA - q.y * sinA;
                float ny = q.x * sinA + q.y * cosA;
                q.x = nx; q.y = ny;
            }

            // Z breathe (additive)
            if (enableZBreathe)
                q.z = std::clamp(q.z + currentZ, -1.0f, 1.0f);

            // Hue shift
            if (enableHueShift)
                HsvShift(q.r, q.g, q.b, currentHue);

            outPoly.push_back(q);
        }

        out.push_back(std::move(outPoly));
    }

    return out;
}

// ---------------------------------------------------------------------------
// RGB <-> HSV hue shift
// ---------------------------------------------------------------------------

void BeatSyncEngine::HsvShift(float& r, float& g, float& b, float hueShift)
{
    // RGB -> HSV
    float maxC = std::max({r, g, b});
    float minC = std::min({r, g, b});
    float delta = maxC - minC;

    float h = 0.0f, s = 0.0f, v = maxC;
    if (delta > 1e-5f) {
        s = delta / maxC;
        if      (r >= maxC) h = (g - b) / delta;
        else if (g >= maxC) h = 2.0f + (b - r) / delta;
        else                h = 4.0f + (r - g) / delta;
        h /= 6.0f;
        if (h < 0) h += 1.0f;
    }

    // Shift hue
    h = std::fmod(h + hueShift, 1.0f);

    // HSV -> RGB
    if (s < 1e-5f) { r = g = b = v; return; }
    float hh = h * 6.0f;
    int   i  = static_cast<int>(hh);
    float f  = hh - i;
    float p  = v * (1 - s);
    float q2 = v * (1 - s * f);
    float t2 = v * (1 - s * (1 - f));
    switch (i % 6) {
        case 0: r=v;  g=t2; b=p;  break;
        case 1: r=q2; g=v;  b=p;  break;
        case 2: r=p;  g=v;  b=t2; break;
        case 3: r=p;  g=q2; b=v;  break;
        case 4: r=t2; g=p;  b=v;  break;
        default:r=v;  g=p;  b=q2; break;
    }
}
