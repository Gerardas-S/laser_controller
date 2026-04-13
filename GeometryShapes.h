#pragma once

#include "GeometryPoint3D.h"
#include "AudioFeatures.h"
#include <cmath>
#include <vector>
#include <numbers>

// -----------------------------------------------------------------------------
// GeometryShapes
//
// Static factory functions for 3D laser-friendly primitives.
// All shapes are parameterised — callers pass time/phase/audio values.
// All coordinates are normalised to [-1, 1].
// -----------------------------------------------------------------------------

namespace GeometryShapes {

static constexpr float PI  = 3.14159265358979323846f;
static constexpr float TAU = 6.28318530717958647692f;

// ---------------------------------------------------------------------------
// Lissajous figure (2D/3D)
// a, b, c = frequency ratios per axis
// delta, gamma = phase offsets
// ---------------------------------------------------------------------------
inline Polyline3D Lissajous(int points, float a, float b, float c,
                             float delta, float gamma,
                             float r, float g, float bl, float z = 0.0f)
{
    Polyline3D poly;
    poly.reserve(points + 1);
    for (int i = 0; i <= points; ++i) {
        float t = TAU * i / points;
        float x = std::sin(a * t + delta);
        float y = std::sin(b * t);
        float zv = (c > 0.0f) ? std::sin(c * t + gamma) : z;
        poly.push_back({ x, y, zv, r, g, bl });
    }
    return poly;
}

// ---------------------------------------------------------------------------
// Archimedean spiral (flat, in XY)
// ---------------------------------------------------------------------------
inline Polyline3D Spiral(int points, float turns, float innerR, float outerR,
                          float z, float r, float g, float bl)
{
    Polyline3D poly;
    poly.reserve(points);
    for (int i = 0; i < points; ++i) {
        float t     = static_cast<float>(i) / points;
        float angle = TAU * turns * t;
        float radius = innerR + (outerR - innerR) * t;
        poly.push_back({
            radius * std::cos(angle),
            radius * std::sin(angle),
            z, r, g, bl
        });
    }
    return poly;
}

// ---------------------------------------------------------------------------
// Toroidal helix
// R = major radius, r = minor radius, n = windings
// ---------------------------------------------------------------------------
inline Polyline3D ToroidalHelix(int points, float R, float rr, int n,
                                  float r, float g, float bl)
{
    Polyline3D poly;
    poly.reserve(points);
    for (int i = 0; i < points; ++i) {
        float t     = TAU * i / points;
        float phi   = t;
        float theta = n * t;
        float x = (R + rr * std::cos(theta)) * std::cos(phi);
        float y = (R + rr * std::cos(theta)) * std::sin(phi);
        float z = rr * std::sin(theta);
        poly.push_back({ x, y, z, r, g, bl });
    }
    return poly;
}

// ---------------------------------------------------------------------------
// Rose curve r = cos(k*theta)
// ---------------------------------------------------------------------------
inline Polyline3D Rose(int points, float k, float z,
                        float r, float g, float bl)
{
    Polyline3D poly;
    poly.reserve(points);
    int periods = (static_cast<int>(k) % 2 == 0) ? 2 : 1;
    for (int i = 0; i < points; ++i) {
        float t     = PI * periods * i / points;
        float radius = std::cos(k * t);
        poly.push_back({
            radius * std::cos(t),
            radius * std::sin(t),
            z, r, g, bl
        });
    }
    return poly;
}

// ---------------------------------------------------------------------------
// Hypotrochoid (Spirograph)
// R = outer circle radius, ri = inner circle radius, d = pen distance
// ---------------------------------------------------------------------------
inline Polyline3D Hypotrochoid(int points, float R, float ri, float d,
                                 float z, float r, float g, float bl)
{
    Polyline3D poly;
    poly.reserve(points);
    float scale = 1.0f / (R);
    for (int i = 0; i < points; ++i) {
        float t = TAU * i / points;
        float x = ((R - ri) * std::cos(t) + d * std::cos((R - ri) / ri * t)) * scale;
        float y = ((R - ri) * std::sin(t) - d * std::sin((R - ri) / ri * t)) * scale;
        poly.push_back({ x, y, z, r, g, bl });
    }
    return poly;
}

// ---------------------------------------------------------------------------
// Superellipse / Lamé curve
// n = 2 → ellipse, n < 2 → astroid-like, n > 2 → rounded rect
// ---------------------------------------------------------------------------
inline Polyline3D Superellipse(int points, float a, float b, float n,
                                 float z, float r, float g, float bl)
{
    Polyline3D poly;
    poly.reserve(points + 1);
    for (int i = 0; i <= points; ++i) {
        float t    = TAU * i / points;
        float cosT = std::cos(t);
        float sinT = std::sin(t);
        float x = a * std::copysign(std::pow(std::abs(cosT), 2.0f / n), cosT);
        float y = b * std::copysign(std::pow(std::abs(sinT), 2.0f / n), sinT);
        poly.push_back({ x, y, z, r, g, bl });
    }
    return poly;
}

// ---------------------------------------------------------------------------
// 3D Knot — (2,3) torus knot by default
// p, q = winding numbers
// ---------------------------------------------------------------------------
inline Polyline3D TorusKnot(int points, int p, int q, float R, float rr,
                              float r, float g, float bl)
{
    Polyline3D poly;
    poly.reserve(points);
    for (int i = 0; i < points; ++i) {
        float t   = TAU * i / points;
        float phi = t;
        float x = (R + rr * std::cos(q * phi)) * std::cos(p * phi);
        float y = (R + rr * std::cos(q * phi)) * std::sin(p * phi);
        float z = rr * std::sin(q * phi);
        float scale = 1.0f / (R + rr);
        poly.push_back({ x * scale, y * scale, z * scale, r, g, bl });
    }
    return poly;
}

// ---------------------------------------------------------------------------
// Waveform polyline from raw audio samples
// ---------------------------------------------------------------------------
inline Polyline3D Waveform(const std::vector<float>& samples,
                             float z, float r, float g, float bl)
{
    if (samples.empty()) return {};
    Polyline3D poly;
    poly.reserve(samples.size());
    int N = static_cast<int>(samples.size());
    for (int i = 0; i < N; ++i) {
        float x = 2.0f * i / (N - 1) - 1.0f;
        float y = std::clamp(samples[i], -1.0f, 1.0f);
        poly.push_back({ x, y, z, r, g, bl });
    }
    return poly;
}

// ---------------------------------------------------------------------------
// Spectrum bar chart as vertical lines (each mel band = one polyline)
// ---------------------------------------------------------------------------
inline Frame3D SpectrumBars(const AudioFeatures& feat,
                              float z, float barR, float barG, float barB)
{
    Frame3D frame;
    int N = AudioFeatures::kMelBands;
    for (int i = 0; i < N; ++i) {
        float x   = 2.0f * i / (N - 1) - 1.0f;
        float top = feat.melBands[i] * 2.0f - 1.0f;  // remap [0,1] -> [-1,1]
        Polyline3D bar;
        bar.push_back({ x, -1.0f, z, barR, barG, barB });
        bar.push_back({ x, top,   z, barR, barG, barB });
        frame.push_back(std::move(bar));
    }
    return frame;
}

// ---------------------------------------------------------------------------
// Parametric surface patch (sampled as polylines for galvo)
// fn: (u, v) -> GeometryPoint3D
// ---------------------------------------------------------------------------
template<typename Fn>
inline Frame3D ParametricSurface(int uSteps, int vSteps, Fn fn)
{
    Frame3D frame;
    // Sample as uSteps iso-u lines
    for (int ui = 0; ui <= uSteps; ++ui) {
        float u = static_cast<float>(ui) / uSteps;
        Polyline3D poly;
        for (int vi = 0; vi <= vSteps; ++vi) {
            float v = static_cast<float>(vi) / vSteps;
            poly.push_back(fn(u, v));
        }
        frame.push_back(std::move(poly));
    }
    return frame;
}

} // namespace GeometryShapes
