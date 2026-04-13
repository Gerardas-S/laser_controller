#pragma once
#include <vector>
#include <cmath>
#include "HeliosOutput.h"

// -----------------------------------------------------------------------------
// GeometryPoint3D
//
// A 3D point with color. X, Y, Z are all normalized -1.0 to 1.0.
// Z maps to HeliosPointExt::user4 (0x0000 = back, 0x8000 = center, 0xFFFF = front)
// when output via HeliosOutput. For 2D DACs Z is simply ignored.
// -----------------------------------------------------------------------------

struct GeometryPoint3D {
    float x, y, z;       // -1.0 to 1.0
    float r, g, b;       // 0.0 to 1.0

    GeometryPoint3D() : x(0), y(0), z(0), r(1), g(1), b(1) {}
    GeometryPoint3D(float x, float y, float z, float r=1, float g=1, float b=1)
        : x(x), y(y), z(z), r(r), g(g), b(b) {}
};

// A polyline in 3D space
using Polyline3D = std::vector<GeometryPoint3D>;

// A full frame — multiple polylines
using Frame3D = std::vector<Polyline3D>;

// -----------------------------------------------------------------------------
// Conversion: Frame3D -> Frame (2D LaserPoint, Z discarded for standard DAC)
// For extended DAC support, Z is packed into user4.
// -----------------------------------------------------------------------------

inline std::vector<std::vector<LaserPoint>> Frame3DToLaser(const Frame3D& frame3d)
{
    std::vector<std::vector<LaserPoint>> result;
    result.reserve(frame3d.size());

    for (const auto& poly3d : frame3d) {
        std::vector<LaserPoint> poly2d;
        poly2d.reserve(poly3d.size());
        for (const auto& p : poly3d) {
            // Project: perspective divide by (1 + z * 0.3) for mild depth effect
            float depth = 1.0f + p.z * 0.3f;
            float px = (depth > 0.01f) ? p.x / depth : p.x;
            float py = (depth > 0.01f) ? p.y / depth : p.y;
            poly2d.push_back({ px, py, p.r, p.g, p.b });
        }
        result.push_back(std::move(poly2d));
    }
    return result;
}
