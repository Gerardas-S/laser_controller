#include "ILDAFile.h"
#include <fstream>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <cmath>

// ---------------------------------------------------------------------------
// Byte helpers
// ---------------------------------------------------------------------------

uint16_t ILDAFile::ToBE(uint16_t v)
{
    return (v >> 8) | (v << 8);
}

uint16_t ILDAFile::FromBE(uint16_t v)
{
    return (v >> 8) | (v << 8);
}

int16_t ILDAFile::CoordToILDA(float v)
{
    float clamped = std::max(-1.0f, std::min(1.0f, v));
    return static_cast<int16_t>(clamped * 32767.0f);
}

float ILDAFile::ILDAToCoord(int16_t v)
{
    return static_cast<float>(v) / 32767.0f;
}

uint8_t ILDAFile::FloatToByte(float v)
{
    return static_cast<uint8_t>(std::max(0.0f, std::min(1.0f, v)) * 255.0f);
}

float ILDAFile::ByteToFloat(uint8_t v)
{
    return static_cast<float>(v) / 255.0f;
}

// ---------------------------------------------------------------------------
// Save
// ---------------------------------------------------------------------------

bool ILDAFile::Save(const std::string& path,
                    const std::vector<Frame>& animation,
                    const std::string& name)
{
    if (animation.empty()) {
        std::cerr << "[ILDA] Nothing to save.\n";
        return false;
    }

    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "[ILDA] Cannot open for writing: " << path << "\n";
        return false;
    }

    uint16_t totalFrames = static_cast<uint16_t>(
        std::min(animation.size(), (size_t)65535));

    for (uint16_t fi = 0; fi < totalFrames; ++fi) {
        const Frame& frame = animation[fi];

        // Count total points across all polylines
        uint32_t totalPts = 0;
        for (const auto& poly : frame)
            totalPts += static_cast<uint32_t>(poly.size());

        if (totalPts == 0) continue;
        if (totalPts > 65535) totalPts = 65535;  // ILDA cap per frame

        // Write header
        ILDAHeader hdr{};
        std::memcpy(hdr.sig, "ILDA", 4);
        hdr.format      = 5;
        hdr.numRecords  = ToBE(static_cast<uint16_t>(totalPts));
        hdr.frameNumber = ToBE(fi);
        hdr.totalFrames = ToBE(totalFrames);

        // Null-padded name fields
        std::memset(hdr.name,    0, 8);
        std::memset(hdr.company, 0, 8);
        std::memcpy(hdr.name,    name.c_str(),
                    std::min(name.size(), (size_t)8));
        std::memcpy(hdr.company, "laserctrl",
                    std::min((size_t)8, (size_t)8));

        f.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));

        // Write points
        uint32_t written = 0;
        for (const auto& poly : frame) {
            for (size_t pi = 0; pi < poly.size() && written < totalPts; ++pi, ++written) {
                const LaserPoint& lp = poly[pi];
                ILDAPoint5 pt{};
                pt.x = ToBE(static_cast<uint16_t>(CoordToILDA(lp.x)));
                pt.y = ToBE(static_cast<uint16_t>(CoordToILDA(lp.y)));
                pt.status = (pi == 0) ? 0x40 : 0x00;  // bit6=blank on first point
                pt.b = FloatToByte(lp.b);
                pt.g = FloatToByte(lp.g);
                pt.r = FloatToByte(lp.r);
                f.write(reinterpret_cast<const char*>(&pt), sizeof(pt));
            }
        }
    }

    // EOF section — header with numRecords = 0
    ILDAHeader eof{};
    std::memcpy(eof.sig, "ILDA", 4);
    eof.format = 5;
    f.write(reinterpret_cast<const char*>(&eof), sizeof(eof));

    std::cout << "[ILDA] Saved " << totalFrames << " frames → " << path << "\n";
    return true;
}

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------

std::vector<Frame> ILDAFile::Load(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "[ILDA] Cannot open: " << path << "\n";
        return {};
    }

    std::vector<Frame> animation;

    while (f) {
        ILDAHeader hdr{};
        if (!f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr))) break;

        // Validate signature
        if (std::memcmp(hdr.sig, "ILDA", 4) != 0) {
            std::cerr << "[ILDA] Bad signature at offset "
                      << (static_cast<int>(f.tellg()) - sizeof(hdr)) << "\n";
            break;
        }

        uint16_t numRecords = FromBE(hdr.numRecords);

        // EOF section
        if (numRecords == 0) break;

        // Only handle format 5 (2D true colour)
        if (hdr.format != 5) {
            std::cerr << "[ILDA] Unsupported format " << (int)hdr.format
                      << " in frame " << FromBE(hdr.frameNumber)
                      << " — skipping.\n";
            // Skip the points we can't read
            // (we don't know point size for other formats here, so stop)
            break;
        }

        Frame frame;
        std::vector<LaserPoint> currentPoly;

        for (uint16_t pi = 0; pi < numRecords; ++pi) {
            ILDAPoint5 pt{};
            if (!f.read(reinterpret_cast<char*>(&pt), sizeof(pt))) {
                std::cerr << "[ILDA] Unexpected end of file.\n";
                break;
            }

            bool blank = (pt.status & 0x40) != 0;

            LaserPoint lp;
            lp.x = ILDAToCoord(static_cast<int16_t>(FromBE(
                                    static_cast<uint16_t>(pt.x))));
            lp.y = ILDAToCoord(static_cast<int16_t>(FromBE(
                                    static_cast<uint16_t>(pt.y))));
            lp.r = ByteToFloat(pt.r);
            lp.g = ByteToFloat(pt.g);
            lp.b = ByteToFloat(pt.b);

            if (blank && !currentPoly.empty()) {
                // Blank point starts a new polyline — flush the current one
                frame.push_back(std::move(currentPoly));
                currentPoly.clear();
            }

            currentPoly.push_back(lp);
        }

        // Flush last polyline
        if (!currentPoly.empty())
            frame.push_back(std::move(currentPoly));

        if (!frame.empty())
            animation.push_back(std::move(frame));
    }

    std::cout << "[ILDA] Loaded " << animation.size()
              << " frames from " << path << "\n";
    return animation;
}

// ---------------------------------------------------------------------------
// PrintInfo
// ---------------------------------------------------------------------------

void ILDAFile::PrintInfo(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "[ILDA] Cannot open: " << path << "\n";
        return;
    }

    int frameCount = 0;
    int totalPoints = 0;

    while (f) {
        ILDAHeader hdr{};
        if (!f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr))) break;
        if (std::memcmp(hdr.sig, "ILDA", 4) != 0) break;

        uint16_t numRecords  = FromBE(hdr.numRecords);
        uint16_t frameNumber = FromBE(hdr.frameNumber);
        uint16_t totalFrames = FromBE(hdr.totalFrames);

        if (numRecords == 0) break;  // EOF

        std::cout << "  Frame " << frameNumber << "/" << totalFrames
                  << "  points=" << numRecords
                  << "  format=" << (int)hdr.format << "\n";

        ++frameCount;
        totalPoints += numRecords;

        // Skip points
        f.seekg(numRecords * sizeof(ILDAPoint5), std::ios::cur);
    }

    std::cout << "[ILDA] " << path << ": "
              << frameCount << " frames, "
              << totalPoints << " total points\n";
}
