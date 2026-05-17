#include "ILDAFile.h"
#include <fstream>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <cmath>

// ---------------------------------------------------------------------------
// Suggested Default Color Palette — ILDA Spec Rev 011, Appendix A.
// 64 entries used by virtually all "indexed-color" .ild files that ship
// without a Format 2 palette section.  Colors 64..255 fall back to white.
// ---------------------------------------------------------------------------

const std::array<std::array<uint8_t, 3>, 64>& ILDAFile::DefaultILDAPalette()
{
    static const std::array<std::array<uint8_t, 3>, 64> P = {{
        {{255,   0,   0}}, {{255,  16,   0}}, {{255,  32,   0}}, {{255,  48,   0}},
        {{255,  64,   0}}, {{255,  80,   0}}, {{255,  96,   0}}, {{255, 112,   0}},
        {{255, 128,   0}}, {{255, 144,   0}}, {{255, 160,   0}}, {{255, 176,   0}},
        {{255, 192,   0}}, {{255, 208,   0}}, {{255, 224,   0}}, {{255, 240,   0}},
        {{255, 255,   0}}, {{224, 255,   0}}, {{192, 255,   0}}, {{160, 255,   0}},
        {{128, 255,   0}}, {{ 96, 255,   0}}, {{ 64, 255,   0}}, {{ 32, 255,   0}},
        {{  0, 255,   0}}, {{  0, 255,  36}}, {{  0, 255,  73}}, {{  0, 255, 109}},
        {{  0, 255, 146}}, {{  0, 255, 182}}, {{  0, 255, 219}}, {{  0, 255, 255}},
        {{  0, 227, 255}}, {{  0, 198, 255}}, {{  0, 170, 255}}, {{  0, 142, 255}},
        {{  0, 113, 255}}, {{  0,  85, 255}}, {{  0,  56, 255}}, {{  0,  28, 255}},
        {{  0,   0, 255}}, {{ 32,   0, 255}}, {{ 64,   0, 255}}, {{ 96,   0, 255}},
        {{128,   0, 255}}, {{160,   0, 255}}, {{192,   0, 255}}, {{224,   0, 255}},
        {{255,   0, 255}}, {{255,  32, 255}}, {{255,  64, 255}}, {{255,  96, 255}},
        {{255, 128, 255}}, {{255, 160, 255}}, {{255, 192, 255}}, {{255, 224, 255}},
        {{255, 255, 255}}, {{255, 224, 224}}, {{255, 192, 192}}, {{255, 160, 160}},
        {{255, 128, 128}}, {{255,  96,  96}}, {{255,  64,  64}}, {{255,  32,  32}},
    }};
    return P;
}

ILDAFile::ProjectorPalette&
ILDAFile::EnsurePalette(PaletteState& state, uint8_t projector)
{
    auto it = state.find(projector);
    if (it != state.end()) return it->second;
    ProjectorPalette& pal = state[projector];
    const auto& def = DefaultILDAPalette();
    for (int i = 0; i < 64; ++i) {
        pal.rgb[i][0] = def[i][0];
        pal.rgb[i][1] = def[i][1];
        pal.rgb[i][2] = def[i][2];
    }
    // Fill 64..255 with white so out-of-range indices stay visible
    // (spec leaves these undefined; treating them as black would silently
    // swallow content from files that wrap around the palette).
    for (int i = 64; i < 256; ++i) {
        pal.rgb[i][0] = 255;
        pal.rgb[i][1] = 255;
        pal.rgb[i][2] = 255;
    }
    return pal;
}

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
// ReadSection — generic per-section reader.
//
// Returns true if a section was read successfully and `points` (for frame
// formats) or `palettes` (for Format 2) was populated.  Returns false on
// EOF marker, bad signature, or I/O error.  `eof` is set to true ONLY for a
// clean EOF marker (header with numRecords=0).
// ---------------------------------------------------------------------------

bool ILDAFile::ReadSection(std::ifstream& f,
                            ILDAHeader& hdr,
                            std::vector<DecodedPoint>& points,
                            PaletteState& palettes,
                            bool& eof,
                            const std::string& pathForErrors)
{
    eof = false;
    points.clear();

    if (!f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr))) return false;

    if (std::memcmp(hdr.sig, "ILDA", 4) != 0) {
        std::cerr << "[ILDA] " << pathForErrors << ": bad signature at offset "
                  << (static_cast<int>(f.tellg()) - sizeof(hdr)) << "\n";
        return false;
    }

    uint16_t numRecords = FromBE(hdr.numRecords);
    if (numRecords == 0) { eof = true; return false; }

    uint8_t projector = hdr.projector;

    auto readBE16 = [](int16_t raw) -> int16_t {
        return static_cast<int16_t>(FromBE(static_cast<uint16_t>(raw)));
    };

    switch (hdr.format) {
    case 0: {  // 3D indexed
        ProjectorPalette& pal = EnsurePalette(palettes, projector);
        points.reserve(numRecords);
        for (uint16_t pi = 0; pi < numRecords; ++pi) {
            ILDAPoint0 pt{};
            if (!f.read(reinterpret_cast<char*>(&pt), sizeof(pt))) return false;
            DecodedPoint dp;
            dp.x = readBE16(pt.x);
            dp.y = readBE16(pt.y);
            dp.z = readBE16(pt.z);
            dp.blank = (pt.status & 0x40) != 0;
            dp.r = pal.rgb[pt.colorIdx][0];
            dp.g = pal.rgb[pt.colorIdx][1];
            dp.b = pal.rgb[pt.colorIdx][2];
            points.push_back(dp);
        }
        return true;
    }
    case 1: {  // 2D indexed
        ProjectorPalette& pal = EnsurePalette(palettes, projector);
        points.reserve(numRecords);
        for (uint16_t pi = 0; pi < numRecords; ++pi) {
            ILDAPoint1 pt{};
            if (!f.read(reinterpret_cast<char*>(&pt), sizeof(pt))) return false;
            DecodedPoint dp;
            dp.x = readBE16(pt.x);
            dp.y = readBE16(pt.y);
            dp.z = 0;
            dp.blank = (pt.status & 0x40) != 0;
            dp.r = pal.rgb[pt.colorIdx][0];
            dp.g = pal.rgb[pt.colorIdx][1];
            dp.b = pal.rgb[pt.colorIdx][2];
            points.push_back(dp);
        }
        return true;
    }
    case 2: {  // Color palette — replaces this projector's palette state
        ProjectorPalette& pal = EnsurePalette(palettes, projector);
        // Spec: palette SHALL have 2..256 entries.  Be tolerant on read.
        uint16_t n = std::min<uint16_t>(numRecords, 256);
        for (uint16_t i = 0; i < n; ++i) {
            ILDAPoint2 e{};
            if (!f.read(reinterpret_cast<char*>(&e), sizeof(e))) return false;
            pal.rgb[i][0] = e.r;
            pal.rgb[i][1] = e.g;
            pal.rgb[i][2] = e.b;
        }
        // Skip any entries above 256 (shouldn't happen, but defensive)
        if (numRecords > 256)
            f.seekg((numRecords - 256) * sizeof(ILDAPoint2), std::ios::cur);
        // No DecodedPoints to return for a palette section
        return true;
    }
    case 4: {  // 3D true colour
        points.reserve(numRecords);
        for (uint16_t pi = 0; pi < numRecords; ++pi) {
            ILDAPoint4 pt{};
            if (!f.read(reinterpret_cast<char*>(&pt), sizeof(pt))) return false;
            DecodedPoint dp;
            dp.x = readBE16(pt.x);
            dp.y = readBE16(pt.y);
            dp.z = readBE16(pt.z);
            dp.blank = (pt.status & 0x40) != 0;
            dp.r = pt.r;
            dp.g = pt.g;
            dp.b = pt.b;
            points.push_back(dp);
        }
        return true;
    }
    case 5: {  // 2D true colour
        points.reserve(numRecords);
        for (uint16_t pi = 0; pi < numRecords; ++pi) {
            ILDAPoint5 pt{};
            if (!f.read(reinterpret_cast<char*>(&pt), sizeof(pt))) return false;
            DecodedPoint dp;
            dp.x = readBE16(pt.x);
            dp.y = readBE16(pt.y);
            dp.z = 0;
            dp.blank = (pt.status & 0x40) != 0;
            dp.r = pt.r;
            dp.g = pt.g;
            dp.b = pt.b;
            points.push_back(dp);
        }
        return true;
    }
    default:
        std::cerr << "[ILDA] " << pathForErrors << ": unknown format "
                  << (int)hdr.format << " in section "
                  << FromBE(hdr.frameNumber)
                  << " — cannot determine point size, aborting.\n";
        return false;
    }
}

// ---------------------------------------------------------------------------
// Load — one HeliosPoint sequence per frame, in spec point order.
//
// Per ILDA spec section 2.4 an ILDA file IS the point sequence the DAC will
// play; we do not split, re-order, or re-bake it.  Per spec 5.2.4 a blanked
// point's RGB SHALL be treated as zero on read; we honor that here.
// ---------------------------------------------------------------------------

std::vector<std::vector<HeliosPoint>>
ILDAFile::Load(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "[ILDA] Cannot open: " << path << "\n";
        return {};
    }

    std::vector<std::vector<HeliosPoint>> animation;
    PaletteState palettes;
    std::vector<DecodedPoint> decoded;
    int paletteSections = 0;

    while (f) {
        ILDAHeader hdr{};
        bool eof = false;
        if (!ReadSection(f, hdr, decoded, palettes, eof, path)) {
            if (eof) break;
            return {};
        }

        if (hdr.format == 2) { ++paletteSections; continue; }
        if (decoded.empty()) continue;

        std::vector<HeliosPoint> frame;
        frame.reserve(decoded.size());
        for (const auto& dp : decoded) {
            // 16-bit signed [-32767..32767, centre 0]
            //  → 12-bit unsigned [0..4095, centre 2048]
            int x12 = (int)std::lround(dp.x / 16.0f) + 2048;
            int y12 = (int)std::lround(dp.y / 16.0f) + 2048;
            x12 = std::clamp(x12, 0, 4095);
            y12 = std::clamp(y12, 0, 4095);

            HeliosPoint hp;
            hp.x = static_cast<uint16_t>(x12);
            hp.y = static_cast<uint16_t>(y12);
            if (dp.blank) {
                hp.r = hp.g = hp.b = hp.i = 0;
            } else {
                hp.r = dp.r;
                hp.g = dp.g;
                hp.b = dp.b;
                hp.i = std::max(dp.r, std::max(dp.g, dp.b));
            }
            frame.push_back(hp);
        }
        if (!frame.empty())
            animation.push_back(std::move(frame));
    }

    std::cout << "[ILDA] Loaded " << animation.size() << " frames from "
              << path;
    if (paletteSections > 0)
        std::cout << "  (consumed " << paletteSections << " palette section"
                  << (paletteSections == 1 ? "" : "s") << ")";
    std::cout << "\n";
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

    auto pointSize = [](uint8_t fmt) -> int {
        switch (fmt) {
            case 0: return 8;
            case 1: return 6;
            case 2: return 3;
            case 4: return 10;
            case 5: return 8;
            default: return 0;   // unknown -> can't seek
        }
    };

    int frameCount = 0, paletteCount = 0, totalPoints = 0;

    while (f) {
        ILDAHeader hdr{};
        if (!f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr))) break;
        if (std::memcmp(hdr.sig, "ILDA", 4) != 0) break;

        uint16_t numRecords  = FromBE(hdr.numRecords);
        uint16_t frameNumber = FromBE(hdr.frameNumber);
        uint16_t totalFrames = FromBE(hdr.totalFrames);
        if (numRecords == 0) break;  // EOF

        int psz = pointSize(hdr.format);
        if (psz == 0) {
            std::cerr << "[ILDA] PrintInfo: unknown format "
                      << (int)hdr.format << " — stopping.\n";
            break;
        }

        if (hdr.format == 2) {
            std::cout << "  Palette " << frameNumber
                      << "  entries=" << numRecords
                      << "  projector=" << (int)hdr.projector << "\n";
            ++paletteCount;
        } else {
            std::cout << "  Frame " << frameNumber << "/" << totalFrames
                      << "  points=" << numRecords
                      << "  format=" << (int)hdr.format
                      << "  projector=" << (int)hdr.projector << "\n";
            ++frameCount;
            totalPoints += numRecords;
        }
        f.seekg(static_cast<std::streamoff>(numRecords) * psz, std::ios::cur);
    }

    std::cout << "[ILDA] " << path << ": "
              << frameCount << " frames, " << totalPoints << " total points";
    if (paletteCount > 0)
        std::cout << ", " << paletteCount << " palette section"
                  << (paletteCount == 1 ? "" : "s");
    std::cout << "\n";
}
