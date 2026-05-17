#pragma once
#include "HeliosOutput.h"
#include "libs/helios/HeliosDac.h"
#include <array>
#include <map>
#include <string>
#include <vector>

// -----------------------------------------------------------------------------
// ILDAFile
//
// Reads ILDA animation files conforming to the ILDA Image Data Transfer Format
// Specification, Revision 011 (Nov 2014).  All five point formats are
// supported on read:
//
//   Format 0 – 3D coordinates with indexed color   (8 bytes/point)
//   Format 1 – 2D coordinates with indexed color   (6 bytes/point)
//   Format 2 – Color palette (replaces projector palette state, stateful)
//   Format 4 – 3D coordinates with true color     (10 bytes/point)
//   Format 5 – 2D coordinates with true color      (8 bytes/point)
//
// All multi-byte numeric fields are big-endian per the spec.  3D Z coordinates
// are ignored (we project to 2D).  Blanking bit (status & 0x40) overrides any
// color information per the spec: a blanked point is always black.  Indexed
// frames without a preceding Format 2 section use the standard ILDA default
// palette (Appendix A — 64 entries, indexes 0..63 are defined hues + white).
//
// Per the spec (section 2.4) an ILDA file IS the point sequence the DAC will
// play — "data samples which are directly sent to the galvanometer scanners
// […] NOT raw vector information which needs further processing".  Load()
// therefore returns one flat point sequence per frame, with the spec's
// blank/color decoding applied.  Runtime feeds these straight to
// HeliosOutput::SendPhysical — never SendFrame.  Re-baking the file through
// BuildFrame would inject our own blank/dwell/travel envelopes around every
// blank-separated sub-polyline, inflating frame size and replacing the
// file author's intended timing with ours (visible as tearing, missing
// strokes, and over-bright settle points).
//
// Files written by Save() use Format 5 with company="LZRBAKED" — purely
// informational metadata identifying our encoder; the runtime treats them
// identically to any other ILDA file on read.
// -----------------------------------------------------------------------------

using Frame = std::vector<std::vector<LaserPoint>>;

class ILDAFile {
public:
    // Save animation to path. Returns false on I/O error.
    static bool Save(const std::string& path,
                     const std::vector<Frame>& animation,
                     const std::string& name = "laserctrl");

    // Load animation as one flat HeliosPoint sequence per frame (12-bit ILDA
    // coords + intensity), preserving every point in spec order including
    // blanking.  Feed each frame straight to HeliosOutput::SendPhysical.
    static std::vector<std::vector<HeliosPoint>>
        Load(const std::string& path);

    // Print frame/point counts without fully loading.
    static void PrintInfo(const std::string& path);

private:
#pragma pack(push, 1)
    struct ILDAHeader {
        char     sig[4];          // "ILDA"
        uint8_t  reserved[3];     // 0x00 0x00 0x00
        uint8_t  format;          // 5 = 2D true colour
        char     name[8];         // frame name, null-padded
        char     company[8];      // company name, null-padded
        uint16_t numRecords;      // big-endian; 0 in EOF section
        uint16_t frameNumber;     // big-endian, 0-based
        uint16_t totalFrames;     // big-endian; 0 in EOF section
        uint8_t  projector;       // 0
        uint8_t  reserved2;       // 0
    };  // 32 bytes

    struct ILDAPoint0 {   // 8 bytes — 3D indexed
        int16_t x, y, z;
        uint8_t status;
        uint8_t colorIdx;
    };
    struct ILDAPoint1 {   // 6 bytes — 2D indexed
        int16_t x, y;
        uint8_t status;
        uint8_t colorIdx;
    };
    struct ILDAPoint2 {   // 3 bytes — palette entry (R, G, B)
        uint8_t r, g, b;
    };
    struct ILDAPoint4 {   // 10 bytes — 3D true colour
        int16_t x, y, z;
        uint8_t status;
        uint8_t b, g, r;
    };
    struct ILDAPoint5 {   // 8 bytes — 2D true colour
        int16_t x, y;
        uint8_t status;
        uint8_t b, g, r;
    };
#pragma pack(pop)

    // ---------------------------------------------------------------------
    // Decoded point — common format used internally for every input format.
    // Z is preserved for future 3D work but ignored by current renderers.
    // ---------------------------------------------------------------------
    struct DecodedPoint {
        int16_t x, y, z;
        bool    blank;
        uint8_t r, g, b;
    };

    // ---------------------------------------------------------------------
    // Per-projector palette state.  Format 2 sections replace the entry
    // for that projector; subsequent indexed frames look up colors here.
    // Until the first Format 2 is seen, a projector uses the ILDA default
    // palette from Appendix A of the spec.
    // ---------------------------------------------------------------------
    struct ProjectorPalette { uint8_t rgb[256][3]; };
    using PaletteState = std::map<uint8_t, ProjectorPalette>;

    static const std::array<std::array<uint8_t, 3>, 64>& DefaultILDAPalette();
    static ProjectorPalette& EnsurePalette(PaletteState& state, uint8_t projector);

    // Read one section (header + records).  Updates `palettes` for Format 2;
    // fills `points` for Formats 0/1/4/5; leaves `points` empty for palettes.
    // Returns false on EOF marker, bad signature, or I/O error.  `eof` is
    // set true only when a clean EOF marker was reached.
    static bool ReadSection(std::ifstream& f,
                            ILDAHeader& hdr,
                            std::vector<DecodedPoint>& points,
                            PaletteState& palettes,
                            bool& eof,
                            const std::string& pathForErrors);

    static uint16_t ToBE(uint16_t v);
    static uint16_t FromBE(uint16_t v);
    static int16_t  CoordToILDA(float v);
    static float    ILDAToCoord(int16_t v);
    static uint8_t  FloatToByte(float v);
    static float    ByteToFloat(uint8_t v);
};
