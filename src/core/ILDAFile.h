#pragma once
#include "HeliosOutput.h"
#include "libs/helios/HeliosDac.h"
#include <string>
#include <vector>

// -----------------------------------------------------------------------------
// ILDAFile
//
// Reads and writes ILDA animation files (.ild), Format 5 (2D true colour).
//
// Point layout on disk (8 bytes, all big-endian):
//   int16  X      [-32767, 32767]  ←→  LaserPoint.x  [-1, 1]
//   int16  Y      [-32767, 32767]  ←→  LaserPoint.y  [-1, 1]
//   uint8  status  bit6=blank (1=laser off / move, 0=laser on / draw)
//   uint8  B       [0, 255]        ←→  LaserPoint.b  [0, 1]
//   uint8  G       [0, 255]        ←→  LaserPoint.g  [0, 1]
//   uint8  R       [0, 255]        ←→  LaserPoint.r  [0, 1]
//
// Blanking convention:
//   The first point of every polyline is written with blank=1.
//   All subsequent points in the polyline are written with blank=0.
//   On load, a blank point begins a new polyline — perfectly reversible.
//
// File structure:
//   [ 32-byte frame header | N × 8-byte points ] × numFrames
//   [ 32-byte EOF header (numRecords = 0) ]
// -----------------------------------------------------------------------------

using Frame = std::vector<std::vector<LaserPoint>>;

class ILDAFile {
public:
    // Save animation to path. Returns false on I/O error.
    static bool Save(const std::string& path,
                     const std::vector<Frame>& animation,
                     const std::string& name = "laserctrl");

    // Load animation from path as logical polylines (blank-flag separates
    // polylines).  Used for the legacy un-baked .ild format and runtime
    // SendFrame() pathway.  Returns empty vector on failure.
    static std::vector<Frame> Load(const std::string& path);

    // Load animation as flat per-frame HeliosPoint sequences (12-bit ILDA
    // coords + intensity), preserving every point including blanking.
    // This is the path for "physically baked" .ild files produced by the
    // post-Phase-1 encode.py — every point in the file is a point the DAC
    // will play, so no further processing is needed at runtime.
    static std::vector<std::vector<HeliosPoint>>
        LoadFlat(const std::string& path);

    // True if the file's first frame header has company="LZRBAKED",
    // indicating encode.py has baked all blanking/dwells/travels into the
    // file.  Use LoadFlat + HeliosOutput::SendPhysical for these.
    static bool IsBaked(const std::string& path);

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

    struct ILDAPoint5 {
        int16_t  x;       // big-endian
        int16_t  y;       // big-endian
        uint8_t  status;  // bit 6 = blank
        uint8_t  b;
        uint8_t  g;
        uint8_t  r;
    };  // 8 bytes
#pragma pack(pop)

    static uint16_t ToBE(uint16_t v);
    static uint16_t FromBE(uint16_t v);
    static int16_t  CoordToILDA(float v);
    static float    ILDAToCoord(int16_t v);
    static uint8_t  FloatToByte(float v);
    static float    ByteToFloat(uint8_t v);
};
