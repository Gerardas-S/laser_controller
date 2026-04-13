#include "MusicGeometryEngine.h"
#include <algorithm>
#include <iostream>

// ---------------------------------------------------------------------------
// Constructor — register all modes
// ---------------------------------------------------------------------------

MusicGeometryEngine::MusicGeometryEngine()
{
    modes_["lissajous"]    = [this](const AudioFeatures& f, double t){ return ModeLissajous(f, t); };
    modes_["torusknot"]    = [this](const AudioFeatures& f, double t){ return ModeTorusKnot(f, t); };
    modes_["spectrum"]     = [this](const AudioFeatures& f, double t){ return ModeSpectrum(f, t); };
    modes_["spiral"]       = [this](const AudioFeatures& f, double t){ return ModeSpiral(f, t); };
    modes_["rose"]         = [this](const AudioFeatures& f, double t){ return ModeRose(f, t); };
    modes_["hypotrochoid"] = [this](const AudioFeatures& f, double t){ return ModeHypotrochoid(f, t); };
    modes_["pulsar"]       = [this](const AudioFeatures& f, double t){ return ModePulsar(f, t); };
    modes_["waveform"]     = [this](const AudioFeatures& f, double t){ return ModeWaveform(f, t); };
    modes_["terrain"]      = [this](const AudioFeatures& f, double t){ return ModeTerrain(f, t); };
    modes_["melody"]       = [this](const AudioFeatures& f, double t){ return ModeMelody(f, t); };
    modes_["chroma"]       = [this](const AudioFeatures& f, double t){ return ModeChroma(f, t); };
}

void MusicGeometryEngine::SetMode(const std::string& name)
{
    if (modes_.count(name)) {
        currentMode_ = name;
        modeTime_    = 0.0;
        std::cout << "[MusicGeometry] Mode: " << name << "\n";
    } else {
        std::cerr << "[MusicGeometry] Unknown mode: " << name << "\n";
    }
}

std::vector<std::string> MusicGeometryEngine::GetModeNames() const
{
    std::vector<std::string> names;
    for (const auto& kv : modes_) names.push_back(kv.first);
    return names;
}

Frame3D MusicGeometryEngine::Generate(const AudioFeatures& feat, double timeSec)
{
    modeTime_ = timeSec;

    // Track beat continuity
    if (feat.isBeat) lastBeatTime_ = timeSec;

    auto it = modes_.find(currentMode_);
    if (it == modes_.end()) return {};
    return it->second(feat, timeSec);
}

// ---------------------------------------------------------------------------
// Colour helper
// ---------------------------------------------------------------------------

void MusicGeometryEngine::HsvToRgb(float h, float s, float v,
                                      float& r, float& g, float& b)
{
    h = std::fmod(h, 1.0f) * 6.0f;
    int   i = static_cast<int>(h);
    float f = h - i;
    float p = v * (1 - s);
    float q = v * (1 - s * f);
    float t2= v * (1 - s * (1 - f));
    switch (i % 6) {
        case 0: r=v; g=t2; b=p; break;
        case 1: r=q; g=v;  b=p; break;
        case 2: r=p; g=v;  b=t2;break;
        case 3: r=p; g=q;  b=v; break;
        case 4: r=t2;g=p;  b=v; break;
        default:r=v; g=p;  b=q; break;
    }
}

// ---------------------------------------------------------------------------
// Mode: Lissajous
// Frequency ratios morph with spectral centroid. Bass scales amplitude.
// Beat triggers ratio snap to new harmonic pair.
// ---------------------------------------------------------------------------

Frame3D MusicGeometryEngine::ModeLissajous(const AudioFeatures& f, double t)
{
    // Snap to new ratio on beat
    if (f.isBeat) {
        static const float aRatios[] = { 1,2,3,3,4,5,5,6 };
        static const float bRatios[] = { 1,1,2,4,3,4,6,5 };
        int idx = static_cast<int>(f.bpm / 30.0f) % 8;
        lissATarget_ = aRatios[idx];
        lissBTarget_ = bRatios[idx];
    }

    // Smoothly interpolate ratios
    lissA_ = Lerp(lissA_, lissATarget_, 0.05f);
    lissB_ = Lerp(lissB_, lissBTarget_, 0.05f);

    float delta = static_cast<float>(t * 0.3 * (1.0 + f.spectralCentroid));
    float gamma = static_cast<float>(t * 0.1 * (1.0 + f.bassSmooth));

    // Scale amplitude by RMS
    float amp = 0.4f + f.rmsSmooth * 0.6f;

    // Hue follows spectral centroid
    float r, g2, b;
    HsvToRgb(f.spectralCentroid, 0.8f, 0.9f, r, g2, b);

    int pts = std::max(200, targetPoints / 2);
    Polyline3D curve = GeometryShapes::Lissajous(
        pts, lissA_, lissB_, 0.0f, delta, gamma, r, g2, b);

    // Scale by amplitude
    for (auto& p : curve) { p.x *= amp; p.y *= amp; }

    // Z modulation: bass pushes depth
    float zAmp = f.bassSmooth * 0.4f;
    for (int i = 0; i < static_cast<int>(curve.size()); ++i) {
        float phase = GeometryShapes::TAU * i / curve.size();
        curve[i].z = zAmp * std::sin(phase * 3.0f + static_cast<float>(t));
    }

    return { curve };
}

// ---------------------------------------------------------------------------
// Mode: Torus Knot
// Winding numbers change with beat. Rotation driven by tempo.
// ---------------------------------------------------------------------------

Frame3D MusicGeometryEngine::ModeTorusKnot(const AudioFeatures& f, double t)
{
    if (f.isBeat) {
        knotP_ = 2 + (static_cast<int>(f.bpm / 40.0f) % 4);
        knotQ_ = knotP_ + 1 + (static_cast<int>(f.bassSmooth * 3.0f));
    }

    float R  = 0.6f + f.rmsSmooth * 0.3f;
    float rr = 0.15f + f.bassSmooth * 0.2f;

    float r, g2, b;
    HsvToRgb(static_cast<float>(t * 0.05), 0.9f, 1.0f, r, g2, b);

    int pts = targetPoints;
    Polyline3D knot = GeometryShapes::TorusKnot(pts, knotP_, knotQ_, R, rr, r, g2, b);

    // Rotate around Z with BPM-driven speed
    float rotSpeed = (f.bpm > 0) ? GeometryShapes::TAU * f.bpm / 60.0f : 1.0f;
    float angle = static_cast<float>(t * rotSpeed * 0.1);
    float cosA = std::cos(angle), sinA = std::sin(angle);
    for (auto& p : knot) {
        float nx = p.x * cosA - p.y * sinA;
        float ny = p.x * sinA + p.y * cosA;
        p.x = nx; p.y = ny;
    }

    return { knot };
}

// ---------------------------------------------------------------------------
// Mode: Spectrum bars (3D — z = frequency band energy)
// ---------------------------------------------------------------------------

Frame3D MusicGeometryEngine::ModeSpectrum(const AudioFeatures& f, double t)
{
    Frame3D frame = GeometryShapes::SpectrumBars(f, 0.0f, 1.0f, 0.5f, 0.0f);

    // Colour bars by band
    for (int i = 0; i < static_cast<int>(frame.size()); ++i) {
        float hue = static_cast<float>(i) / frame.size();
        float r2, g2, b2;
        HsvToRgb(hue + static_cast<float>(t * 0.1), 0.9f, f.melBands[i] + 0.1f,
                  r2, g2, b2);
        for (auto& p : frame[i]) { p.r = r2; p.g = g2; p.b = b2; }

        // Z = extrude by bass on low bands
        float zVal = (i < 8) ? f.bassSmooth * f.melBands[i] : 0.0f;
        for (auto& p : frame[i]) p.z = zVal;
    }
    return frame;
}

// ---------------------------------------------------------------------------
// Mode: Spiral — expands/contracts with RMS, rotates with BPM
// ---------------------------------------------------------------------------

Frame3D MusicGeometryEngine::ModeSpiral(const AudioFeatures& f, double t)
{
    float turns  = 3.0f + f.spectralCentroid * 5.0f;
    float inner  = 0.05f;
    float outer  = 0.3f + f.rmsSmooth * 0.7f;
    float z      = std::sin(static_cast<float>(t * 0.5)) * f.bassSmooth * 0.5f;

    float r2, g2, b2;
    HsvToRgb(f.bassSmooth * 0.3f + static_cast<float>(t * 0.02), 1.0f, 0.9f,
              r2, g2, b2);

    int pts = targetPoints;
    Polyline3D spiral = GeometryShapes::Spiral(pts, turns, inner, outer, z, r2, g2, b2);

    // Rotate with BPM
    float angle = (f.bpm > 0)
        ? static_cast<float>(t * GeometryShapes::TAU * f.bpm / 60.0 * 0.25)
        : static_cast<float>(t * 0.5);
    float cosA = std::cos(angle), sinA = std::sin(angle);
    for (auto& p : spiral) {
        float nx = p.x * cosA - p.y * sinA;
        float ny = p.x * sinA + p.y * cosA;
        p.x = nx; p.y = ny;
    }

    return { spiral };
}

// ---------------------------------------------------------------------------
// Mode: Rose curve
// k modulates with spectral centroid. Petals pulse with beats.
// ---------------------------------------------------------------------------

Frame3D MusicGeometryEngine::ModeRose(const AudioFeatures& f, double t)
{
    float k = 2.0f + std::floor(f.spectralCentroid * 6.0f);

    float r2, g2, b2;
    HsvToRgb(f.spectralCentroidSmooth * 0.7f, 0.95f, 0.9f, r2, g2, b2);

    float z = f.bassSmooth * 0.3f * std::sin(static_cast<float>(t * 2.0));
    int pts = targetPoints;
    Polyline3D rose = GeometryShapes::Rose(pts, k, z, r2, g2, b2);

    // Scale pulses on beat
    float scale = 0.6f + f.rmsSmooth * 0.4f;
    if (f.isBeat) scale = std::min(scale * 1.3f, 1.0f);
    for (auto& p : rose) { p.x *= scale; p.y *= scale; }

    return { rose };
}

// ---------------------------------------------------------------------------
// Mode: Hypotrochoid (Spirograph)
// R, r, d parameters drift slowly with audio
// ---------------------------------------------------------------------------

Frame3D MusicGeometryEngine::ModeHypotrochoid(const AudioFeatures& f, double t)
{
    float R = 5.0f;
    float ri = 2.0f + f.spectralCentroid * 3.0f;
    float d  = 1.0f + f.bassSmooth * 4.0f;
    float z  = f.midHigh * 0.3f;

    float r2, g2, b2;
    HsvToRgb(static_cast<float>(t * 0.03) + f.spectralCentroid * 0.5f,
              0.85f, 0.95f, r2, g2, b2);

    int pts = std::max(300, targetPoints);
    Polyline3D hypo = GeometryShapes::Hypotrochoid(pts, R, ri, d, z, r2, g2, b2);

    float scale = 0.5f + f.rmsSmooth * 0.5f;
    for (auto& p : hypo) { p.x *= scale; p.y *= scale; }

    return { hypo };
}

// ---------------------------------------------------------------------------
// Mode: Pulsar — expanding/collapsing concentric rings, beat-driven
// ---------------------------------------------------------------------------

Frame3D MusicGeometryEngine::ModePulsar(const AudioFeatures& f, double t)
{
    Frame3D frame;

    int rings = 4;
    float timeSinceBeat = static_cast<float>(t - lastBeatTime_);
    float beatPeriod    = (f.bpm > 0) ? 60.0f / f.bpm : 1.0f;
    float phase         = std::fmod(timeSinceBeat, beatPeriod) / beatPeriod;

    for (int ring = 0; ring < rings; ++ring) {
        float ringPhase = std::fmod(phase + static_cast<float>(ring) / rings, 1.0f);
        float radius    = ringPhase * 0.9f;
        float brightness= 1.0f - ringPhase;

        float hue = static_cast<float>(ring) / rings + f.spectralCentroid * 0.3f;
        float r2, g2, b2;
        HsvToRgb(hue, 0.8f, brightness, r2, g2, b2);

        float z = (f.bassSmooth * 0.5f) * (1.0f - ringPhase);

        int pts = std::max(60, targetPoints / rings);
        Polyline3D circle = GeometryShapes::Lissajous(
            pts, 1, 1, 0, GeometryShapes::PI * 0.5f, 0, r2, g2, b2, z);
        for (auto& p : circle) { p.x *= radius; p.y *= radius; }

        frame.push_back(std::move(circle));
    }

    return frame;
}

// ---------------------------------------------------------------------------
// Mode: Waveform — mel bands rendered as waveform slices in Z
// ---------------------------------------------------------------------------

Frame3D MusicGeometryEngine::ModeWaveform(const AudioFeatures& f, double t)
{
    Frame3D frame;

    // Render 8 slices — each a sine wave scaled by a mel band group
    int slices = 8;
    int ptsPerSlice = targetPoints / slices;

    for (int s = 0; s < slices; ++s) {
        float z = -0.7f + 1.4f * static_cast<float>(s) / (slices - 1);

        // Average 4 mel bands per slice
        int band0 = s * 4;
        float energy = 0.0f;
        for (int b = 0; b < 4 && band0 + b < AudioFeatures::kMelBands; ++b)
            energy += f.melBands[band0 + b];
        energy /= 4.0f;

        float hue = static_cast<float>(s) / slices + static_cast<float>(t * 0.05);
        float r2, g2, b2;
        HsvToRgb(hue, 0.9f, 0.8f, r2, g2, b2);

        Polyline3D slice;
        for (int i = 0; i < ptsPerSlice; ++i) {
            float x = 2.0f * i / (ptsPerSlice - 1) - 1.0f;
            float freq = 2.0f + static_cast<float>(s) * 0.5f;
            float y = energy * std::sin(GeometryShapes::TAU * freq * x
                                        + static_cast<float>(t * 2.0));
            slice.push_back({ x, y, z, r2, g2, b2 });
        }
        frame.push_back(std::move(slice));
    }

    return frame;
}

// ---------------------------------------------------------------------------
// Mode: Terrain — parametric surface where height = mel band interpolated
// ---------------------------------------------------------------------------

Frame3D MusicGeometryEngine::ModeTerrain(const AudioFeatures& f, double t)
{
    int uSteps = 12, vSteps = 8;

    auto terrainFn = [&](float u, float v) -> GeometryPoint3D {
        float x = u * 2.0f - 1.0f;
        float y = v * 2.0f - 1.0f;

        // Map u to mel band index
        int band = static_cast<int>(u * (AudioFeatures::kMelBands - 1));
        band = std::clamp(band, 0, AudioFeatures::kMelBands - 1);
        float h = f.melBands[band];

        // Add wave distortion driven by bass
        float wave = f.bassSmooth * 0.2f *
            std::sin(GeometryShapes::TAU * (u * 3.0f + v * 2.0f +
                     static_cast<float>(t * 0.5)));

        float z = h * 0.6f + wave - 0.3f;

        float hue = h * 0.6f + f.spectralCentroid * 0.3f;
        float r2, g2, b2;
        HsvToRgb(hue, 0.85f, 0.7f + h * 0.3f, r2, g2, b2);

        return { x, z, y, r2, g2, b2 };  // y/z swapped for top-down view
    };

    return GeometryShapes::ParametricSurface(uSteps, vSteps, terrainFn);
}

// ---------------------------------------------------------------------------
// Mode: Melody — pitch-following Lissajous / spiral
//
// Requires AudioFeatures.pitch populated by PitchAnalyzer (CREPE).
// Falls back gracefully when pitch == 0 (silence / unvoiced).
//
// Design:
//   - Pitch drives the Y-frequency of a Lissajous figure (higher note → finer loops)
//   - Octave (coarse) drives the X-frequency (1..7 maps to 1..7)
//   - Beat events snap the phase delta for visual punctuation
//   - RMS drives amplitude; spectral centroid drives colour
//   - When unvoiced, gently fades to a simple circle
// ---------------------------------------------------------------------------

Frame3D MusicGeometryEngine::ModeMelody(const AudioFeatures& f, double t)
{
    static const float kLogC1 = std::log2(32.70f);    // C1 = 32.70 Hz
    static const float kLogC8 = std::log2(4186.0f);   // C8 = 4186 Hz
    static const float kOctaveRange = kLogC8 - kLogC1; // 7 octaves

    // ---- Pitch decoding ------------------------------------------
    float hz    = f.pitch;           // 0 = unvoiced
    float voiced = (hz > 0.0f) ? std::min(f.pitchConfidence / 0.7f, 1.0f) : 0.0f;

    // Smooth pitch norm [0,1] across frames
    float targetNorm = (hz > 0.0f) ? f.pitchSmooth : melodyPitchSmooth_;
    float alpha = (hz > 0.0f) ? 0.2f : 0.02f;   // fast glide when voiced, slow decay
    melodyPitchSmooth_ = Lerp(melodyPitchSmooth_, targetNorm, alpha);

    // Map pitch norm to Lissajous Y frequency: 1..8 (semitone-granular)
    // We quantise to the nearest semitone for clean figure-eight shapes
    float semitones  = melodyPitchSmooth_ * 84.0f;   // 7 octaves × 12 semitones
    float freqY_raw  = 1.0f + semitones / 12.0f;     // 1..8
    float freqY      = std::round(freqY_raw);         // snap to integer ratio
    float freqX      = std::max(1.0f, std::round(freqY / 2.0f));  // harmonic pair

    // Phase delta drives the slow rotation of the figure
    float delta = static_cast<float>(t * 0.2 * (1.0 + f.spectralCentroid * 0.5));
    float gamma = static_cast<float>(t * 0.05);

    // Amplitude = RMS + voiced blending
    float amp = (0.3f + f.rmsSmooth * 0.5f) * (0.4f + voiced * 0.6f);

    // Colour: voiced notes → warm hue from pitch position; silence → cool blue
    float hue = (hz > 0.0f)
        ? 0.05f + melodyPitchSmooth_ * 0.5f   // warm red→green over pitch range
        : 0.6f + static_cast<float>(t * 0.02f); // cool blue when silent
    float sat  = 0.9f;
    float val  = 0.3f + f.rmsSmooth * 0.7f;
    float r2, g2, b2;
    HsvToRgb(hue, sat, val, r2, g2, b2);

    int pts = std::max(200, targetPoints);
    Polyline3D curve = GeometryShapes::Lissajous(pts, freqX, freqY,
                                                  0.0f, delta, gamma, r2, g2, b2);
    for (auto& p : curve) { p.x *= amp; p.y *= amp; }

    // Z = pitch oscillation — voiced notes breathe in Z with note duration
    float zAmp = voiced * f.rmsSmooth * 0.35f;
    for (int i = 0; i < static_cast<int>(curve.size()); ++i) {
        float ph = GeometryShapes::TAU * i / curve.size();
        curve[i].z = zAmp * std::sin(ph * freqY + static_cast<float>(t * 1.5));
    }

    // When a beat hits, briefly widen the figure
    if (f.isBeat) {
        float punch = 1.0f + 0.3f * f.beatConfidence;
        for (auto& p : curve) { p.x *= punch; p.y *= punch; }
    }

    return { curve };
}

// ---------------------------------------------------------------------------
// Mode: Chroma — 12-pointed star whose arm lengths = chroma energy
//
// Each of the 12 arms corresponds to one pitch class (C, C#, D, …, B).
// Arm length = chroma[i] normalised.  Arms rotate slowly with time.
// Beat events "explode" the star outward briefly.
//
// Requires AudioFeatures.chroma[] (computed from FFT chroma in AudioAnalyzer).
// Falls back to uniform star when chroma is all zeros.
// ---------------------------------------------------------------------------

Frame3D MusicGeometryEngine::ModeChroma(const AudioFeatures& f, double t)
{
    const int kPitchClasses = 12;

    // Check if chroma is populated (non-zero)
    float chromaSum = 0.0f;
    for (float c : f.chroma) chromaSum += c;
    bool hasChroma = (chromaSum > 0.01f);

    // ---- Slow rotation of the whole star -------------------------
    float rotOffset = static_cast<float>(t * 0.05);   // one revolution / 20 s

    // Beat pulse: brief scale factor that decays
    float beatPulse = 1.0f;
    if (f.isBeat) beatPulse = 1.0f + 0.4f * f.beatConfidence;

    // ---- Build the star -------------------------------------------
    // Each arm: tip + base neighbours, forming a thin spike
    Frame3D frame;
    frame.reserve(kPitchClasses);

    for (int i = 0; i < kPitchClasses; ++i) {
        float angle = GeometryShapes::TAU * i / kPitchClasses + rotOffset;

        // Arm length from chroma bin (uniform if chroma missing)
        float chromaVal = hasChroma ? f.chroma[i] : 1.0f / kPitchClasses;
        float len = (0.2f + chromaVal * 0.8f) * (0.5f + f.rmsSmooth * 0.5f) * beatPulse;
        len = std::clamp(len, 0.05f, 1.0f);

        // Colour: pitch class → hue (C=red, G=green, B=blue, chromatic wheel)
        float hue = static_cast<float>(i) / kPitchClasses
                    + f.spectralCentroid * 0.15f
                    + static_cast<float>(t * 0.02);
        float val = 0.5f + chromaVal * 0.5f;
        float r2, g2, b2;
        HsvToRgb(hue, 0.9f, val, r2, g2, b2);

        // Arm polyline: origin → tip → origin (for out-and-back draw)
        float tipX = len * std::cos(angle);
        float tipY = len * std::sin(angle);

        // Small Z offset driven by bass, different per arm
        float z = f.bassSmooth * 0.2f * std::sin(static_cast<float>(t * 2.0) +
                                                   angle);

        Polyline3D arm;
        arm.reserve(3);
        arm.push_back({ 0.0f,  0.0f, 0.0f, r2, g2, b2 });   // origin (blank)
        arm.push_back({ tipX,  tipY, z,    r2, g2, b2 });    // tip (lit)

        // Add a secondary harmonic arm at half-length (octave above)
        float arm2X = (len * 0.4f) * std::cos(angle + GeometryShapes::PI / kPitchClasses);
        float arm2Y = (len * 0.4f) * std::sin(angle + GeometryShapes::PI / kPitchClasses);
        arm.push_back({ arm2X, arm2Y, z * 0.5f, r2 * 0.6f, g2 * 0.6f, b2 * 0.6f });

        frame.push_back(std::move(arm));
    }

    // ---- Central ring (brightest chroma bin highlighted) ----------
    int dominant = f.dominantChroma;
    float domAngle = GeometryShapes::TAU * dominant / kPitchClasses + rotOffset;
    float domLen   = hasChroma ? (0.3f + f.chroma[dominant] * 0.7f) * beatPulse : 0.3f;

    float hr, hg, hb;
    float domHue = static_cast<float>(dominant) / kPitchClasses;
    HsvToRgb(domHue, 1.0f, 1.0f, hr, hg, hb);

    int ringPts = std::max(40, targetPoints / 4);
    Polyline3D ring = GeometryShapes::Lissajous(ringPts, 1, 1, 0.0f,
                                                  GeometryShapes::PI * 0.5f, 0.0f,
                                                  hr, hg, hb);
    float ringR = domLen * 0.25f;
    float cx    = ringR * std::cos(domAngle);
    float cy    = ringR * std::sin(domAngle);
    for (auto& p : ring) {
        p.x = p.x * ringR + cx;
        p.y = p.y * ringR + cy;
    }
    frame.push_back(std::move(ring));

    return frame;
}
