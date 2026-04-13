#include "MusicPipeline.h"
#include <iostream>
#include <sstream>
#include <chrono>

MusicPipeline::MusicPipeline()
    : analyzer_(std::make_unique<AudioAnalyzer>())
    , engine_(std::make_unique<MusicGeometryEngine>())
{
}

MusicPipeline::~MusicPipeline()
{
    Stop();
}

// ---------------------------------------------------------------------------

bool MusicPipeline::Initialize(HeliosOutput& laser,
                                const std::string& crepePath)
{
    laser_ = &laser;

    // Try to load CREPE pitch model (optional — melody/chroma modes need it)
    if (!crepePath.empty()) {
        if (pitchAnalyzer.Initialize(crepePath)) {
            pitchAnalyzer.SetSourceSampleRate(44100);
            pitchReady_ = true;
        } else {
            std::cout << "[MusicPipeline] CREPE model not found — "
                         "melody/chroma modes will use zero pitch.\n"
                         "  Run: python scripts/export_music_models.py\n";
        }
    }

    std::cout << "[MusicPipeline] Initialized\n";
    PrintModes();
    return true;
}

bool MusicPipeline::Start()
{
    if (running_) return true;
    if (!laser_) {
        std::cerr << "[MusicPipeline] Call Initialize() first\n";
        return false;
    }

    // Register audio callback — store features
    analyzer_->SetCallback([this](const AudioFeatures& f) {
        std::lock_guard<std::mutex> lock(featureMutex_);
        latestFeatures_ = f;
    });

    // Push raw audio hops to PitchAnalyzer (non-blocking, runs inference inline)
    if (pitchReady_) {
        analyzer_->SetRawCallback([this](const float* samples, int count, int /*sr*/) {
            pitchAnalyzer.PushSamples(samples, count);
        });
    }

    if (!analyzer_->Start()) {
        std::cerr << "[MusicPipeline] AudioAnalyzer failed to start\n";
        return false;
    }

    running_      = true;
    lastRenderSec_ = 0.0;
    renderThread_ = std::thread(&MusicPipeline::RenderLoop, this);
    std::cout << "[MusicPipeline] Started — mode: " << engine_->GetMode() << "\n";
    return true;
}

void MusicPipeline::Stop()
{
    running_ = false;
    if (renderThread_.joinable()) renderThread_.join();
    if (analyzer_) analyzer_->Stop();
    std::cout << "[MusicPipeline] Stopped\n";
}

void MusicPipeline::SetMode(const std::string& mode)
{
    engine_->SetMode(mode);
}

std::string MusicPipeline::GetMode() const
{
    return engine_->GetMode();
}

// ---------------------------------------------------------------------------
// Render loop
// ---------------------------------------------------------------------------

void MusicPipeline::RenderLoop()
{
    using clock = std::chrono::steady_clock;
    auto start  = clock::now();

    while (running_) {
        auto now = clock::now();
        double t = std::chrono::duration<double>(now - start).count();
        float  dt = static_cast<float>(t - lastRenderSec_);
        lastRenderSec_ = t;

        // Get latest audio features
        AudioFeatures feat;
        {
            std::lock_guard<std::mutex> lock(featureMutex_);
            feat = latestFeatures_;
        }

        // Merge CREPE pitch estimate into features
        if (pitchReady_)
            pitchAnalyzer.FillFeatures(feat);

        // Generate 3D geometry
        Frame3D frame3d = engine_->Generate(feat, t);

        // Apply beat-sync transforms
        beatSync.Update(feat, dt);
        Frame3D transformed = beatSync.Apply(frame3d);

        // Convert to laser and send
        auto laserFrame = Frame3DToLaser(transformed);
        if (!laserFrame.empty())
            laser_->SendFrame(laserFrame);

        // Yield — laser DAC thread handles actual timing
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

// ---------------------------------------------------------------------------
// Command handler
// ---------------------------------------------------------------------------

bool MusicPipeline::HandleCommand(const std::string& line)
{
    if (line.rfind("music", 0) != 0) return false;

    std::istringstream ss(line.substr(5));
    std::string sub;
    ss >> sub;

    if (sub.empty()) {
        std::cout << "[MusicPipeline] Current mode: " << engine_->GetMode() << "\n";
        PrintModes();
        return true;
    }

    if (sub == "stop") {
        Stop();
        return true;
    }

    if (sub == "list") {
        PrintModes();
        return true;
    }

    if (sub == "start") {
        Start();
        return true;
    }

    // Beat sync toggles
    if (sub == "norotate")  { beatSync.enableRotation  = false; return true; }
    if (sub == "rotate")    { beatSync.enableRotation  = true;  return true; }
    if (sub == "noscale")   { beatSync.enableScale     = false; return true; }
    if (sub == "scale")     { beatSync.enableScale     = true;  return true; }
    if (sub == "nohue")     { beatSync.enableHueShift  = false; return true; }
    if (sub == "hue")       { beatSync.enableHueShift  = true;  return true; }
    if (sub == "nobreathe") { beatSync.enableZBreathe  = false; return true; }
    if (sub == "breathe")   { beatSync.enableZBreathe  = true;  return true; }

    // Pitch status
    if (sub == "pitch") {
        if (pitchReady_) {
            float hz   = pitchAnalyzer.GetPitchHz();
            float conf = pitchAnalyzer.GetConfidence();
            if (hz > 0.0f)
                std::cout << "[MusicPipeline] Pitch: " << hz << " Hz  "
                             "conf: " << conf << "\n";
            else
                std::cout << "[MusicPipeline] Pitch: (unvoiced / silence)\n";
        } else {
            std::cout << "[MusicPipeline] CREPE not loaded — run "
                         "scripts/export_music_models.py first\n";
        }
        return true;
    }

    // Try as mode name
    auto modes = engine_->GetModeNames();
    bool found = false;
    for (const auto& m : modes) if (m == sub) { found = true; break; }

    if (found) {
        SetMode(sub);
        if (!IsRunning()) Start();
        return true;
    }

    std::cerr << "[MusicPipeline] Unknown command: " << sub << "\n";
    PrintModes();
    return true;
}

void MusicPipeline::PrintModes() const
{
    std::cout << "[MusicPipeline] Available modes:\n";
    for (const auto& m : engine_->GetModeNames())
        std::cout << "  music " << m << "\n";
    std::cout << "  music stop | start | list\n";
    std::cout << "  music rotate | norotate | scale | noscale | hue | nohue | breathe | nobreathe\n";
    std::cout << "  music pitch  (show current CREPE pitch estimate)\n";
    if (!pitchReady_)
        std::cout << "  [melody/chroma need CREPE: python scripts/export_music_models.py]\n";
}
