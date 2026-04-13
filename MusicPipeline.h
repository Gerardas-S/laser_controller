#pragma once

#include "AudioAnalyzer.h"
#include "MusicGeometryEngine.h"
#include "BeatSyncEngine.h"
#include "PitchAnalyzer.h"
#include "GeometryPoint3D.h"
#include "HeliosOutput.h"
#include <atomic>
#include <thread>
#include <mutex>
#include <string>
#include <memory>

// -----------------------------------------------------------------------------
// MusicPipeline
//
// Top-level orchestrator for the music-driven geometry branch.
// Owns: AudioAnalyzer -> MusicGeometryEngine -> BeatSyncEngine -> laser
//
// Designed to be a drop-in addition to laser_controller.cpp alongside
// the video pipeline. Uses the same HeliosOutput instance.
//
// Usage:
//   MusicPipeline pipeline;
//   pipeline.Initialize(laser);
//   pipeline.SetMode("lissajous");
//   pipeline.Start();
//   // ... laser runs in background thread
//   pipeline.Stop();
//
// Console commands (call from main loop):
//   pipeline.HandleCommand("music lissajous");
//   pipeline.HandleCommand("music torusknot");
//   pipeline.HandleCommand("music stop");
// -----------------------------------------------------------------------------

class MusicPipeline {
public:
    MusicPipeline();
    ~MusicPipeline();

    // Initialize with shared laser output.
    // Optionally loads CREPE pitch model for melody/chroma modes.
    bool Initialize(HeliosOutput& laser,
                    const std::string& crepePath = "models/crepe/model.onnx");

    // Start/stop audio capture and render loop.
    bool Start();
    void Stop();
    bool IsRunning() const { return running_; }

    // Switch geometry mode.
    void SetMode(const std::string& mode);
    std::string GetMode() const;

    // Handle a console command. Returns true if command was consumed.
    // Format: "music <mode>"  or  "music stop"  or  "music list"
    bool HandleCommand(const std::string& line);

    // Print available modes to stdout.
    void PrintModes() const;

    // BeatSyncEngine exposed for runtime tuning.
    BeatSyncEngine beatSync;

    // PitchAnalyzer — loaded only when CREPE model is present.
    // Provides feat.pitch / feat.pitchNorm for melody & chroma modes.
    PitchAnalyzer  pitchAnalyzer;

private:
    void RenderLoop();

    HeliosOutput*                     laser_   = nullptr;
    std::unique_ptr<AudioAnalyzer>    analyzer_;
    std::unique_ptr<MusicGeometryEngine> engine_;

    std::atomic<bool>  running_{ false };
    bool               pitchReady_ = false;
    std::thread        renderThread_;

    // Latest audio features (pushed by AudioAnalyzer callback)
    AudioFeatures      latestFeatures_;
    mutable std::mutex featureMutex_;

    double             renderTime_ = 0.0;
    double             lastRenderSec_ = 0.0;
};
