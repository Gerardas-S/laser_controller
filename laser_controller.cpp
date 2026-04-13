#define NOMINMAX
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <limits>
#include <algorithm>
#include <numeric>
#include "HeliosOutput.h"
#include "HEDInference.h"
#include "SAMInference.h"
#include "ILDAFile.h"
#include <opencv2/videoio.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <filesystem>
#include <functional>

namespace fs = std::filesystem;
static const fs::path kResourcesDir = "resources";

// -----------------------------------------------------------------------------
// Nearest-neighbour polyline reordering
// -----------------------------------------------------------------------------

using Frame = std::vector<std::vector<LaserPoint>>;

// Centroid of a polyline
static std::pair<float,float> Centroid(const std::vector<LaserPoint>& poly)
{
    float sx = 0, sy = 0;
    for (const auto& p : poly) { sx += p.x; sy += p.y; }
    float n = static_cast<float>(poly.size());
    return { sx / n, sy / n };
}

// ReorderFrame
//
// Step 1 — Spatial sort by centroid (scanline: top→bottom, left→right).
//   Deterministic across frames — same region of screen = same position
//   in the draw list, so similar frames draw in a consistent order.
//
// Step 2 — Nearest-neighbour within that sorted order, also considering
//   flipping each contour. Minimises galvo travel without sacrificing
//   the temporal stability gained in step 1.

Frame ReorderFrame(const Frame& frame)
{
    if (frame.size() <= 1) return frame;

    // Step 1: sort by centroid Y (primary) then X (secondary)
    // Quantise Y into coarse rows (~10% of screen height) so small
    // vertical jitter between frames doesn't constantly re-sort contours.
    const float kRowHeight = 0.1f;
    std::vector<int> idx(frame.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) {
        auto [ax, ay] = Centroid(frame[a]);
        auto [bx, by] = Centroid(frame[b]);
        int rowA = static_cast<int>((ay + 1.0f) / kRowHeight);
        int rowB = static_cast<int>((by + 1.0f) / kRowHeight);
        if (rowA != rowB) return rowA < rowB;
        return ax < bx;
    });

    // Step 2: nearest-neighbour with flip, preserving the sorted order
    // as a strong prior (only swap adjacent pairs that are clearly better)
    Frame sorted;
    sorted.reserve(frame.size());
    for (int i : idx) sorted.push_back(frame[i]);

    Frame result;
    result.reserve(sorted.size());
    std::vector<bool> visited(sorted.size(), false);

    // Start from first in spatial order
    visited[0] = true;
    result.push_back(sorted[0]);

    for (size_t step = 1; step < sorted.size(); ++step) {
        const LaserPoint& tail = result.back().back();
        float bestDist = std::numeric_limits<float>::max();
        int   bestIdx  = -1;
        bool  bestFlip = false;

        for (int j = 0; j < (int)sorted.size(); ++j) {
            if (visited[j] || sorted[j].empty()) continue;
            float dx, dy, d;

            dx = sorted[j].front().x - tail.x;
            dy = sorted[j].front().y - tail.y;
            d  = dx*dx + dy*dy;
            if (d < bestDist) { bestDist = d; bestIdx = j; bestFlip = false; }

            dx = sorted[j].back().x - tail.x;
            dy = sorted[j].back().y - tail.y;
            d  = dx*dx + dy*dy;
            if (d < bestDist) { bestDist = d; bestIdx = j; bestFlip = true; }
        }

        if (bestIdx < 0) break;
        visited[bestIdx] = true;

        if (bestFlip) {
            auto flipped = sorted[bestIdx];
            std::reverse(flipped.begin(), flipped.end());
            result.push_back(std::move(flipped));
        } else {
            result.push_back(sorted[bestIdx]);
        }
    }
    return result;
}

// -----------------------------------------------------------------------------
// Alignment circle
// -----------------------------------------------------------------------------

Frame Scene_AlignmentCircle()
{
    std::vector<LaserPoint> pts;
    for (int i = 0; i <= 300; ++i) {
        float a = 2.0f * 3.14159265f * i / 300;
        pts.push_back({ 0.5f * std::cos(a), 0.5f * std::sin(a), 1, 1, 1 });
    }
    return { pts };
}

// -----------------------------------------------------------------------------
// Resources scanner — returns sorted list of .ild paths in resources/
// -----------------------------------------------------------------------------

std::vector<fs::path> ScanResources()
{
    std::vector<fs::path> files;
    if (!fs::exists(kResourcesDir)) return files;
    for (const auto& entry : fs::directory_iterator(kResourcesDir)) {
        if (entry.is_regular_file() &&
            entry.path().extension() == ".ild")
            files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());
    return files;
}

// -----------------------------------------------------------------------------
// Shared state
// -----------------------------------------------------------------------------

std::atomic<int>  g_sceneIndex{ 0 };   // 0 = circle, 7 = animation
std::atomic<bool> g_running{ true };
std::atomic<bool> g_configDirty{ false };

HeliosConfig g_config;
std::mutex   g_configMutex;

std::unique_ptr<HEDInference> g_hed;
std::unique_ptr<SAMInference> g_sam;

std::vector<Frame>  g_videoAnimation;
std::mutex          g_animationMutex;
std::atomic<bool>   g_videoProcessing{ false };
std::atomic<bool>   g_videoReady{ false };
std::string         g_nowPlaying = "alignment circle";

std::thread g_videoThread;

enum class VideoMode { HED, SAM };
VideoMode g_videoMode = VideoMode::HED;

// -----------------------------------------------------------------------------
// Video thread
// -----------------------------------------------------------------------------

void VideoThread(std::string videoPath, VideoMode mode)
{
    g_videoReady      = false;
    g_videoProcessing = true;

    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "[video] Failed to open: " << videoPath << "\n";
        g_videoProcessing = false;
        return;
    }

    int total = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    std::cout << "[video] Opened (" << total << " frames) — processing...\n";

    std::vector<Frame> animation;
    animation.reserve(total > 0 ? total : 256);

    int fi = 0;
    while (g_running && g_videoProcessing) {
        cv::Mat frame;
        if (!cap.read(frame)) break;

        Frame lf;
        if (mode == VideoMode::HED && g_hed) lf = g_hed->ProcessFrame(frame);
        else if (mode == VideoMode::SAM && g_sam) lf = g_sam->ProcessFrame(frame);

        if (!lf.empty()) animation.push_back(ReorderFrame(lf));

        if (++fi % 10 == 0)
            std::cout << "[video] " << fi << " / " << total << "\r" << std::flush;
    }

    std::cout << "\n[video] " << animation.size() << " frames done.\n";

    // Save to resources/
    fs::create_directories(kResourcesDir);
    fs::path ildPath = kResourcesDir /
                       fs::path(videoPath).stem().concat(".ild");
    ILDAFile::Save(ildPath.string(), animation);

    {
        std::lock_guard<std::mutex> lock(g_animationMutex);
        g_videoAnimation = std::move(animation);
    }

    g_nowPlaying      = ildPath.filename().string();
    g_videoProcessing = false;
    g_videoReady      = true;
    g_sceneIndex      = 7;
    std::cout << "[video] Playing. Type 0 for circle or pick another file.\n> ";
}

// -----------------------------------------------------------------------------
// Render thread
// -----------------------------------------------------------------------------

void RenderThread(HeliosOutput& laser)
{
    using clock = std::chrono::steady_clock;

    Frame alignCircle = Scene_AlignmentCircle();
    int   animFrame   = 0;
    auto  lastAdvance = clock::now();

    while (g_running)
    {
        if (g_configDirty.exchange(false)) {
            std::lock_guard<std::mutex> lock(g_configMutex);
            laser.SetConfig(g_config);
        }

        if (g_sceneIndex == 7 && g_videoReady) {
            auto   now     = clock::now();
            double elapsed = std::chrono::duration<double>(now - lastAdvance).count();
            double frameDt = 1.0 / g_config.target_fps;

            if (elapsed >= frameDt) {
                {
                    std::lock_guard<std::mutex> lock(g_animationMutex);
                    int n = static_cast<int>(g_videoAnimation.size());
                    if (n > 0) animFrame = (animFrame + 1) % n;
                }
                lastAdvance = now;
            }

            std::lock_guard<std::mutex> lock(g_animationMutex);
            int n = static_cast<int>(g_videoAnimation.size());
            if (n > 0) { laser.SendFrame(g_videoAnimation[animFrame % n]); }
            else        { laser.SendFrame(alignCircle); }
        } else {
            // Circle: processing in progress, or user selected circle
            if (g_sceneIndex == 7 && !g_videoReady)
                animFrame = 0;  // reset so playback starts from frame 0
            laser.SendFrame(alignCircle);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

// -----------------------------------------------------------------------------
// Menu
// -----------------------------------------------------------------------------

void PrintConfig(const HeliosConfig& c)
{
    std::cout << "\n--- Config ---\n";
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "  target_fps       = " << c.target_fps       << "\n";
    std::cout << "  max_pps          = " << c.max_pps          << "\n";
    std::cout << "  blank_points     = " << c.blank_points     << "\n";
    std::cout << "  pre_on_points    = " << c.pre_on_points    << "\n";
    std::cout << "  post_on_points   = " << c.post_on_points   << "\n";
    std::cout << "  min_vertex_hold  = " << c.min_vertex_hold  << "\n";
    std::cout << "  max_vertex_hold  = " << c.max_vertex_hold  << "\n";
    std::cout << "  curve_threshold  = " << c.curve_threshold  << "\n";
    std::cout << "--------------\n";
}

void PrintMenu()
{
    auto files = ScanResources();

    std::cout << "\n";

    // Playlist
    std::string playing = g_nowPlaying;
    auto mark = [&](const std::string& name) {
        return (name == playing) ? " <<" : "";
    };

    std::cout << "  [0] alignment circle" << mark("alignment circle") << "\n";
    for (int i = 0; i < (int)files.size(); ++i) {
        std::string fname = files[i].filename().string();
        std::cout << "  [" << (i + 1) << "] " << fname << mark(fname) << "\n";
    }

    if (g_videoProcessing)
        std::cout << "  [ processing... ]\n";

    std::cout << "\n";
    std::cout << "  video hed <path>   process video -> save to resources/ -> play\n";
    std::cout << "  video sam <path>\n";
    std::cout << "  stop               back to circle\n";
    std::cout << "  set <param> <val>  tune laser params\n";
    std::cout << "  config             show current config\n";
    std::cout << "  refresh            rescan resources/\n";
    std::cout << "  q                  quit\n";
    std::cout << "> ";
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main()
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);

    fs::create_directories(kResourcesDir);

    g_config.target_fps      = 30;
    g_config.max_pps         = 30000;
    g_config.blank_points    = 20;
    g_config.pre_on_points   = 8;
    g_config.post_on_points  = 8;
    g_config.min_vertex_hold = 0;
    g_config.max_vertex_hold = 20;
    g_config.curve_threshold = 20.0f;
    g_config.enable_reorder  = false;

    HeliosOutput laser;
    if (!laser.Initialize(g_config)) {
        std::cerr << "Failed to initialize laser\n";
        return 1;
    }

    g_sceneIndex = 0;
    std::thread renderThread(RenderThread, std::ref(laser));

    PrintConfig(g_config);
    PrintMenu();

    std::string line;
    while (std::getline(std::cin, line))
    {
        if (line.empty()) { PrintMenu(); continue; }

        if (line == "q") { g_running = false; break; }

        if (line == "config") {
            std::lock_guard<std::mutex> lock(g_configMutex);
            PrintConfig(g_config);
            PrintMenu();
            continue;
        }

        if (line == "refresh") { PrintMenu(); continue; }

        if (line == "stop") {
            g_videoProcessing = false;
            if (g_videoThread.joinable()) g_videoThread.join();
            g_videoReady = false;
            { std::lock_guard<std::mutex> lock(g_animationMutex);
              g_videoAnimation.clear(); }
            g_sceneIndex  = 0;
            g_nowPlaying  = "alignment circle";
            std::cout << "Stopped.\n";
            PrintMenu();
            continue;
        }

        // Numbered selection — pick from playlist
        bool isNumber = !line.empty() &&
                        std::all_of(line.begin(), line.end(), ::isdigit);
        if (isNumber) {
            int idx = std::stoi(line);
            if (idx == 0) {
                g_sceneIndex = 0;
                g_nowPlaying = "alignment circle";
                std::cout << "Alignment circle.\n";
                PrintMenu();
                continue;
            }

            auto files = ScanResources();
            if (idx < 1 || idx > (int)files.size()) {
                std::cerr << "No file at [" << idx << "]. Try refresh.\n";
                PrintMenu();
                continue;
            }

            std::string path = files[idx - 1].string();
            auto anim = ILDAFile::Load(path);
            if (anim.empty()) {
                std::cerr << "Failed to load: " << path << "\n";
                PrintMenu();
                continue;
            }

            if (g_videoThread.joinable()) {
                g_videoProcessing = false;
                g_videoReady      = false;
                g_videoThread.join();
            }
            {
                std::lock_guard<std::mutex> lock(g_animationMutex);
                g_videoAnimation = std::move(anim);
            }
            g_nowPlaying = files[idx - 1].filename().string();
            g_videoReady = true;
            g_sceneIndex = 7;
            PrintMenu();
            continue;
        }

        // Video processing
        if (line.rfind("video ", 0) == 0) {
            std::istringstream vss(line.substr(6));
            std::string modeStr, path;
            vss >> modeStr;
            std::getline(vss, path);
            if (!path.empty() && path.front() == ' ') path = path.substr(1);

            if (modeStr != "hed" && modeStr != "sam") {
                std::cerr << "Usage: video hed <path>  OR  video sam <path>\n";
                PrintMenu(); continue;
            }

            if (g_videoThread.joinable()) {
                g_videoProcessing = false;
                g_videoReady      = false;
                g_videoThread.join();
            }

            VideoMode mode = (modeStr == "sam") ? VideoMode::SAM : VideoMode::HED;

            if (mode == VideoMode::HED && !g_hed) {
                g_hed = std::make_unique<HEDInference>();
                if (!g_hed->Initialize(HEDInference::Backend::CPU) ||
                    !g_hed->LoadModel(L"models/hed/model.onnx")) {
                    std::cerr << "HED init failed\n";
                    g_hed.reset(); PrintMenu(); continue;
                }
            }
            if (mode == VideoMode::SAM && !g_sam) {
                g_sam = std::make_unique<SAMInference>();
                if (!g_sam->Initialize(SAMInference::Backend::CPU) ||
                    !g_sam->LoadModels(L"models/sam/encoder.onnx",
                                       L"models/sam/decoder.onnx")) {
                    std::cerr << "SAM init failed\n";
                    g_sam.reset(); PrintMenu(); continue;
                }
            }

            g_videoMode   = mode;
            g_sceneIndex  = 7;   // show circle while processing
            g_nowPlaying  = "alignment circle";
            g_videoThread = std::thread(VideoThread, path, mode);
            std::cout << "Processing (" << modeStr << "): " << path << "\n";
            std::cout << "Circle on laser until done. Console remains live.\n";
            PrintMenu();
            continue;
        }

        // Parameter tuning
        if (line.rfind("set ", 0) == 0) {
            std::istringstream ss(line.substr(4));
            std::string param; float value;
            if (!(ss >> param >> value)) {
                std::cout << "Usage: set <param> <value>\n";
                PrintMenu(); continue;
            }
            {
                std::lock_guard<std::mutex> lock(g_configMutex);
                bool found = true;
                if      (param == "target_fps")     g_config.target_fps      = (int)value;
                else if (param == "max_pps")         g_config.max_pps         = (int)value;
                else if (param == "blank_points")    g_config.blank_points    = (int)value;
                else if (param == "pre_on_points")   g_config.pre_on_points   = (int)value;
                else if (param == "post_on_points")  g_config.post_on_points  = (int)value;
                else if (param == "min_vertex_hold") g_config.min_vertex_hold = (int)value;
                else if (param == "max_vertex_hold") g_config.max_vertex_hold = (int)value;
                else if (param == "curve_threshold") g_config.curve_threshold = value;
                else if (param == "reorder")         g_config.enable_reorder  = (value > 0.5f);
                else { std::cout << "Unknown param: " << param << "\n"; found = false; }
                if (found) { std::cout << "Set " << param << " = " << value << "\n";
                             g_configDirty = true; }
            }
            PrintMenu(); continue;
        }

        std::cout << "Unknown command.\n";
        PrintMenu();
    }

    g_videoProcessing = false;
    if (g_videoThread.joinable()) g_videoThread.join();
    g_running = false;
    renderThread.join();
    laser.Close();
    return 0;
}
