#define WIN32_LEAN_AND_MEAN
#include <windows.h>
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
#include <cstdio>
#include "HeliosOutput.h"
#include "ILDAFile.h"
#include <opencv2/videoio.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <filesystem>
#include <functional>

namespace fs = std::filesystem;
// Directory containing the running executable — all relative paths anchor here.
static fs::path ExeDir()
{
    wchar_t buf[MAX_PATH];
    GetModuleFileNameW(nullptr, buf, MAX_PATH);
    return fs::path(buf).parent_path();
}

static const fs::path kProjectDir   = ExeDir().parent_path().parent_path(); // x64/Debug/../../
static const fs::path kResourcesDir = kProjectDir / "resources";
// Python executable — use the project venv so torch+CUDA and sam2 are available.
// Falls back to bare "python" if the venv does not exist yet.
static const fs::path kVenvPython   = kProjectDir / ".venv" / "Scripts" / "python.exe";
static const std::string kPython    = []() -> std::string {
    std::error_code ec;
    if (!fs::exists(kVenvPython, ec)) return "python";
    // Use forward slashes — backslashes before closing " confuse cmd.exe parsing.
    std::string p = kVenvPython.generic_string();
    return "\"" + p + "\"";
}();

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

//Frame ReorderFrame(const Frame& frame)
//{
//    if (frame.size() <= 1) return frame;
//
//    // Step 1: sort by centroid Y (primary) then X (secondary)
//    // Quantise Y into coarse rows (~10% of screen height) so small
//    // vertical jitter between frames doesn't constantly re-sort contours.
//    const float kRowHeight = 0.1f;
//    std::vector<int> idx(frame.size());
//    std::iota(idx.begin(), idx.end(), 0);
//    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) {
//        auto [ax, ay] = Centroid(frame[a]);
//        auto [bx, by] = Centroid(frame[b]);
//        int rowA = static_cast<int>((ay + 1.0f) / kRowHeight);
//        int rowB = static_cast<int>((by + 1.0f) / kRowHeight);
//        if (rowA != rowB) return rowA < rowB;
//        return ax < bx;
//    });
//
//    // Step 2: nearest-neighbour with flip, preserving the sorted order
//    // as a strong prior (only swap adjacent pairs that are clearly better)
//    Frame sorted;
//    sorted.reserve(frame.size());
//    for (int i : idx) sorted.push_back(frame[i]);
//
//    Frame result;
//    result.reserve(sorted.size());
//    std::vector<bool> visited(sorted.size(), false);
//
//    // Start from first in spatial order
//    visited[0] = true;
//    result.push_back(sorted[0]);
//
//    for (size_t step = 1; step < sorted.size(); ++step) {
//        const LaserPoint& tail = result.back().back();
//        float bestDist = std::numeric_limits<float>::max();
//        int   bestIdx  = -1;
//        bool  bestFlip = false;
//
//        for (int j = 0; j < (int)sorted.size(); ++j) {
//            if (visited[j] || sorted[j].empty()) continue;
//            float dx, dy, d;
//
//            dx = sorted[j].front().x - tail.x;
//            dy = sorted[j].front().y - tail.y;
//            d  = dx*dx + dy*dy;
//            if (d < bestDist) { bestDist = d; bestIdx = j; bestFlip = false; }
//
//            dx = sorted[j].back().x - tail.x;
//            dy = sorted[j].back().y - tail.y;
//            d  = dx*dx + dy*dy;
//            if (d < bestDist) { bestDist = d; bestIdx = j; bestFlip = true; }
//        }
//
//        if (bestIdx < 0) break;
//        visited[bestIdx] = true;
//
//        if (bestFlip) {
//            auto flipped = sorted[bestIdx];
//            std::reverse(flipped.begin(), flipped.end());
//            result.push_back(std::move(flipped));
//        } else {
//            result.push_back(sorted[bestIdx]);
//        }
//    }
//    return result;
//}

// -----------------------------------------------------------------------------
// Alignment circle
// -----------------------------------------------------------------------------

Frame Scene_AlignmentCircle()
{
    const int   N      = 300;
    const float R      = 0.5f;
    const float TWO_PI = 2.0f * 3.14159265f;

    std::vector<LaserPoint> pts;
    pts.reserve(N);
    for (int i = 0; i < N; ++i) {
        float a = TWO_PI * i / N;
        pts.push_back({ R * std::cos(a), R * std::sin(a), 1, 1, 1 });
    }
    return { pts };
}

// -----------------------------------------------------------------------------
// Resources scanner — returns sorted list of .ild paths
// Checks resources/ (legacy) and resources/animations/ (new pipeline)
// -----------------------------------------------------------------------------

std::vector<fs::path> ScanResources()
{
    std::vector<fs::path> files;
    auto scanDir = [&](const fs::path& dir) {
        if (!fs::exists(dir)) return;
        for (const auto& entry : fs::directory_iterator(dir))
            if (entry.is_regular_file() && entry.path().extension() == ".ild")
                files.push_back(entry.path());
    };
    scanDir(kResourcesDir);
    scanDir(kResourcesDir / "animations");
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

// Logical polyline animation buffer — kept for runtime-generated content
// (alignment circle).  Played via HeliosOutput::SendFrame which synthesises
// blanking/dwells live.  In-process direct-inference paths were removed;
// the SAM2 pipeline (segment → vectorize → encode) is the sole
// source of video-derived ILDA files.
std::vector<Frame>  g_videoAnimation;

// Physically-baked animation from encode.py — every point in the buffer is a
// point the DAC will play, in 12-bit ILDA coords.  Played via SendPhysical.
std::vector<std::vector<HeliosPoint>> g_videoAnimationPhysical;

std::mutex          g_animationMutex;
std::atomic<bool>   g_videoProcessing{ false };
std::atomic<bool>   g_videoReady{ false };
std::atomic<bool>   g_videoBaked{ false };   // which buffer is active
std::string         g_nowPlaying = "alignment circle";
float               g_animationScale = 1.0f;
float               g_animationBrightness = 1.0f;

std::thread g_videoThread;

enum class VideoMode { SAM2, SAM2_CANNY, SAM2_HED, SAM2_SEG, SAM2_ALL };
VideoMode g_videoMode = VideoMode::SAM2;

// -----------------------------------------------------------------------------
// Video thread
// -----------------------------------------------------------------------------

void VideoThread(std::string videoPath, VideoMode mode)
{
    g_videoReady      = false;
    g_videoProcessing = true;

    fs::create_directories(kResourcesDir);

    // -------------------------------------------------------------------------
    // SAM2 variants — three-stage pipeline
    //
    //  Stage 1  segment.py   : video  → masks.npz          (cached per video+model)
    //  Stage 2  vectorize.py : masks  → polylines.json      (per method)
    //  Stage 3  encode.py    : polys  → animation.ild       (temporal filter + ILDA)
    //
    //  Output directories:
    //    resources/masks/       {stem}_sam2-{model}.npz
    //    resources/polylines/   {stem}_sam2-{model}_{method}.json
    //    resources/animations/  {stem}_sam2-{model}_{method}.ild
    // -------------------------------------------------------------------------
    {
        // --- Path setup ---
        const std::string kModel    = "tiny";   // SAM2 model size (change here to upgrade)
        const std::string kDevice   = "cuda";

        std::string videoStem = fs::path(videoPath).stem().string();
        std::string maskTag   = videoStem + "_sam2-" + kModel;

        fs::path masksDir     = kResourcesDir / "masks";
        fs::path polylinesDir = kResourcesDir / "polylines";
        fs::path animsDir     = kResourcesDir / "animations";

        fs::create_directories(masksDir);
        fs::create_directories(polylinesDir);
        fs::create_directories(animsDir);

        fs::path segScript = kProjectDir / "scripts" / "segment.py";
        fs::path vecScript = kProjectDir / "scripts" / "vectorize.py";
        fs::path encScript = kProjectDir / "scripts" / "encode.py";
        fs::path ckptDir   = kProjectDir / "models"  / "sam2";
        fs::path masksPath  = masksDir / (maskTag + ".npz");

        // Map VideoMode → interior method name used in filenames and --method arg
        auto ModeMethod = [&]() -> std::string {
            switch (mode) {
                case VideoMode::SAM2_CANNY: return "canny";
                case VideoMode::SAM2_HED:   return "hed";
                case VideoMode::SAM2_SEG:   return "seg";
                default:                    return "all";   // SAM2, SAM2_ALL
            }
        };
        const std::string method = ModeMethod();

        // --- Generic script runner: launches subprocess, streams stdout, returns success ---
        auto RunScript = [&](const std::string& label, const std::string& cmd) -> bool
        {
                std::cout << '\n' << label << '\n' << std::flush;

                // Torch native libs shipped inside the venv (ensure this path exists)
                std::string torchLib = (kProjectDir / ".venv" / "Lib" / "site-packages" / "torch" / "lib").generic_string();

                // Save current PATH and set a new PATH with torchLib prefixed
                char* oldPathC = nullptr;
                size_t oldPathLen = 0;
                _dupenv_s(&oldPathC, &oldPathLen, "PATH");
                std::string oldPath = oldPathC ? oldPathC : "";
                free(oldPathC);
                std::string newPath = torchLib + ";" + oldPath;
                SetEnvironmentVariableA("PATH", newPath.c_str());

                // Run the command via shell and stream output
                // Keep the same quoting behavior you used before for cmd
                std::string wrapped = "\"" + cmd + " 2>&1\"";
                FILE* pipe = _popen(wrapped.c_str(), "r");
                if (!pipe) {
                    std::cerr << label << " Failed to launch subprocess\n";
                    // restore PATH before returning
                    SetEnvironmentVariableA("PATH", oldPath.c_str());
                    return false;
                }

                char buf[512];
                while (fgets(buf, sizeof(buf), pipe))
                    std::cout << buf << std::flush;

                int ret = _pclose(pipe);

                // Restore original PATH
                SetEnvironmentVariableA("PATH", oldPath.c_str());

                if (ret != 0) {
                    std::cerr << label << " Script exited with code " << ret << '\n';
                }
                return ret == 0;
        };


        // ── Stage 1: Segment ─────────────────────────────────────────────────
        // Skip if masks file already exists (cache).
        if (!fs::exists(masksPath)) {
            std::string cmd =
                kPython + " \"" + segScript.string()  + "\""
                " --video \""          + videoPath           + "\""
                " --output \""         + masksPath.string()  + "\""
                " --model "            + kModel              +
                " --checkpoint-dir \"" + ckptDir.string()    + "\""
                " --device "           + kDevice;
            if (!RunScript("[segment]", cmd)) {
                g_videoProcessing = false; return;
            }
        } else {
            std::cout << "[segment] Using cached masks: "
                      << masksPath.filename() << '\n' << std::flush;
        }

        if (!g_videoProcessing) return;

        // ── Stage 2: Vectorize ────────────────────────────────────────────────
        // For 'all' mode, vectorize.py writes one JSON per method;
        // the output arg is the base path and suffixes are inserted by the script.
        fs::path vecBasePath = polylinesDir / (maskTag + (method == "all" ? ".json"
                                                        : "_" + method + ".json"));
        {
            std::string cmd =
                kPython + " \"" + vecScript.string()    + "\""
                " --video \""        + videoPath             + "\""
                " --masks \""        + masksPath.string()    + "\""
                " --output \""       + vecBasePath.string()  + "\""
                " --method "         + method                +
                " --device "         + kDevice;
            if (!RunScript("[vectorize]", cmd)) {
                g_videoProcessing = false; return;
            }
        }

        if (!g_videoProcessing) return;

        // ── Stage 3: Encode ───────────────────────────────────────────────────
        // For 'all' mode, run encode.py once per method JSON.
        // For single methods, run once.
        static const std::vector<std::string> kAllMethods = {"canny","hed"};
        const std::vector<std::string> encMethods =
            (method == "all") ? kAllMethods : std::vector<std::string>{method};

        for (const auto& m : encMethods) {
            if (!g_videoProcessing) break;

            fs::path polyPath = polylinesDir / (maskTag + "_" + m + ".json");
            fs::path animPath = animsDir     / (maskTag + "_" + m + ".ild");

            if (!fs::exists(polyPath)) {
                std::cout << "[encode] Skipping " << m
                          << " (polylines file not found)\n" << std::flush;
                continue;
            }
            std::string cmd =
                kPython + " \"" + encScript.string()   + "\""
                " --polylines \""  + polyPath.string() + "\""
                " --output \""     + animPath.string() + "\"";
            RunScript("[encode:" + m + "]", cmd);
        }

        if (!g_videoProcessing) return;

        // ── Load and play ─────────────────────────────────────────────────────
        // Priority order for 'all' mode: hed > canny
        static const std::vector<std::string> kPlayPriority = {"hed","canny"};
        const std::vector<std::string>& candidates =
            (method == "all") ? kPlayPriority : encMethods;

        std::vector<std::vector<HeliosPoint>> finalAnim;   // physically baked
        std::string        finalName;
        fs::path           finalPath;

        for (const auto& m : candidates) {
            fs::path animPath = animsDir / (maskTag + "_" + m + ".ild");
            if (!fs::exists(animPath)) continue;
            auto anim = ILDAFile::LoadFlat(animPath.string());
            if (!anim.empty()) {
                finalAnim = std::move(anim);
                finalName = animPath.filename().string();
                finalPath = animPath;
                break;
            }
        }

        if (finalAnim.empty()) {
            std::cerr << "[sam2] No animation produced.\n";
            g_videoProcessing = false;
            return;
        }

        {
            std::lock_guard<std::mutex> lock(g_animationMutex);
            g_videoAnimationPhysical = std::move(finalAnim);
            g_videoAnimation.clear();
            g_videoBaked = true;
        }
        g_nowPlaying      = finalName;
        g_videoProcessing = false;
        g_videoReady      = true;
        g_sceneIndex      = 7;

        if (method == "all")
            std::cout << "\n[sam2-all] Done — 2 ILDs in resources/animations/  "
                         "type 'load' to switch between them.\n> " << std::flush;
        else
            std::cout << "[sam2] Playing " << finalName << "\n> " << std::flush;
        return;
    }

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
                    int n = g_videoBaked
                                ? static_cast<int>(g_videoAnimationPhysical.size())
                                : static_cast<int>(g_videoAnimation.size());
                    if (n > 0) animFrame = (animFrame + 1) % n;
                }
                lastAdvance = now;
            }

            std::lock_guard<std::mutex> lock(g_animationMutex);
            if (g_videoBaked) {
                int n = static_cast<int>(g_videoAnimationPhysical.size());
                if (n > 0) {
                    const auto& frame = g_videoAnimationPhysical[animFrame % n];
                    if (g_animationScale != 1.0f) {
                        std::vector<HeliosPoint> scaled = frame;
                        for (auto& p : scaled) {
                            //scale around center 2048, 2048
                            float cx = 2048.0f, cy = 2048.0f;
                            p.x = static_cast<uint16_t>(std::clamp(cx + (p.x - cx) * g_animationScale, 0.0f, 4095.0f));
                            p.y = static_cast<uint16_t>(std::clamp(cy + (p.y - cy) * g_animationScale, 0.0f, 4095.0f));
                        }
                        laser.SendPhysical(scaled);
                    }
                    else {
                        laser.SendPhysical(frame);
                    }
                }
                else {
                    laser.SendFrame(alignCircle);
                }
            } else {
                int n = static_cast<int>(g_videoAnimation.size());
                if (n > 0) laser.SendFrame(g_videoAnimation[animFrame % n]);
                else       laser.SendFrame(alignCircle);
            }
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

void PrintMenu(const HeliosOutput& laser)
{
    auto files = ScanResources();

    std::cout << "\n";
    std::cout << "  DAC: " << (laser.IsConnected() ? "connected" : "offline") << "\n\n";

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
    std::cout << "  connect / disconnect\n\n";
    std::cout << "  --- SAM2 pipeline ---\n";
    std::cout << "  process  <video> [prompt]         Full pipeline: segment + vectorize(all) + encode(all)\n";
    std::cout << "  --- Individual stages ---\n";
    std::cout << "  segment  <video> [prompt]        Stage 1: video -> masks  (cached)\n";
    std::cout << "  preview-masks <video>             View segmentation masks\n";
    std::cout << "  vectorize <video> <method>        Stage 2: masks -> polylines\n";
    std::cout << "                                      method: canny | hed | edter | diffusion_edge\n";
    std::cout << "  preview-polylines <video> <method> View vectorized paths\n";
    std::cout << "  encode   <video> <method>         Stage 3: polylines -> ILDA + play\n\n";
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
    if (!laser.Initialize(g_config))
        std::cout << "[DAC] No DAC found — running offline. Type 'connect' to retry.\n";

    g_sceneIndex = 0;
    std::thread renderThread(RenderThread, std::ref(laser));

    PrintConfig(g_config);
    PrintMenu(laser);

    std::string line;
    while (std::getline(std::cin, line))
    {
        if (line.empty()) { PrintMenu(laser); continue; }

        if (line == "q") { g_running = false; break; }

        if (line == "connect") {
            if (laser.IsConnected()) std::cout << "Already connected.\n";
            else if (laser.Connect()) std::cout << "DAC connected.\n";
            else std::cout << "No DAC found.\n";
            PrintMenu(laser); continue;
        }

        if (line == "disconnect") {
            laser.Disconnect();
            PrintMenu(laser); continue;
        }

        if (line == "config") {
            std::lock_guard<std::mutex> lock(g_configMutex);
            PrintConfig(g_config);
            PrintMenu(laser);
            continue;
        }

        if (line == "refresh") { PrintMenu(laser); continue; }

        if (line == "stop") {
            g_videoProcessing = false;
            if (g_videoThread.joinable()) g_videoThread.join();
            g_videoReady = false;
            { std::lock_guard<std::mutex> lock(g_animationMutex);
              g_videoAnimation.clear();
              g_videoAnimationPhysical.clear();
              g_videoBaked = false; }
            g_sceneIndex  = 0;
            g_nowPlaying  = "alignment circle";
            std::cout << "Stopped.\n";
            PrintMenu(laser);
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
                PrintMenu(laser);
                continue;
            }

            auto files = ScanResources();
            if (idx < 1 || idx > (int)files.size()) {
                std::cerr << "No file at [" << idx << "]. Try refresh.\n";
                PrintMenu(laser);
                continue;
            }

            std::string path = files[idx - 1].string();

            if (g_videoThread.joinable()) {
                g_videoProcessing = false;
                g_videoReady      = false;
                g_videoThread.join();
            }

            // Auto-detect baked vs logical .ild and dispatch to the correct loader
            bool baked = ILDAFile::IsBaked(path);
            if (baked) {
                auto anim = ILDAFile::LoadFlat(path);
                if (anim.empty()) {
                    std::cerr << "Failed to load (baked): " << path << "\n";
                    PrintMenu(laser); continue;
                }
                std::lock_guard<std::mutex> lock(g_animationMutex);
                g_videoAnimationPhysical = std::move(anim);
                g_videoAnimation.clear();
                g_videoBaked = true;
            } else {
                auto anim = ILDAFile::Load(path);
                if (anim.empty()) {
                    std::cerr << "Failed to load (logical): " << path << "\n";
                    PrintMenu(laser); continue;
                }
                std::lock_guard<std::mutex> lock(g_animationMutex);
                g_videoAnimation         = std::move(anim);
                g_videoAnimationPhysical.clear();
                g_videoBaked             = false;
            }

            g_nowPlaying = files[idx - 1].filename().string();
            g_videoReady = true;
            g_sceneIndex = 7;
            PrintMenu(laser);
            continue;
        }

        // ── SAM2 pipeline helpers ─────────────────────────────────────────────
        // Shared path computation: given a video path, derive all stage paths.
        // model tag is hardcoded here — change kSam2Model to upgrade.
        static const std::string kSam2Model  = "tiny";
        static const std::string kSam2Device = "cuda";

        // Resolve a video argument: if the file exists as-is use it, otherwise
        // look inside resources/.  Allows typing just "clip.mp4" instead of
        // "resources/clip.mp4".
        auto ResolveVideo = [&](const std::string& arg) -> std::string {
            if (fs::exists(arg)) return arg;
            fs::path candidate = kResourcesDir / arg;
            if (fs::exists(candidate)) return candidate.generic_string();
            return arg;   // return as-is; downstream will report file-not-found
        };

        auto PipelinePaths = [&](const std::string& videoPath, const std::string& method)
            -> std::tuple<fs::path, fs::path, fs::path>
        {
            // returns {masksPath, polylinesPath, animPath}
            std::string stem    = fs::path(videoPath).stem().string();
            std::string maskTag = stem + "_sam2-" + kSam2Model;
            fs::path masksPath  = kResourcesDir / "masks"     / (maskTag + ".npz");
            fs::path polyPath   = kResourcesDir / "polylines" / (maskTag + "_" + method + ".json");
            fs::path animPath   = kResourcesDir / "animations"/ (maskTag + "_" + method + ".ild");
            return {masksPath, polyPath, animPath};
        };

        // Generic subprocess runner used by all three preview commands.
        // Blocking — streams stdout until script exits.
        auto RunSync = [&](const std::string& cmd) {
            std::string wrapped = "\"" + cmd + " 2>&1\"";
            FILE* pipe = _popen(wrapped.c_str(), "r");
            if (!pipe) { std::cerr << "Failed to launch subprocess\n"; return; }
            char buf[512];
            while (fgets(buf, sizeof(buf), pipe))
                std::cout << buf << std::flush;
            _pclose(pipe);
        };

        // ── process <video> [prompt] — full pipeline: segment → vectorize×4 → encode×4
        if (line.rfind("process ", 0) == 0) {
            std::istringstream ss(line.substr(8));
            std::string videoPath, prompt;
            ss >> videoPath;
            videoPath = ResolveVideo(videoPath);
            std::getline(ss, prompt);
            if (!prompt.empty() && prompt.front() == ' ') prompt = prompt.substr(1);

            if (videoPath.empty()) {
                std::cerr << "Usage: process <video> [prompt]\n";
                PrintMenu(laser); continue;
            }

            // Build all paths up-front (method doesn't affect masks/segment)
            auto [masksPath, _p, _a] = PipelinePaths(videoPath, "hed");
            fs::path segScript  = kProjectDir / "scripts" / "segment.py";
            fs::path vecScript  = kProjectDir / "scripts" / "vectorize.py";
            fs::path encScript  = kProjectDir / "scripts" / "encode.py";
            fs::path ckptDir    = kProjectDir / "models"  / "sam2";
            fs::path gdinoModel = kProjectDir / "models"  / "gdino" /
                                  "groundingdino_swint_ogc.pth";
            fs::path edterModel      = kProjectDir / "models" / "edter"          / "EDTER-BSDS-VOC-StageI.pth";
            fs::path diffEdgeModel   = kProjectDir / "models" / "diffusion_edge" / "bsds.pt";
            fs::path diffEdgeStage1  = kProjectDir / "models" / "diffusion_edge" / "first_stage_total_320.pt";
            fs::path diffEdgeConfig  = kProjectDir / "models" / "diffusion_edge" / "bsds_sample.yaml";

            // segment command
            std::string segCmd =
                kPython + " \"" + segScript.string()        + "\""
                " --video \""         + videoPath            + "\""
                " --output \""        + masksPath.string()   + "\""
                " --model "           + kSam2Model           +
                " --checkpoint-dir \"" + ckptDir.string()    + "\""
                " --device "          + kSam2Device          +
                " --gdino-model \""   + gdinoModel.string()  + "\"";
            if (!prompt.empty())
                segCmd += " --prompt \"" + prompt + "\"";

            // vectorize --method all writes one JSON per method automatically
            // We point --output at the base polylines path (method tag inserted by script)
            auto [_m, polyBase, _b] = PipelinePaths(videoPath, "all");
            // polyBase ends in "_all.json"; the script strips that and uses stem
            // Actually vectorize.py inserts the method suffix itself, so pass the base name
            std::string stem    = fs::path(videoPath).stem().string();
            std::string maskTag = stem + "_sam2-" + kSam2Model;
            fs::path polyDir    = kResourcesDir / "polylines";
            fs::path animDir    = kResourcesDir / "animations";
            fs::path polyAllOut = polyDir / (maskTag + ".json"); // script adds _method

            std::string vecCmd =
                kPython + " \"" + vecScript.string()                    + "\""
                " --video \""                    + videoPath             + "\""
                " --masks \""                    + masksPath.string()    + "\""
                " --output \""                   + polyAllOut.string()   + "\""
                " --method all"
                " --edter-model \""              + edterModel.string()   + "\""
                " --diffusion-edge-model \""     + diffEdgeModel.string()  + "\""
                " --diffusion-edge-first-stage-model \"" + diffEdgeStage1.string() + "\""
                " --diffusion-edge-config \""    + diffEdgeConfig.string() + "\""
                " --device "                     + kSam2Device;

            // encode commands — one per method
            static const std::vector<std::string> kAllMethods = {"canny","hed","edter","diffusion_edge"};
            std::vector<std::string> encCmds;
            std::vector<fs::path>    animPaths;
            for (const auto& m : kAllMethods) {
                fs::path polyPath = polyDir  / (maskTag + "_" + m + ".json");
                fs::path animPath = animDir  / (maskTag + "_" + m + ".ild");
                encCmds.push_back(
                    kPython + " \"" + encScript.string()  + "\""
                    " --polylines \"" + polyPath.string() + "\""
                    " --output \""    + animPath.string() + "\"");
                animPaths.push_back(animPath);
            }

            if (g_videoThread.joinable()) {
                g_videoProcessing = false;
                g_videoReady      = false;
                g_videoThread.join();
            }
            g_videoProcessing = true;
            g_videoThread = std::thread([=, &laser]() mutable {

                auto RunStage = [](const std::string& label, const std::string& cmd) -> bool {
                    std::cout << "\n" << label << "\n" << std::flush;
                    FILE* pipe = _popen(("\"" + cmd + " 2>&1\"").c_str(), "r");
                    if (!pipe) { std::cerr << label << " failed to launch\n"; return false; }
                    char buf[512];
                    while (fgets(buf, sizeof(buf), pipe))
                        std::cout << buf << std::flush;
                    int ret = _pclose(pipe);
                    if (ret != 0) std::cerr << label << " exited with code " << ret << "\n";
                    return ret == 0;
                };

                fs::create_directories(masksPath.parent_path());
                fs::create_directories(polyDir);
                fs::create_directories(animDir);

                // Stage 1 — segment (skip if masks already exist)
                if (fs::exists(masksPath)) {
                    std::cout << "\n[process] Masks cached — skipping segment.\n" << std::flush;
                } else {
                    if (!RunStage("[process:segment]", segCmd)) {
                        std::cerr << "[process] Aborting pipeline.\n> " << std::flush;
                        g_videoProcessing = false;
                        return;
                    }
                }

                // Stage 2 — vectorize all methods
                if (!RunStage("[process:vectorize]", vecCmd)) {
                    std::cerr << "[process] Vectorize failed — aborting.\n> " << std::flush;
                    g_videoProcessing = false;
                    return;
                }

                // Stage 3 — encode each method
                for (std::size_t i = 0; i < encCmds.size(); ++i) {
                    RunStage("[process:encode:" + kAllMethods[i] + "]", encCmds[i]);
                }

                std::cout << "\n[process] Done. ILDA files written to resources/animations/\n"
                          << "> " << std::flush;
                g_videoProcessing = false;
            });

            std::cout << "Processing: " << videoPath << "\n"
                      << "Runs segment → vectorize(all) → encode(all). This may take a while.\n"
                      << "Console remains live. Type 'stop' to abort.\n";
            PrintMenu(laser);
            continue;
        }

        // ── segment <video> [prompt] ──────────────────────────────────────────
        if (line.rfind("segment ", 0) == 0) {
            std::istringstream ss(line.substr(8));
            std::string videoPath, prompt;
            ss >> videoPath;
            videoPath = ResolveVideo(videoPath);
            std::getline(ss, prompt);
            if (!prompt.empty() && prompt.front() == ' ') prompt = prompt.substr(1);

            if (videoPath.empty()) {
                std::cerr << "Usage: segment <video> [prompt]\n";
                PrintMenu(laser); continue;
            }

            auto [masksPath, polyPath, animPath] = PipelinePaths(videoPath, "canny");

            fs::create_directories(masksPath.parent_path());
            fs::path segScript  = kProjectDir / "scripts" / "segment.py";
            fs::path ckptDir    = kProjectDir / "models"  / "sam2";
            fs::path gdinoModel = kProjectDir / "models"  / "gdino" /
                                  "groundingdino_swint_ogc.pth";

            std::string cmd =
                kPython + " \"" + segScript.string()   + "\""
                " --video \""         + videoPath          + "\""
                " --output \""        + masksPath.string() + "\""
                " --model "           + kSam2Model         +
                " --checkpoint-dir \"" + ckptDir.string()  + "\""
                " --device "          + kSam2Device        +
                " --gdino-model \""   + gdinoModel.string()+ "\"";
            if (!prompt.empty())
                cmd += " --prompt \"" + prompt + "\"";

            if (g_videoThread.joinable()) {
                g_videoProcessing = false;
                g_videoReady      = false;
                g_videoThread.join();
            }
            g_videoProcessing = true;
            g_videoThread = std::thread([cmd, masksPath, &laser]() mutable {
                std::cout << "\n[segment] Running ...\n" << std::flush;
                FILE* pipe = _popen(("\"" + cmd + " 2>&1\"").c_str(), "r");
                if (pipe) {
                    char buf[512];
                    while (fgets(buf, sizeof(buf), pipe))
                        std::cout << buf << std::flush;
                    int ret = _pclose(pipe);
                    if (ret == 0 && fs::exists(masksPath))
                        std::cout << "\n[segment] Done: " << masksPath.filename()
                                  << "  — run preview-masks or vectorize next\n> "
                                  << std::flush;
                    else
                        std::cerr << "\n[segment] Failed (exit " << ret << ")\n> "
                                  << std::flush;
                }
                g_videoProcessing = false;
            });
            std::cout << "Segmenting: " << videoPath << "\n";
            std::cout << "Console remains live. Type 'stop' to abort.\n";
            PrintMenu(laser);
            continue;
        }

        // ── preview-masks <video> ─────────────────────────────────────────────
        if (line.rfind("preview-masks ", 0) == 0) {
            std::string videoPath = ResolveVideo(line.substr(14));
            if (videoPath.empty()) {
                std::cerr << "Usage: preview-masks <video>\n";
                PrintMenu(laser); continue;
            }
            auto [masksPath, polyPath, animPath] = PipelinePaths(videoPath, "canny");
            if (!fs::exists(masksPath)) {
                std::cerr << "[preview] Masks not found: " << masksPath << "\n"
                             "  Run 'segment' first.\n";
                PrintMenu(laser); continue;
            }
            fs::path script = kProjectDir / "scripts" / "preview_masks.py";
            std::string cmd =
                kPython + " \"" + script.string()      + "\""
                " --masks \""   + masksPath.string() + "\""
                " --video \""   + videoPath           + "\"";
            std::cout << "[preview] Opening mask viewer — close the window to return.\n";
            RunSync(cmd);
            PrintMenu(laser);
            continue;
        }

        // ── vectorize <video> <method> ────────────────────────────────────────
        if (line.rfind("vectorize ", 0) == 0) {
            std::istringstream ss(line.substr(10));
            std::string videoPath, method;
            ss >> videoPath >> method;
            videoPath = ResolveVideo(videoPath);

            static const std::vector<std::string> kMethods = {"canny","hed","edter","diffusion_edge"};
            if (videoPath.empty() || method.empty() ||
                std::find(kMethods.begin(), kMethods.end(), method) == kMethods.end()) {
                std::cerr << "Usage: vectorize <video> <canny|hed|edter|diffusion_edge>\n";
                PrintMenu(laser); continue;
            }

            auto [masksPath, polyPath, animPath] = PipelinePaths(videoPath, method);
            if (!fs::exists(masksPath)) {
                std::cerr << "[vectorize] Masks not found: " << masksPath << "\n"
                             "  Run 'segment' first.\n";
                PrintMenu(laser); continue;
            }
            fs::create_directories(polyPath.parent_path());
            fs::path vecScript      = kProjectDir / "scripts" / "vectorize.py";
            fs::path edterModel     = kProjectDir / "models" / "edter"          / "EDTER-BSDS-VOC-StageI.pth";
            fs::path diffEdgeModel  = kProjectDir / "models" / "diffusion_edge" / "bsds.pt";
            fs::path diffEdgeStage1 = kProjectDir / "models" / "diffusion_edge" / "first_stage_total_320.pt";
            fs::path diffEdgeCfg    = kProjectDir / "models" / "diffusion_edge" / "bsds_sample.yaml";

            std::string cmd =
                kPython + " \"" + vecScript.string()                    + "\""
                " --video \""                    + videoPath             + "\""
                " --masks \""                    + masksPath.string()    + "\""
                " --output \""                   + polyPath.string()     + "\""
                " --method "                     + method                +
                " --edter-model \""              + edterModel.string()   + "\""
                " --diffusion-edge-model \""     + diffEdgeModel.string()  + "\""
                " --diffusion-edge-first-stage-model \"" + diffEdgeStage1.string() + "\""
                " --diffusion-edge-config \""    + diffEdgeCfg.string()    + "\""
                " --device "                     + kSam2Device;

            if (g_videoThread.joinable()) {
                g_videoProcessing = false;
                g_videoReady      = false;
                g_videoThread.join();
            }
            g_videoProcessing = true;
            g_videoThread = std::thread([cmd, polyPath]() mutable {
                std::cout << "\n[vectorize] Running ...\n" << std::flush;

                FILE* pipe = _popen(("\"" + cmd + " 2>&1\"").c_str(), "r");

                if (pipe) {
                    char buf[512];
                    while (fgets(buf, sizeof(buf), pipe))
                        std::cout << buf << std::flush;
                    int ret = _pclose(pipe);
                    if (ret == 0 && fs::exists(polyPath))
                        std::cout << "\n[vectorize] Done: " << polyPath.filename()
                                  << "  — run preview-polylines or encode next\n> "
                                  << std::flush;
                    else
                        std::cerr << "\n[vectorize] Failed (exit " << ret << ")\n> "
                                  << std::flush;
                }
                g_videoProcessing = false;
            });
            std::cout << "Vectorizing (" << method << "): " << videoPath << "\n";
            PrintMenu(laser);
            continue;
        }

        // ── preview-polylines <video> <method> ────────────────────────────────
        if (line.rfind("preview-polylines ", 0) == 0) {
            std::istringstream ss(line.substr(18));
            std::string videoPath, method;
            ss >> videoPath >> method;
            videoPath = ResolveVideo(videoPath);
            if (videoPath.empty() || method.empty()) {
                std::cerr << "Usage: preview-polylines <video> <method>\n";
                PrintMenu(laser); continue;
            }
            auto [masksPath, polyPath, animPath] = PipelinePaths(videoPath, method);
            if (!fs::exists(polyPath)) {
                std::cerr << "[preview] Polylines not found: " << polyPath << "\n"
                             "  Run 'vectorize' first.\n";
                PrintMenu(laser); continue;
            }
            fs::path script = kProjectDir / "scripts" / "preview_polylines.py";
            std::string cmd =
                kPython + " \"" + script.string()    + "\""
                " --polylines \"" + polyPath.string() + "\"";
            std::cout << "[preview] Opening polyline viewer — close the window to return.\n";
            RunSync(cmd);
            PrintMenu(laser);
            continue;
        }

        // ── encode <video> <method> ───────────────────────────────────────────
        if (line.rfind("encode ", 0) == 0) {
            std::istringstream ss(line.substr(7));
            std::string videoPath, method;
            ss >> videoPath >> method;
            videoPath = ResolveVideo(videoPath);

            static const std::vector<std::string> kMethods = {"canny","hed","edter","diffusion_edge"};
            if (videoPath.empty() || method.empty() ||
                std::find(kMethods.begin(), kMethods.end(), method) == kMethods.end()) {
                std::cerr << "Usage: encode <video> <canny|hed|edter|diffusion_edge>\n";
                PrintMenu(laser); continue;
            }

            auto [masksPath, polyPath, animPath] = PipelinePaths(videoPath, method);
            if (!fs::exists(polyPath)) {
                std::cerr << "[encode] Polylines not found: " << polyPath << "\n"
                             "  Run 'vectorize' first.\n";
                PrintMenu(laser); continue;
            }
            fs::create_directories(animPath.parent_path());
            fs::path encScript = kProjectDir / "scripts" / "encode.py";

            std::string cmd =
                kPython + " \"" + encScript.string()  + "\""
                " --polylines \"" + polyPath.string() + "\""
                " --output \""    + animPath.string()  + "\"";

            if (g_videoThread.joinable()) {
                g_videoProcessing = false;
                g_videoReady      = false;
                g_videoThread.join();
            }
            g_videoProcessing = true;
            g_videoThread = std::thread([cmd, animPath, &laser]() mutable {
                std::cout << "\n[encode] Running ...\n" << std::flush;
                FILE* pipe = _popen(("\"" + cmd + " 2>&1\"").c_str(), "r");
                if (pipe) {
                    char buf[512];
                    while (fgets(buf, sizeof(buf), pipe))
                        std::cout << buf << std::flush;
                    _pclose(pipe);
                }
                // encode.py always writes baked files now
                auto anim = ILDAFile::LoadFlat(animPath.string());
                if (!anim.empty()) {
                    {
                        std::lock_guard<std::mutex> lock(g_animationMutex);
                        g_videoAnimationPhysical = std::move(anim);
                        g_videoAnimation.clear();
                        g_videoBaked             = true;
                    }
                    g_nowPlaying      = animPath.filename().string();
                    g_videoReady      = true;
                    g_sceneIndex      = 7;
                    std::cout << "\n[encode] Playing: " << animPath.filename()
                              << "\n> " << std::flush;
                } else {
                    std::cerr << "\n[encode] Failed to load: " << animPath << "\n> "
                              << std::flush;
                }
                g_videoProcessing = false;
            });
            std::cout << "Encoding (" << method << "): " << videoPath << "\n";
            PrintMenu(laser);
            continue;
        }

        // Parameter tuning
        if (line.rfind("set ", 0) == 0) {
            std::istringstream ss(line.substr(4));
            std::string param; float value;
            if (!(ss >> param >> value)) {
                std::cout << "Usage: set <param> <value>\n";
                PrintMenu(laser); continue;
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
            PrintMenu(laser); continue;
        }

        if (line.rfind("scale ", 0) == 0) {
            try {
                float val = std::stof(line.substr(6));
                if (val >= -1.0f && val <= 2.0f) {
                    g_animationScale = val;
                    std::cout << "Set animation scale to " << g_animationScale << "\n";
                }
                else {
                    std::cout << "Scale must be between -1 and 2.\n";
                }
            }
            catch (...) {
                std::cout << "Usage: scale [number from -1 to 2]\n";
            }
            PrintMenu(laser);
            continue;
        }

        if (line.rfind("brightness ", 0) == 0) {
            try {
                float val = std::stof(line.substr(6));
                if (val >= 0.0f && val <= 1.0f) {
                    g_animationBrightness = val;
                    std::cout << "Set animation brightness to " << g_animationBrightness << "\n";
                }
                else {
                    std::cout << "Scale must be between 0 and 1.\n";
                }
            }
            catch (...) {
                std::cout << "Usage: scale [number from 0 to 1]\n";
            }
            PrintMenu(laser);
            continue;
        }

        std::cout << "Unknown command.\n";
        PrintMenu(laser);
    }

    g_videoProcessing = false;
    if (g_videoThread.joinable()) g_videoThread.join();
    g_running = false;
    renderThread.join();
    laser.Close();
    return 0;
}
