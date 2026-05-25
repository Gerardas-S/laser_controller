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
    // N=100 places the firmware-loop refresh at ~300 Hz on a 30 kPPS DAC
    // (well above flicker fusion) with sub-pixel chord error against the
    // true circle.  Pushing N higher only slows refresh; pushing N below
    // ~80 reveals polygonal chord error.
    //
    // We emit EXACTLY N points (not N+1).  The firmware-loop transition
    // from pt[N-1] to pt[0] is a chord of length R·2·sin(π/N) — identical
    // to every other chord — drawn with the laser already on.  A duplicate
    // closing point would double-illuminate pt[0] every loop and re-create
    // the bright-dot artifact this scene exists to avoid.  BuildFrame's
    // closed-polyline short-circuit (SEAM_TOL check) treats this as a
    // looping path because Distance(pt[0], pt[N-1]) < SEAM_TOL.
    const int   N      = 100;
    const float R      = 0.35f;
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

// Loaded ILDA animation — one HeliosPoint sequence per frame, in spec point
// order.  All loaded .ild files land here; runtime sends them straight
// through SendPhysical.  Per the ILDA spec a file IS the point sequence the
// DAC plays, so there is no "logical polyline" intermediate form — the
// runtime never re-bakes a loaded file through BuildFrame.
std::vector<std::vector<HeliosPoint>> g_videoAnimation;

std::mutex          g_animationMutex;
std::atomic<bool>   g_videoProcessing{ false };
std::atomic<bool>   g_videoReady{ false };
std::string         g_nowPlaying = "alignment circle";
float               g_animationScale = 1.0f;
float               g_animationBrightness = 1.0f;
// Playback speed multiplier — scales the software frame-advance rate.
//   1.0 = normal (advance rate = target_fps)
//   0.5 = half speed (frames held twice as long)
//   2.0 = double speed (frames advanced twice as fast)
// Multiplicative with target_fps:  effective animation FPS = target_fps × speed.
// Does NOT affect DAC point rate — only how often animFrame increments.
// The DAC always plays at max_pps; the firmware auto-loops the buffered
// frame between RenderThread upload ticks.
float               g_animationSpeed = 1.0f;

std::thread g_videoThread;

// The SAM2 pipeline runs as anonymous lambdas attached to g_videoThread
// inside each command handler (process / segment / vectorize / encode).

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
            // frameDt = base interval (1/target_fps) divided by speed multiplier.
            // std::max guards against zero/negative speed → infinite frameDt.
            float  spd     = std::max(0.01f, g_animationSpeed);
            double frameDt = 1.0 / (g_config.target_fps * spd);

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
            if (n == 0) {
                laser.SendFrame(alignCircle);
            } else {
                const float scale  = g_animationScale;
                const float bright = g_animationBrightness;
                const bool  modify = (scale != 1.0f) || (bright != 1.0f);
                const auto& frame  = g_videoAnimation[animFrame % n];

                if (modify) {
                    std::vector<HeliosPoint> out = frame;
                    for (auto& p : out) {
                        if (scale != 1.0f) {
                            // Scale around the centre of the 12-bit ILDA space.
                            const float cx = 2048.0f, cy = 2048.0f;
                            p.x = static_cast<uint16_t>(std::clamp(cx + (p.x - cx) * scale, 0.0f, 4095.0f));
                            p.y = static_cast<uint16_t>(std::clamp(cy + (p.y - cy) * scale, 0.0f, 4095.0f));
                        }
                        if (bright != 1.0f) {
                            p.r = static_cast<uint8_t>(std::clamp(p.r * bright, 0.0f, 255.0f));
                            p.g = static_cast<uint8_t>(std::clamp(p.g * bright, 0.0f, 255.0f));
                            p.b = static_cast<uint8_t>(std::clamp(p.b * bright, 0.0f, 255.0f));
                            p.i = std::max(p.r, std::max(p.g, p.b));
                        }
                    }
                    laser.SendPhysical(out);
                } else {
                    laser.SendPhysical(frame);
                }
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
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n--- Animation playback (applies to loaded .ild files) ---\n";
    std::cout << "  target_fps       = " << c.target_fps           << "  (host upload cadence)\n";
    std::cout << "  speed            = " << g_animationSpeed       << "  (multiplier; effective fps = target_fps * speed)\n";
    std::cout << "  scale            = " << g_animationScale       << "\n";
    std::cout << "  brightness       = " << g_animationBrightness  << "\n";
    std::cout << "\n--- Live-scene tuning (alignment circle / SendFrame; no effect on .ild playback) ---\n";
    std::cout << "  max_pps          = " << c.max_pps          << "\n";
    std::cout << "  blank_points     = " << c.blank_points     << "\n";
    std::cout << "  pre_on_points    = " << c.pre_on_points    << "\n";
    std::cout << "  post_on_points   = " << c.post_on_points   << "\n";
    std::cout << "  min_vertex_hold  = " << c.min_vertex_hold  << "\n";
    std::cout << "  max_vertex_hold  = " << c.max_vertex_hold  << "\n";
    std::cout << "  curve_threshold  = " << c.curve_threshold  << "\n";
    std::cout << "  reorder          = " << (c.enable_reorder ? "1" : "0") << "\n";
    std::cout << "-------------------------------------------------\n";
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
    std::cout << "  --- 4-stage pipeline ---\n";
    std::cout << "  segment  <video>                  Stage 1: video -> masks (blind 5-point grid)\n";
    std::cout << "  segment  <video> <prompt>         Stage 1: video -> masks via Grounding DINO text query\n";
    std::cout << "  segment  <video> --interactive    Stage 1: video -> masks via click prompting on frame 0\n";
    std::cout << "                                      L-click=subject  R-click=background  U=undo  Enter=confirm\n";
    std::cout << "  preview-masks <video>             View segmentation masks\n";
    std::cout << "\n";
    std::cout << "  vectorize <video> <stage0> <edge> <thin> <vec>\n";
    std::cout << "                                    Stage 2: masks -> polylines (4-stage pipeline)\n";
    std::cout << "                                      stage0: none | cb         (cb = CLAHE + bilateral)\n";
    std::cout << "                                      edge:   canny | hed | nbed | de    (de = DiffusionEdge)\n";
    std::cout << "                                      thin:   none | nms | zs | steger   (zs = Zhang-Suen)\n";
    std::cout << "                                      vec:    dp | ge | iarussi          (ge = greedy Eulerian)\n";
    std::cout << "  preview-polylines <video> <stage0> <edge> <thin> <vec>  View vectorized paths\n";
    std::cout << "\n";
    std::cout << "  encode   <video> <stage0> <edge> <thin> <vec>          Stage 3: polylines -> ILDA + play\n";
    std::cout << "\n";
    std::cout << "  (Batch matrix sweeps: run `python scripts/experiment_runner.py` outside the C++ app.)\n\n";

    std::cout << "  --- Parameters (set <param> <val>) ---\n";
    std::cout << "    Animation playback (effective immediately on loaded .ild files):\n";
    std::cout << "      target_fps     1..240        host upload cadence (Hz);\n"
                 "                                   DAC firmware loops the buffered frame between uploads\n";
    std::cout << "      speed          0.05..10     speed multiplier; effective fps = target_fps * speed\n";
    std::cout << "      scale         -1..2          zoom around centre (negative = mirror)\n";
    std::cout << "      brightness     0..1          colour multiplier\n";
    std::cout << "\n";
    std::cout << "    Live-scene tuning (alignment circle / SendFrame; NO effect on .ild playback):\n";
    std::cout << "      max_pps         1000..60000   DAC point rate\n";
    std::cout << "      blank_points    0..200        blank dwell at stroke boundaries\n";
    std::cout << "      pre_on_points   0..100        illuminated dwell entering stroke\n";
    std::cout << "      post_on_points  0..100        illuminated dwell exiting stroke\n";
    std::cout << "      min_vertex_hold 0..100        min corner dwell\n";
    std::cout << "      max_vertex_hold 0..200        max corner dwell\n";
    std::cout << "      curve_threshold 0..180        bend angle threshold (deg)\n";
    std::cout << "      reorder         0|1          enable nearest-neighbour reorder\n";
    std::cout << "\n";
    std::cout << "  stop               back to circle\n";
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
              g_videoAnimation.clear(); }
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

            // Per ILDA spec section 2.4, the file IS the point sequence the
            // DAC will play.  We never re-bake it through BuildFrame.
            auto anim = ILDAFile::Load(path);
            if (anim.empty()) {
                std::cerr << "Failed to load: " << path << "\n";
                PrintMenu(laser); continue;
            }
            std::lock_guard<std::mutex> lock(g_animationMutex);
            g_videoAnimation = std::move(anim);

            g_nowPlaying = files[idx - 1].filename().string();
            g_videoReady = true;
            g_sceneIndex = 7;
            PrintMenu(laser);
            continue;
        }

        // ── SAM2 pipeline helpers ─────────────────────────────────────────────
        // Shared path computation: given a video path, derive all stage paths.
        // model tag is hardcoded here — change kSam2Model to upgrade.
        static const std::string kSam2Model  = "large";
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
            std::string stem   = fs::path(videoPath).stem().string();
            fs::path masksPath = kResourcesDir / "masks"      / (stem + ".npz");
            fs::path polyPath  = kResourcesDir / "polylines"  / (stem + "_" + method + ".json");
            fs::path animPath  = kResourcesDir / "animations" / (stem + "_" + method + ".ild");
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

        // ──────────────────────────────────────────────────────────────────────
        // 4-stage pipeline helpers (shared by vectorize / preview-polylines /
        // encode handlers).  Validate stage-method names and construct the
        // pipeline-tagged JSON / ILD paths so distinct (stage0, edge, thin,
        // vec) tuples never clobber each other's outputs.
        // ──────────────────────────────────────────────────────────────────────
        static const std::vector<std::string> kStage0Methods = {"none","cb"};
        static const std::vector<std::string> kEdgeMethods   = {"canny","hed","nbed","de"};
        static const std::vector<std::string> kThinMethods   = {"none","nms","zs","steger"};
        static const std::vector<std::string> kVecMethods    = {"dp","ge","iarussi"};

        auto ValidStage = [](const std::vector<std::string>& valid,
                              const std::string& v) -> bool {
            return std::find(valid.begin(), valid.end(), v) != valid.end();
        };

        // Build per-pipeline JSON / ILD paths from the four stage tags.
        // Mask path includes the SAM2 model tag (different models → different masks).
        // Polyline / animation paths are keyed only by video stem + pipeline stages —
        // the viewer doesn't care which SAM2 model produced the mask.
        // Example: ballet_cb_hed_steger_ge.{json|ild}
        auto BuildPipelinePaths = [&](const std::string& videoPath,
                                       const std::string& stage0,
                                       const std::string& edge,
                                       const std::string& thin,
                                       const std::string& vec)
            -> std::tuple<fs::path, fs::path, fs::path> {
            std::string stem    = fs::path(videoPath).stem().string();
            std::string pipeTag = stage0 + "_" + edge + "_" + thin + "_" + vec;
            fs::path masksPath = kResourcesDir / "masks"      / (stem + ".npz");
            fs::path polyPath  = kResourcesDir / "polylines"  / (stem + "_" + pipeTag + ".json");
            fs::path animPath  = kResourcesDir / "animations" / (stem + "_" + pipeTag + ".ild");
            return {masksPath, polyPath, animPath};
        };

        // `process` command was removed in the 4-stage pipeline refactor.
        // Single-pipeline runs go through `segment` + `vectorize` + `encode`
        // explicitly; batch matrix sweeps run via
        //   python scripts/experiment_runner.py
        // outside the C++ app.

        // ── segment <video> [prompt] ──────────────────────────────────────────
        if (line.rfind("segment ", 0) == 0) {
            std::istringstream ss(line.substr(8));
            std::string videoPath, prompt;
            ss >> videoPath;
            videoPath = ResolveVideo(videoPath);
            std::getline(ss, prompt);
            if (!prompt.empty() && prompt.front() == ' ') prompt = prompt.substr(1);

            if (videoPath.empty()) {
                std::cerr << "Usage: segment <video> [--frames N,N,...] [prompt | --interactive]\n";
                PrintMenu(laser); continue;
            }

            // Extract optional --frames N,N,... from the prompt string
            // Value is a comma-separated list with no spaces, e.g. --frames 0,56,190
            std::string framesArg;
            {
                auto pos = prompt.find("--frames ");
                if (pos != std::string::npos) {
                    std::istringstream fs(prompt.substr(pos + 9));
                    fs >> framesArg;   // reads one token (no spaces in comma-separated list)
                    auto end = prompt.find(' ', pos + 9);
                    prompt.erase(pos, (end == std::string::npos ? prompt.size() : end) - pos);
                    while (prompt.find("  ") != std::string::npos)
                        prompt.replace(prompt.find("  "), 2, " ");
                    if (!prompt.empty() && prompt.front() == ' ') prompt = prompt.substr(1);
                    if (!prompt.empty() && prompt.back()  == ' ') prompt.pop_back();
                }
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
            if (!framesArg.empty())
                cmd += " --frames " + framesArg;
            if (prompt == "--interactive")
                cmd += " --interactive";
            else if (!prompt.empty())
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

        // ── vectorize <video> <stage0> <edge> <thin> <vec> ────────────────────
        if (line.rfind("vectorize ", 0) == 0) {
            std::istringstream ss(line.substr(10));
            std::string videoPath, stage0, edge, thin, vec;
            ss >> videoPath >> stage0 >> edge >> thin >> vec;
            videoPath = ResolveVideo(videoPath);

            if (videoPath.empty() ||
                !ValidStage(kStage0Methods, stage0) ||
                !ValidStage(kEdgeMethods,   edge)   ||
                !ValidStage(kThinMethods,   thin)   ||
                !ValidStage(kVecMethods,    vec)) {
                std::cerr << "Usage: vectorize <video> <stage0> <edge> <thin> <vec>\n"
                          << "  stage0: none | cb         (cb = CLAHE + bilateral)\n"
                          << "  edge:   canny | hed | nbed | de    (de = DiffusionEdge)\n"
                          << "  thin:   none | nms | zs | steger   (zs = Zhang-Suen)\n"
                          << "  vec:    dp | ge | iarussi          (ge = greedy Eulerian)\n";
                PrintMenu(laser); continue;
            }

            auto [masksPath, polyPath, animPath] =
                BuildPipelinePaths(videoPath, stage0, edge, thin, vec);
            if (!fs::exists(masksPath)) {
                std::cerr << "[vectorize] Masks not found: " << masksPath << "\n"
                             "  Run 'segment' first.\n";
                PrintMenu(laser); continue;
            }
            fs::create_directories(polyPath.parent_path());
            fs::path vecScript = kProjectDir / "scripts" / "vectorize.py";

            // 4-stage pipeline: model paths are now hard-coded in
            // pipeline/defaults.py.  CLI only carries stage selection.
            std::string cmd =
                kPython + " \"" + vecScript.string()  + "\""
                " --video \""   + videoPath           + "\""
                " --masks \""   + masksPath.string()  + "\""
                " --output \""  + polyPath.string()   + "\""
                " --stage0 "    + stage0              +
                " --edge "      + edge                +
                " --thin "      + thin                +
                " --vec "       + vec                 +
                " --device "    + kSam2Device;

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
            std::cout << "Vectorizing (" << stage0 << " / " << edge << " / "
                      << thin << " / " << vec << "): " << videoPath << "\n";
            PrintMenu(laser);
            continue;
        }

        // ── preview-polylines <video> <stage0> <edge> <thin> <vec> ───────────
        if (line.rfind("preview-polylines ", 0) == 0) {
            std::istringstream ss(line.substr(18));
            std::string videoPath, stage0, edge, thin, vec;
            ss >> videoPath >> stage0 >> edge >> thin >> vec;
            videoPath = ResolveVideo(videoPath);
            if (videoPath.empty() ||
                !ValidStage(kStage0Methods, stage0) ||
                !ValidStage(kEdgeMethods,   edge)   ||
                !ValidStage(kThinMethods,   thin)   ||
                !ValidStage(kVecMethods,    vec)) {
                std::cerr << "Usage: preview-polylines <video> <stage0> <edge> <thin> <vec>\n";
                PrintMenu(laser); continue;
            }
            auto [masksPath, polyPath, animPath] =
                BuildPipelinePaths(videoPath, stage0, edge, thin, vec);
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

        // ── encode <video> <stage0> <edge> <thin> <vec> ───────────────────────
        if (line.rfind("encode ", 0) == 0) {
            std::istringstream ss(line.substr(7));
            std::string videoPath, stage0, edge, thin, vec;
            ss >> videoPath >> stage0 >> edge >> thin >> vec;
            videoPath = ResolveVideo(videoPath);

            if (videoPath.empty() ||
                !ValidStage(kStage0Methods, stage0) ||
                !ValidStage(kEdgeMethods,   edge)   ||
                !ValidStage(kThinMethods,   thin)   ||
                !ValidStage(kVecMethods,    vec)) {
                std::cerr << "Usage: encode <video> <stage0> <edge> <thin> <vec>\n"
                          << "  stage0: none | cb         (cb = CLAHE + bilateral)\n"
                          << "  edge:   canny | hed | nbed | de    (de = DiffusionEdge)\n"
                          << "  thin:   none | nms | zs | steger   (zs = Zhang-Suen)\n"
                          << "  vec:    dp | ge | iarussi          (ge = greedy Eulerian)\n";
                PrintMenu(laser); continue;
            }

            auto [masksPath, polyPath, animPath] =
                BuildPipelinePaths(videoPath, stage0, edge, thin, vec);
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
                auto anim = ILDAFile::Load(animPath.string());
                if (!anim.empty()) {
                    {
                        std::lock_guard<std::mutex> lock(g_animationMutex);
                        g_videoAnimation = std::move(anim);
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
            std::cout << "Encoding (" << stage0 << " / " << edge << " / "
                      << thin << " / " << vec << "): " << videoPath << "\n";
            PrintMenu(laser);
            continue;
        }

        // Parameter tuning — single dispatcher for every settable knob.
        //
        // Two semantic groups, both routed through `set <param> <value>`:
        //
        //   A) Real-time playback (applies immediately to baked-ILDA frames):
        //        target_fps         — RenderThread's frame-advance rate (g_config)
        //        scale / brightness / speed — animation-layer globals
        //
        //   B) Alignment-circle tuning (BuildFrame path only — NO effect on
        //      baked .ild playback, which goes through SendPhysical and skips
        //      BuildFrame entirely):
        //        max_pps, blank_points, pre/post_on_points,
        //        min/max_vertex_hold, curve_threshold, reorder
        //
        // Mutex policy: g_configMutex is held for the whole dispatch.  g_config
        // writes need it (RenderThread reads g_config under the same lock when
        // pushing SetConfig to HeliosOutput).  Animation-globals writes don't
        // strictly need it (plain floats), but taking it unconditionally keeps
        // the dispatcher uniform and avoids per-group branching.
        if (line.rfind("set ", 0) == 0) {
            std::istringstream ss(line.substr(4));
            std::string p; float v;
            if (!(ss >> p >> v)) {
                std::cout << "Usage: set <param> <value>\n";
                PrintMenu(laser); continue;
            }

            std::lock_guard<std::mutex> lock(g_configMutex);

            // Generic range-check helpers.  Return true if value was accepted
            // (in range) and assigned to `field`.  On out-of-range, print a
            // diagnostic and leave `field` untouched.
            auto setInt = [&](float lo, float hi, int& field) -> bool {
                if (v >= lo && v <= hi) { field = (int)v; return true; }
                std::cout << "Range for " << p << ": [" << (int)lo
                          << ", " << (int)hi << "]\n";
                return false;
            };
            auto setFloat = [&](float lo, float hi, float& field) -> bool {
                if (v >= lo && v <= hi) { field = v; return true; }
                std::cout << "Range for " << p << ": [" << lo
                          << ", " << hi << "]\n";
                return false;
            };

            bool ok       = false;
            bool isConfig = false;   // true → set g_configDirty to push SetConfig

            // ── Group A: real-time playback ────────────────────────────────────
            if      (p == "target_fps")      { ok = setInt  (1,    240,   g_config.target_fps);      isConfig = ok; }
            else if (p == "scale")           { ok = setFloat(-1.0f, 2.0f, g_animationScale); }
            else if (p == "brightness")      { ok = setFloat( 0.0f, 1.0f, g_animationBrightness); }
            else if (p == "speed")           { ok = setFloat( 0.05f, 10.0f, g_animationSpeed); }
            // ── Group B: alignment-circle (BuildFrame) tuning ──────────────────
            else if (p == "max_pps")         { ok = setInt  (1000, 60000, g_config.max_pps);         isConfig = ok; }
            else if (p == "blank_points")    { ok = setInt  (0,    200,   g_config.blank_points);    isConfig = ok; }
            else if (p == "pre_on_points")   { ok = setInt  (0,    100,   g_config.pre_on_points);   isConfig = ok; }
            else if (p == "post_on_points")  { ok = setInt  (0,    100,   g_config.post_on_points);  isConfig = ok; }
            else if (p == "min_vertex_hold") { ok = setInt  (0,    100,   g_config.min_vertex_hold); isConfig = ok; }
            else if (p == "max_vertex_hold") { ok = setInt  (0,    200,   g_config.max_vertex_hold); isConfig = ok; }
            else if (p == "curve_threshold") { ok = setFloat(0.0f, 180.0f, g_config.curve_threshold); isConfig = ok; }
            else if (p == "reorder")         { g_config.enable_reorder = (v > 0.5f); ok = true; isConfig = true; }
            else                             { std::cout << "Unknown param: " << p << "\n"; }

            if (ok) {
                if (isConfig) g_configDirty = true;
                std::cout << "Set " << p << " = " << v << "\n";
            }
            PrintMenu(laser); continue;
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
