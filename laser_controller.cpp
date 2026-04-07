#define NOMINMAX
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <cmath>
#include <sstream>
#include <iomanip>
#include "HeliosOutput.h"

// -----------------------------------------------------------------------------
// Shape generators
// -----------------------------------------------------------------------------

std::vector<LaserPoint> MakeCircle(float cx, float cy, float radius, int numPoints,
    float r, float g, float b)
{
    std::vector<LaserPoint> points;
    for (int i = 0; i <= numPoints; ++i) {
        float angle = 2.0f * 3.14159265f * i / numPoints;
        points.push_back({
            cx + radius * std::cos(angle),
            cy + radius * std::sin(angle),
            r, g, b
            });
    }
    return points;
}

std::vector<LaserPoint> MakeRectangle(float cx, float cy, float w, float h,
    float r, float g, float b)
{
    float l = cx - w * 0.5f;
    float ri = cx + w * 0.5f;
    float t = cy - h * 0.5f;
    float bo = cy + h * 0.5f;
    return {
        {l,  t,  r, g, b}, {ri, t,  r, g, b},
        {ri, bo, r, g, b}, {l,  bo, r, g, b},
        {l,  t,  r, g, b}
    };
}

std::vector<LaserPoint> MakeTriangle(float cx, float cy, float size,
    float r, float g, float b)
{
    return {
        {cx,                cy - size,        r, g, b},
        {cx + size * 0.87f, cy + size * 0.5f, r, g, b},
        {cx - size * 0.87f, cy + size * 0.5f, r, g, b},
        {cx,                cy - size,        r, g, b}
    };
}

std::vector<LaserPoint> MakeStar(float cx, float cy, float outerR, float innerR,
    float r, float g, float b)
{
    std::vector<LaserPoint> points;
    for (int i = 0; i < 10; ++i) {
        float angle = 3.14159265f * i / 5.0f - 3.14159265f / 2.0f;
        float radius = (i % 2 == 0) ? outerR : innerR;
        points.push_back({ cx + radius * std::cos(angle),
                           cy + radius * std::sin(angle), r, g, b });
    }
    points.push_back(points.front());
    return points;
}

// -----------------------------------------------------------------------------
// Scenes
// -----------------------------------------------------------------------------

using Frame = std::vector<std::vector<LaserPoint>>;

Frame Scene_SingleCircle() { return { MakeCircle(0,0, 0.5f, 300, 1,0,1) }; }
Frame Scene_TwoCircles() {
    return { MakeCircle(-0.4f,0,0.3f,200,1,0,0),
            MakeCircle(0.4f,0,0.3f,200,0,0,1) };
}
Frame Scene_ThreeShapes() {
    return { MakeCircle(0,0.6f,0.2f,150,1,0,0),
            MakeRectangle(-0.6f,-0.5f,0.3f,0.2f,0,1,0),
            MakeTriangle(0.6f,-0.5f,0.2f,0,0,1) };
}
Frame Scene_Star() { return { MakeStar(0,0,0.5f,0.2f,1,1,0) }; }
Frame Scene_FourCorners() {
    return { MakeCircle(-0.6f, 0.6f,0.15f,100,1,0,0),
            MakeCircle(0.6f, 0.6f,0.15f,100,0,1,0),
            MakeCircle(0.6f,-0.6f,0.15f,100,0,0,1),
            MakeCircle(-0.6f,-0.6f,0.15f,100,1,1,0) };
}
Frame Scene_RectAndCircle() {
    return { MakeRectangle(0,0,0.8f,0.8f,0,1,1),
            MakeCircle(0,0,0.3f,200,1,0,0.5f) };
}

// -----------------------------------------------------------------------------
// Shared state
// -----------------------------------------------------------------------------

std::atomic<int>  g_sceneIndex{ 1 };
std::atomic<bool> g_running{ true };
std::atomic<bool> g_configDirty{ false };

HeliosConfig g_config;
std::mutex   g_configMutex;

// -----------------------------------------------------------------------------
// Render thread
// -----------------------------------------------------------------------------

void RenderThread(HeliosOutput& laser)
{
    int   lastScene = -1;
    Frame currentFrame;

    while (g_running)
    {
        // Apply config update if pending
        if (g_configDirty.exchange(false)) {
            std::lock_guard<std::mutex> lock(g_configMutex);
            laser.SetConfig(g_config);
        }

        int scene = g_sceneIndex.load();
        if (scene != lastScene) {
            switch (scene) {
            case 1: currentFrame = Scene_SingleCircle();  break;
            case 2: currentFrame = Scene_TwoCircles();    break;
            case 3: currentFrame = Scene_ThreeShapes();   break;
            case 4: currentFrame = Scene_Star();          break;
            case 5: currentFrame = Scene_FourCorners();   break;
            case 6: currentFrame = Scene_RectAndCircle(); break;
            default: currentFrame = Scene_SingleCircle(); break;
            }
            lastScene = scene;
        }

        if (!currentFrame.empty())
            laser.SendFrame(currentFrame);

        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

// -----------------------------------------------------------------------------
// Print helpers
// -----------------------------------------------------------------------------

void PrintConfig(const HeliosConfig& c)
{
    std::cout << "\n--- Current Config ---\n";
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "  target_fps        = " << c.target_fps << "\n";
    std::cout << "  max_pps           = " << c.max_pps << "\n";
    std::cout << "  blank_points      = " << c.blank_points << "\n";
    std::cout << "  pre_on_points     = " << c.pre_on_points << "\n";
    std::cout << "  post_on_points    = " << c.post_on_points << "\n";
    std::cout << "  min_corner_dwell  = " << c.min_vertex_hold << "\n";
    std::cout << "  max_corner_dwell  = " << c.max_vertex_hold << "\n";
    std::cout << "  curve_threshold   = " << c.curve_threshold << "\n";
    std::cout << "  enable_reorder    = " << (c.enable_reorder ? "yes" : "no") << "\n";
    std::cout << "----------------------\n";
}

void PrintMenu()
{
    std::cout << "\n--- Laser Console ---\n";
    std::cout << "Scenes:  1-6\n";
    std::cout << "Tune:    set <param> <value>\n";
    std::cout << "         e.g.  set step_size 25\n";
    std::cout << "         e.g.  set max_corner_dwell 20\n";
    std::cout << "         e.g.  set curve_threshold 30\n";
    std::cout << "         e.g.  set blank_points 15\n";
    std::cout << "         e.g.  set target_fps 30\n";
    std::cout << "         e.g.  set reorder 1\n";
    std::cout << "Config:  config\n";
    std::cout << "Quit:    q\n";
    std::cout << "> ";
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main()
{
    // Default config
    g_config.target_fps = 60;
    g_config.max_pps = 30000;
    g_config.blank_points = 20;
    g_config.pre_on_points = 8;
    g_config.post_on_points = 8;
    g_config.min_vertex_hold = 0;
    g_config.max_vertex_hold = 20;
    g_config.curve_threshold = 20.0f;
    g_config.enable_reorder = false;

    HeliosOutput laser;
    if (!laser.Initialize(g_config)) {
        std::cerr << "Failed to initialize laser\n";
        return 1;
    }

    g_sceneIndex = 1;
    std::thread renderThread(RenderThread, std::ref(laser));

    PrintConfig(g_config);
    PrintMenu();

    std::string line;
    while (std::getline(std::cin, line))
    {
        if (line.empty()) { PrintMenu(); continue; }

        if (line == "q") {
            g_running = false;
            break;
        }

        if (line == "config") {
            std::lock_guard<std::mutex> lock(g_configMutex);
            PrintConfig(g_config);
            PrintMenu();
            continue;
        }

        // Scene switch — single digit
        if (line.size() == 1 && line[0] >= '1' && line[0] <= '6') {
            g_sceneIndex = line[0] - '0';
            std::cout << "Scene " << g_sceneIndex << "\n";
            PrintMenu();
            continue;
        }

        // Parameter set — "set <param> <value>"
        if (line.rfind("set ", 0) == 0) {
            std::istringstream ss(line.substr(4));
            std::string param;
            float value;
            if (!(ss >> param >> value)) {
                std::cout << "Usage: set <param> <value>\n";
                PrintMenu();
                continue;
            }

            {
                std::lock_guard<std::mutex> lock(g_configMutex);
                bool found = true;
                if (param == "target_fps")       g_config.target_fps = (int)value;
                else if (param == "max_pps")           g_config.max_pps = (int)value;
                else if (param == "blank_points")      g_config.blank_points = (int)value;
                else if (param == "pre_on_points")     g_config.pre_on_points = (int)value;
                else if (param == "post_on_points")    g_config.post_on_points = (int)value;
                else if (param == "min_vertex_hold")  g_config.min_vertex_hold = (int)value;
                else if (param == "max_vertex_hold")  g_config.max_vertex_hold = (int)value;
                else if (param == "curve_threshold")   g_config.curve_threshold = value;
                else if (param == "reorder")           g_config.enable_reorder = (value > 0.5f);
                else { std::cout << "Unknown param: " << param << "\n"; found = false; }

                if (found) {
                    std::cout << "Set " << param << " = " << value << "\n";
                    g_configDirty = true;
                }
            }
            PrintMenu();
            continue;
        }

        std::cout << "Unknown command\n";
        PrintMenu();
    }

    renderThread.join();
    laser.Close();
    return 0;
}