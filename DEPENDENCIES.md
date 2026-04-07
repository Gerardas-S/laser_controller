# Dependencies

## Required Libraries

### ONNX Runtime
- Version: 1.17.0 or later
- Download: https://github.com/microsoft/onnxruntime/releases
- Extract to: `external/onnxruntime/`
- Needed files:
  - `onnxruntime.dll` (Windows)
  - `onnxruntime_providers_cuda.dll` (for GPU)
  - Include: `onnxruntime_cxx_api.h`

### PortAudio
- Version: v19.7.0 or later
- Download: http://www.portaudio.com/download.html
- Extract to: `external/portaudio/`
- Needed files:
  - `portaudio_x64.dll` (Windows)
  - Include: `portaudio.h`

### Helios DAC SDK
- Download: https://github.com/Grix/helios_dac
- Extract to: `external/helios/`
- Needed files:
  - `HeliosLib.h`
  - `HeliosDacAPI.dll`

## Installation

1. Create `external/` directory in project root
2. Download libraries from links above
3. Extract to respective subdirectories
4. Build project