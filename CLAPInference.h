#pragma once

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include <sstream>

// Forward declaration
class AudioPreprocessor;

class CLAPInference {
public:
    enum class Backend {
        CPU,
        GPU
    };

    CLAPInference();
    ~CLAPInference();

    bool Initialize(Backend backend);
    bool LoadModel(const std::wstring& modelPath);

    // Audio embedding
    std::vector<float> GetEmbedding(const float* audioBuffer, size_t bufferSize, int sampleRate);
    std::vector<float> GetEmbeddingFromMelSpec(const float* mel_spec, size_t num_frames, size_t num_mels);

    // Text embedding
    std::vector<float> GetTextEmbedding(const std::string& text);

    // Book processing
    bool LoadBookFromFile(const std::string& bookPath, size_t chunkSize = 300);
    std::string FindBestMatchingChunk(const std::vector<float>& audioEmbedding);

    size_t GetEmbeddingDim() const;
    bool IsLoaded() const;
    const char* GetBackendName() const;

private:
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::SessionOptions> sessionOptions_;
    std::unique_ptr<Ort::Session> session_;

    std::vector<const char*> inputNames_;
    std::vector<const char*> outputNames_;

    std::vector<int64_t> inputShape_;
    std::vector<int64_t> outputShape_;

    Backend backend_;
    bool isLoaded_;
    size_t embeddingDim_;

    std::unique_ptr<AudioPreprocessor> preprocessor_;

    // Book data
    std::vector<std::string> bookChunks_;
    std::vector<std::vector<float>> bookEmbeddings_;

    // Helpers
    std::vector<float> PreprocessAudio(const float* audioBuffer, size_t bufferSize, int sampleRate);
    std::vector<int64_t> TokenizeText(const std::string& text);
    std::vector<int64_t> CreateAttentionMask(size_t length);
    float CosineSimilarity(const std::vector<float>& a, const std::vector<float>& b);
};