#include "CLAPInference.h"
#include <iostream>
#include <algorithm>

// Add preprocessor include (already included in header, but keep for clarity)
#include "AudioPipeline.h"

CLAPInference::CLAPInference()
    : backend_(Backend::CPU)
    , isLoaded_(false)
    , embeddingDim_(0)
    , preprocessor_(nullptr)
{
}

CLAPInference::~CLAPInference() {
    // Free strdup'd strings
    for (const char* name : inputNames_) {
        free(const_cast<char*>(name));
    }
    for (const char* name : outputNames_) {
        free(const_cast<char*>(name));
    }
}

bool CLAPInference::Initialize(Backend backend)
{
    backend_ = backend;

    try
    {
        // Create ONNX Runtime environment
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "CLAPInference");

        // Create session options
        sessionOptions_ = std::make_unique<Ort::SessionOptions>();

        if (backend_ == Backend::GPU)
        {
            std::cout << "Initializing ONNX Runtime with GPU (CUDA) backend...\n";
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = 0;
            sessionOptions_->AppendExecutionProvider_CUDA(cuda_options);
        }
        else
        {
            std::cout << "Initializing ONNX Runtime with CPU backend...\n";
            sessionOptions_->SetIntraOpNumThreads(4);
            sessionOptions_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        }

        std::cout << "ONNX Runtime initialized successfully!\n";
        return true;
    }
    catch (const Ort::Exception& e)
    {
        std::cerr << "ONNX Runtime initialization failed: " << e.what() << "\n";
        return false;
    }
}

bool CLAPInference::LoadModel(const std::wstring& modelPath)
{
    if (!env_ || !sessionOptions_)
    {
        std::cerr << "ONNX Runtime not initialized. Call Initialize() first.\n";
        return false;
    }

    try
    {
        std::wcout << L"Loading CLAP combined model from: " << modelPath << L"\n";

        // Create inference session
        session_ = std::make_unique<Ort::Session>(*env_, modelPath.c_str(), *sessionOptions_);

        Ort::AllocatorWithDefaultOptions allocator;

        // Get all input names
        size_t numInputs = session_->GetInputCount();
        std::cout << "Model has " << numInputs << " inputs:\n";

        for (size_t i = 0; i < numInputs; ++i) {
            Ort::AllocatedStringPtr namePtr = session_->GetInputNameAllocated(i, allocator);
            inputNames_.push_back(_strdup(namePtr.get()));
            std::cout << "  Input " << i << ": " << inputNames_[i] << "\n";
        }

        // Get all output names
        size_t numOutputs = session_->GetOutputCount();
        std::cout << "Model has " << numOutputs << " outputs:\n";

        for (size_t i = 0; i < numOutputs; ++i) {
            Ort::AllocatedStringPtr namePtr = session_->GetOutputNameAllocated(i, allocator);
            outputNames_.push_back(_strdup(namePtr.get()));
            std::cout << "  Output " << i << ": " << outputNames_[i] << "\n";
        }

        embeddingDim_ = 512;  // Both text and audio are 512-dim
        std::cout << "Embedding dimension: " << embeddingDim_ << "\n";

        isLoaded_ = true;
        std::cout << "Combined CLAP model loaded successfully!\n";
        return true;
    }
    catch (const Ort::Exception& e)
    {
        std::cerr << "Failed to load model: " << e.what() << "\n";
        return false;
    }
}

std::vector<float> CLAPInference::GetEmbedding(const float* audioBuffer, size_t bufferSize, int sampleRate)
{
    if (!isLoaded_)
    {
        std::cerr << "Model not loaded\n";
        return {};
    }

    try
    {
        // Preprocess audio (convert to mel spectrogram, normalize, etc.)
        std::vector<float> processedAudio = PreprocessAudio(audioBuffer, bufferSize, sampleRate);

        if (processedAudio.empty())
        {
            std::cerr << "Preprocessing produced empty output\n";
            return {};
        }

        // Determine mel/time dimensions from preprocessor
        if (!preprocessor_)
        {
            std::cerr << "Preprocessor missing\n";
            return {};
        }

        const size_t num_mels = preprocessor_->GetNumMelBins();
        if (num_mels == 0)
        {
            std::cerr << "Invalid mel bin count\n";
            return {};
        }

        if (processedAudio.size() % num_mels != 0)
        {
            std::cerr << "Processed audio size is not divisible by num_mels\n";
            return {};
        }

        const size_t num_frames = processedAudio.size() / num_mels;

        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // Try common layout: [1, time_frames, mel_bins]
        std::vector<int64_t> shape_time_first = { 1, 1, static_cast<int64_t>(num_frames), static_cast<int64_t>(num_mels) };

        try
        {
            Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
                memoryInfo,
                processedAudio.data(),
                processedAudio.size(),
                shape_time_first.data(),
                shape_time_first.size()
            );

            auto outputTensors = session_->Run(
                Ort::RunOptions{ nullptr },
                inputNames_.data(),
                &inputTensor,
                1,
                outputNames_.data(),
                1
            );

            float* outputData = outputTensors[0].GetTensorMutableData<float>();
            auto outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
            size_t outputSize = static_cast<size_t>(outputShape.back());
            return std::vector<float>(outputData, outputData + outputSize);
        }
        catch (const Ort::Exception& e)
        {
            // Fallback: model might expect [1, mel_bins, time_frames] — transpose and try again
            std::cerr << "First layout failed, trying transposed layout: " << e.what() << "\n";

            std::vector<float> transposed(processedAudio.size());
            // transpose from [time, mel] (row-major frames) to [mel, time]
            for (size_t t = 0; t < num_frames; ++t)
            {
                for (size_t m = 0; m < num_mels; ++m)
                {
                    transposed[m * num_frames + t] = processedAudio[t * num_mels + m];
                }
            }

            std::vector<int64_t> shape_mel_first = { 1, 1, static_cast<int64_t>(num_mels), static_cast<int64_t>(num_frames) };

            try
            {
                Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
                    memoryInfo,
                    transposed.data(),
                    transposed.size(),
                    shape_mel_first.data(),
                    shape_mel_first.size()
                );

                auto outputTensors = session_->Run(
                    Ort::RunOptions{ nullptr },
                    inputNames_.data(),
                    &inputTensor,
                    1,
                    outputNames_.data(),
                    1
                );

                float* outputData = outputTensors[0].GetTensorMutableData<float>();
                auto outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
                size_t outputSize = static_cast<size_t>(outputShape.back());
                return std::vector<float>(outputData, outputData + outputSize);
            }
            catch (const Ort::Exception& e2)
            {
                std::cerr << "Inference failed with transposed layout too: " << e2.what() << "\n";
                return {};
            }
        }
    }
    catch (const Ort::Exception& e)
    {
        std::cerr << "Inference failed: " << e.what() << "\n";
        return {};
    }
}

size_t CLAPInference::GetEmbeddingDim() const
{
    return embeddingDim_;
}

bool CLAPInference::IsLoaded() const
{
    return isLoaded_;
}

const char* CLAPInference::GetBackendName() const
{
    return backend_ == Backend::GPU ? "GPU (CUDA)" : "CPU";
}

std::vector<float> CLAPInference::PreprocessAudio(const float* audioBuffer, size_t bufferSize, int sampleRate)
{
    // Lazily create a preprocessor with default config if none exists.
    if (!preprocessor_)
    {
        PreprocessorConfig cfg; // uses defaults (sampling_rate, n_fft, etc.)
        preprocessor_ = std::make_unique<AudioPreprocessor>(cfg);
    }

    // Delegate to the AudioPreprocessor implementation (resample, STFT, mel filters, etc.)
    return preprocessor_->Process(audioBuffer, bufferSize, sampleRate);
}


std::vector<float> CLAPInference::GetEmbeddingFromMelSpec(const float* mel_spec,
    size_t num_frames, size_t num_mels) {
    if (!isLoaded_) {
        std::cerr << "Model not loaded\n";
        return {};
    }

    try {
        size_t total = num_frames * num_mels;
        std::vector<float> input_copy(mel_spec, mel_spec + total);

        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // Create dummy text inputs (not used for audio encoding)
        std::vector<int64_t> dummyTokens(77, 0);
        dummyTokens[0] = 49406;  // Start token
        dummyTokens[1] = 49407;  // End token

        std::vector<int64_t> dummyMask(77, 1);

        // Try layout: [batch=1, channel=1, time_frames, mel_bins]
        std::vector<int64_t> shape_time_first = { 1, 1, static_cast<int64_t>(num_frames),
                                                  static_cast<int64_t>(num_mels) };

        try {
            // Create all 3 required inputs
            std::vector<int64_t> tokenShape = { 1, 77 };
            Ort::Value tokenTensor = Ort::Value::CreateTensor<int64_t>(
                memoryInfo, dummyTokens.data(), dummyTokens.size(), tokenShape.data(), tokenShape.size()
            );

            Ort::Value audioTensor = Ort::Value::CreateTensor<float>(
                memoryInfo,
                input_copy.data(),
                input_copy.size(),
                shape_time_first.data(),
                shape_time_first.size()
            );

            Ort::Value maskTensor = Ort::Value::CreateTensor<int64_t>(
                memoryInfo, dummyMask.data(), dummyMask.size(), tokenShape.data(), tokenShape.size()
            );

            // Prepare inputs in order: input_ids, input_features, attention_mask
            std::vector<Ort::Value> inputs;
            inputs.push_back(std::move(tokenTensor));
            inputs.push_back(std::move(audioTensor));
            inputs.push_back(std::move(maskTensor));

            auto outputTensors = session_->Run(
                Ort::RunOptions{ nullptr },
                inputNames_.data(),
                inputs.data(),
                inputs.size(),
                outputNames_.data(),
                outputNames_.size()
            );

            // Extract audio_embeds (output index 3)
            float* outputData = outputTensors[3].GetTensorMutableData<float>();
            return std::vector<float>(outputData, outputData + embeddingDim_);
        }
        catch (const Ort::Exception& e) {
            std::cerr << "First layout failed, trying transposed layout: " << e.what() << "\n";

            // Transpose to [batch=1, channel=1, mel_bins, time_frames]
            std::vector<float> transposed(total);
            for (size_t t = 0; t < num_frames; ++t) {
                for (size_t m = 0; m < num_mels; ++m) {
                    transposed[m * num_frames + t] = mel_spec[t * num_mels + m];
                }
            }

            std::vector<int64_t> shape_mel_first = { 1, 1, static_cast<int64_t>(num_mels),
                                                     static_cast<int64_t>(num_frames) };

            try {
                // Create all 3 required inputs with transposed audio
                std::vector<int64_t> tokenShape = { 1, 77 };
                Ort::Value tokenTensor = Ort::Value::CreateTensor<int64_t>(
                    memoryInfo, dummyTokens.data(), dummyTokens.size(), tokenShape.data(), tokenShape.size()
                );

                Ort::Value audioTensor = Ort::Value::CreateTensor<float>(
                    memoryInfo,
                    transposed.data(),
                    transposed.size(),
                    shape_mel_first.data(),
                    shape_mel_first.size()
                );

                Ort::Value maskTensor = Ort::Value::CreateTensor<int64_t>(
                    memoryInfo, dummyMask.data(), dummyMask.size(), tokenShape.data(), tokenShape.size()
                );

                std::vector<Ort::Value> inputs;
                inputs.push_back(std::move(tokenTensor));
                inputs.push_back(std::move(audioTensor));
                inputs.push_back(std::move(maskTensor));

                auto outputTensors = session_->Run(
                    Ort::RunOptions{ nullptr },
                    inputNames_.data(),
                    inputs.data(),
                    inputs.size(),
                    outputNames_.data(),
                    outputNames_.size()
                );

                // Extract audio_embeds (output index 3)
                float* outputData = outputTensors[3].GetTensorMutableData<float>();
                return std::vector<float>(outputData, outputData + embeddingDim_);
            }
            catch (const Ort::Exception& e2) {
                std::cerr << "Inference failed with transposed layout too: " << e2.what() << "\n";
                return {};
            }
        }
    }
    catch (const Ort::Exception& e) {
        std::cerr << "Inference failed: " << e.what() << "\n";
        return {};
    }
}

std::vector<int64_t> CLAPInference::TokenizeText(const std::string& text) {
    // CLAP uses CLIP tokenization: max 77 tokens
    // Token IDs: start=49406, end=49407, pad=0

    std::vector<int64_t> tokens(77, 0);  // Initialize with padding
    tokens[0] = 49406;  // Start token

    // Simple character-based tokenization
    // For better results, you'd want a proper BPE tokenizer
    size_t pos = 1;
    for (size_t i = 0; i < text.length() && pos < 76; ++i) {
        unsigned char c = text[i];

        // Map printable ASCII to token range
        if (c >= 32 && c < 127) {
            tokens[pos++] = 259 + (c - 32);
        }
        else if (c == ' ') {
            tokens[pos++] = 220;  // Space token
        }
    }

    tokens[pos] = 49407;  // End token

    return tokens;
}

std::vector<int64_t> CLAPInference::CreateAttentionMask(size_t length) {
    // Attention mask: 1 for real tokens, 0 for padding
    std::vector<int64_t> mask(77, 0);
    for (size_t i = 0; i < std::min(length, size_t(77)); ++i) {
        mask[i] = 1;
    }
    return mask;
}

std::vector<float> CLAPInference::GetTextEmbedding(const std::string& text) {
    if (!isLoaded_) {
        std::cerr << "Model not loaded\n";
        return {};
    }

    try {
        // Tokenize text
        std::vector<int64_t> tokens = TokenizeText(text);
        std::vector<int64_t> attentionMask = CreateAttentionMask(tokens.size());

        // IMPORTANT: Check what shape audio_model expects by looking at model inspection
        // From your output: input_features has dynamic dims, but BatchNorm expects 256
        // The model expects: [batch, channels, height, width] = [1, 1, mel_bins, time_frames]

        // Create dummy audio with correct mel spectrogram dimensions
        // Based on your preprocessor: 64 mel bins, and typical time frames
        const size_t mel_bins = 64;  // Your preprocessor config
        const size_t time_frames = 256;  // Expected by BatchNorm
        std::vector<float> dummyAudio(1 * 1 * time_frames * mel_bins, 0.0f);

        Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // Input tensors
        std::vector<int64_t> tokenShape = { 1, 77 };
        Ort::Value tokenTensor = Ort::Value::CreateTensor<int64_t>(
            memInfo, tokens.data(), tokens.size(), tokenShape.data(), tokenShape.size()
        );

        std::vector<int64_t> audioShape = { 1, 1, static_cast<int64_t>(time_frames), static_cast<int64_t>(mel_bins) };
        Ort::Value audioTensor = Ort::Value::CreateTensor<float>(
            memInfo, dummyAudio.data(), dummyAudio.size(), audioShape.data(), audioShape.size()
        );

        Ort::Value maskTensor = Ort::Value::CreateTensor<int64_t>(
            memInfo, attentionMask.data(), attentionMask.size(), tokenShape.data(), tokenShape.size()
        );

        // Prepare inputs in order: input_ids, input_features, attention_mask
        std::vector<Ort::Value> inputs;
        inputs.push_back(std::move(tokenTensor));
        inputs.push_back(std::move(audioTensor));
        inputs.push_back(std::move(maskTensor));

        // Run inference
        auto outputs = session_->Run(
            Ort::RunOptions{ nullptr },
            inputNames_.data(),
            inputs.data(),
            inputs.size(),
            outputNames_.data(),
            outputNames_.size()
        );

        // Extract text_embeds (output index 2)
        float* textEmbedData = outputs[2].GetTensorMutableData<float>();

        return std::vector<float>(textEmbedData, textEmbedData + embeddingDim_);
    }
    catch (const Ort::Exception& e) {
        std::cerr << "Text encoding failed: " << e.what() << "\n";
        return {};
    }
}

bool CLAPInference::LoadBookFromFile(const std::string& bookPath, size_t chunkSize) {
    std::cout << "Loading book from: " << bookPath << "\n";

    std::ifstream file(bookPath);
    if (!file) {
        std::cerr << "Failed to open book file!\n";
        return false;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string text = buffer.str();

    std::cout << "Book loaded: " << text.length() << " characters\n";

    // Chunk with overlap
    std::cout << "Chunking text...\n";
    const size_t overlap = 50;

    for (size_t i = 0; i < text.length(); i += chunkSize - overlap) {
        size_t end = std::min(i + chunkSize, text.length());
        std::string chunk = text.substr(i, end - i);

        // Clean whitespace
        chunk.erase(0, chunk.find_first_not_of(" \t\n\r"));
        chunk.erase(chunk.find_last_not_of(" \t\n\r") + 1);

        if (chunk.length() > 50) {
            bookChunks_.push_back(chunk);
        }

        if (bookChunks_.size() % 100 == 0) {
            std::cout << "  Created " << bookChunks_.size() << " chunks...\n";
        }
    }

    std::cout << "Created " << bookChunks_.size() << " chunks total\n";

    // Generate text embeddings
    std::cout << "Generating text embeddings (this may take a while)...\n";
    for (size_t i = 0; i < bookChunks_.size(); ++i) {
        if (i % 50 == 0) {
            std::cout << "  Progress: " << i << "/" << bookChunks_.size() << "\n";
        }

        auto embedding = GetTextEmbedding(bookChunks_[i]);
        if (!embedding.empty()) {
            bookEmbeddings_.push_back(embedding);
        }
    }

    std::cout << "Generated " << bookEmbeddings_.size() << " text embeddings!\n";
    return true;
}

float CLAPInference::CosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return -1.0f;

    float dot = 0.0f, mag_a = 0.0f, mag_b = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        mag_a += a[i] * a[i];
        mag_b += b[i] * b[i];
    }

    if (mag_a == 0.0f || mag_b == 0.0f) return 0.0f;
    return dot / (std::sqrt(mag_a) * std::sqrt(mag_b));
}

std::string CLAPInference::FindBestMatchingChunk(const std::vector<float>& audioEmbedding) {
    if (bookEmbeddings_.empty()) {
        std::cerr << "No book embeddings loaded!\n";
        return "";
    }

    float bestSim = -1.0f;
    size_t bestIdx = 0;

    for (size_t i = 0; i < bookEmbeddings_.size(); ++i) {
        float sim = CosineSimilarity(audioEmbedding, bookEmbeddings_[i]);
        if (sim > bestSim) {
            bestSim = sim;
            bestIdx = i;
        }
    }

    std::cout << "  Best match (similarity: " << bestSim << "):\n";
    std::cout << "  \"" << bookChunks_[bestIdx].substr(0, 150) << "...\"\n";

    return bookChunks_[bestIdx];
}