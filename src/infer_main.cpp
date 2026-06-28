#include "model.hpp"
#include "tokenizer.hpp"
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <random>
#include <iomanip>
#include <vector>

using namespace ai2;

static Index sample_from_probs(const Vec& probs, Real temperature, std::mt19937_64& rng) {
    if (probs.empty()) return 0;
    if (temperature <= 0) {
        return std::max_element(probs.begin(), probs.end()) - probs.begin();
    }
    // Apply temperature scaling
    Vec scaled(probs.size());
    Real sum = 0;
    Real inv_temp = 1.0 / std::max(temperature, Real(1e-10));
    for (Index i = 0; i < (Index)probs.size(); ++i) {
        scaled[i] = std::pow(probs[i], inv_temp);
        sum += scaled[i];
    }
    if (sum < Real(1e-30)) return 0;
    std::uniform_real_distribution<Real> dist(0, sum);
    Real r = dist(rng);
    for (Index i = 0; i < (Index)probs.size(); ++i) {
        r -= scaled[i];
        if (r <= 0) return i;
    }
    return (Index)probs.size() - 1;
}

static Vec softmax_from_logits(const Vec& logits) {
    if (logits.empty()) return {};
    Real max_l = *std::max_element(logits.begin(), logits.end());
    Vec p(logits.size());
    Real sum = 0;
    for (Index i = 0; i < (Index)logits.size(); ++i) {
        p[i] = std::exp(logits[i] - max_l);
        sum += p[i];
    }
    if (sum > 0) for (auto& v : p) v /= sum;
    return p;
}

static std::string load_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return "";
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

int main(int argc, char** argv) {
    std::string checkpoint_path = "model/model.bin";
    std::string tokenizer_path = "data/tokenizer.vocab";
    Index max_new_tokens = 256;
    Real temperature = 0.8;
    bool interactive = true;
    std::string prompt;
    bool verbose = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--checkpoint" && i + 1 < argc) checkpoint_path = argv[++i];
        else if (arg == "--tokenizer" && i + 1 < argc) tokenizer_path = argv[++i];
        else if (arg == "--max-tokens" && i + 1 < argc) max_new_tokens = std::stoul(argv[++i]);
        else if (arg == "--temperature" && i + 1 < argc) temperature = std::stod(argv[++i]);
        else if (arg == "--prompt" && i + 1 < argc) { prompt = argv[++i]; interactive = false; }
        else if (arg == "--prompt-file" && i + 1 < argc) { prompt = load_file(argv[++i]); interactive = false; }
        else if (arg == "--verbose") verbose = true;
        else if (arg == "--help") {
            std::cout << "AI² Inference — sub-quadratic LLM\n"
                      << "Usage: ./ai2_infer [options]\n"
                      << "  --checkpoint <path>   Model checkpoint (default: model/model.bin)\n"
                      << "  --tokenizer <path>    Tokenizer vocab (default: data/tokenizer.vocab)\n"
                      << "  --max-tokens <N>      Max new tokens to generate (default: 256)\n"
                      << "  --temperature <T>     Sampling temperature (default: 0.8, 0=greedy)\n"
                      << "  --prompt <text>       Prompt string (non-interactive)\n"
                      << "  --prompt-file <path>  Read prompt from file\n"
                      << "  --verbose             Show timing & token info\n"
                      << "  --help                This message\n";
            return 0;
        }
    }

    // --- Load tokenizer ---
    Tokenizer tokenizer;
    tokenizer.load(tokenizer_path);
    Index vocab_size = tokenizer.vocab_size();
    if (verbose) std::cout << "Tokenizer: " << vocab_size << " tokens" << std::endl;

    // --- Create model ---
    ModelConfig cfg;
    cfg.vocab_size = vocab_size;
    cfg.d_model = 256;
    cfg.d_state = 128;
    cfg.n_rf = 256;
    cfg.d_node = 128;
    cfg.n_experts = 32;
    cfg.k_experts = 4;
    cfg.n_ensemble = 3;
    cfg.m_buckets = 16384;
    cfg.sketch_width = 2048;

    Model model(cfg);

    // --- Load checkpoint into output layer params ---
    {
        std::ifstream f(checkpoint_path, std::ios::binary);
        if (!f.is_open()) {
            std::cerr << "Error: cannot open " << checkpoint_path << std::endl;
            return 1;
        }
        Index step;
        f.read(reinterpret_cast<char*>(&step), sizeof(step));
        Index n;
        f.read(reinterpret_cast<char*>(&n), sizeof(n));
        if (n == (Index)model.param_W_out_.size())
            f.read(reinterpret_cast<char*>(model.param_W_out_.data()), n * sizeof(Real));
        f.read(reinterpret_cast<char*>(&n), sizeof(n));
        if (n == (Index)model.param_b_out_.size())
            f.read(reinterpret_cast<char*>(model.param_b_out_.data()), n * sizeof(Real));
        f.close();
    }
    model.sync_params_to_ssog();

    std::mt19937_64 rng(std::random_device{}());

    // --- Generation lambda: reuse Model::forward for the full pipeline ---
    auto generate = [&](const std::vector<Index>& input_ids) -> std::vector<Index> {
        std::vector<Index> tokens = input_ids;
        for (Index step = 0; step < max_new_tokens; ++step) {
            Index seq_len = (Index)tokens.size();

            // Model::forward processes every position and stores logits
            // Pass tokens as both inputs and dummy targets (0 = skip loss)
            Vec token_vec(tokens.begin(), tokens.end());
            Mat batch_tokens = {token_vec};
            Vec zero_vec(seq_len, 0);
            Mat dummy_targets = {zero_vec};
            model.forward(batch_tokens, dummy_targets);

            // Get the logits for the last position
            const auto& batch_logits = model.logits();
            if (batch_logits.empty() || batch_logits[0].size() != (size_t)seq_len) break;
            const Vec& logprobs = batch_logits[0][seq_len - 1];

            // Convert log-probabilities → probabilities → sample
            Vec probs = softmax_from_logits(logprobs);
            Index next = sample_from_probs(probs, temperature, rng);

            tokens.push_back(next);
            if (next == tokenizer.eos_id()) break;
        }
        return tokens;
    };

    // --- Run ---
    if (interactive) {
        std::cout << "\nAI² Inference ready. Type input (Ctrl+D to exit).\n" << std::endl;
        std::string line;
        while (true) {
            std::cout << "> " << std::flush;
            if (!std::getline(std::cin, line)) break;
            if (line.empty()) continue;

            auto input_ids = tokenizer.encode(line);
            if (verbose) std::cout << "  [encoded " << input_ids.size() << " tokens]" << std::endl;

            auto output_ids = generate(input_ids);
            auto output_text = tokenizer.decode(output_ids);

            // Show only the generated portion
            auto input_decoded = tokenizer.decode(input_ids);
            std::string generated;
            if (output_text.size() > input_decoded.size())
                generated = output_text.substr(input_decoded.size());
            else
                generated = output_text;

            std::cout << generated << std::endl;
            if (verbose)
                std::cout << "  [generated " << (output_ids.size() - input_ids.size())
                          << " tokens, total " << output_ids.size() << "]" << std::endl;
        }
        std::cout << std::endl;
    } else {
        auto input_ids = tokenizer.encode(prompt);
        if (verbose) {
            std::cout << "Input (" << input_ids.size() << " tokens): " << prompt << std::endl;
        }
        auto output_ids = generate(input_ids);
        auto output_text = tokenizer.decode(output_ids);
        std::cout << output_text << std::endl;
        if (verbose)
            std::cout << "  [generated " << (output_ids.size() - input_ids.size()) << " tokens]" << std::endl;
    }

    return 0;
}
