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

using namespace ai2;

static Index multinomial_sample(const Vec& probs, Real temperature, std::mt19937_64& rng) {
    if (temperature <= 0) {
        return std::max_element(probs.begin(), probs.end()) - probs.begin();
    }
    Vec scaled(probs.size());
    Real sum = 0;
    for (Index i = 0; i < probs.size(); ++i) {
        scaled[i] = std::pow(probs[i], 1.0 / std::max(temperature, EPS));
        sum += scaled[i];
    }
    if (sum < EPS) return 0;
    std::uniform_real_distribution<Real> dist(0, sum);
    Real r = dist(rng);
    for (Index i = 0; i < probs.size(); ++i) {
        r -= scaled[i];
        if (r <= 0) return i;
    }
    return probs.size() - 1;
}

static std::string load_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return "";
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

int main(int argc, char** argv) {
    std::string checkpoint_path = "checkpoints/ai2_pretrain_final.bin";
    std::string tokenizer_path = "data/tokenizer.vocab";
    Index max_tokens = 256;
    Real temperature = 0.8;
    bool interactive = true;
    std::string prompt;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--checkpoint" && i + 1 < argc) checkpoint_path = argv[++i];
        else if (arg == "--tokenizer" && i + 1 < argc) tokenizer_path = argv[++i];
        else if (arg == "--max-tokens" && i + 1 < argc) max_tokens = std::stoul(argv[++i]);
        else if (arg == "--temperature" && i + 1 < argc) temperature = std::stod(argv[++i]);
        else if (arg == "--prompt" && i + 1 < argc) { prompt = argv[++i]; interactive = false; }
        else if (arg == "--prompt-file" && i + 1 < argc) { prompt = load_file(argv[++i]); interactive = false; }
        else if (arg == "--help") {
            std::cout << "AI² Inference\n"
                      << "Usage: ./ai2_infer [options]\n"
                      << "  --checkpoint <path>   Model checkpoint (default: checkpoints/ai2_pretrain_final.bin)\n"
                      << "  --tokenizer <path>    Tokenizer vocab (default: data/tokenizer.vocab)\n"
                      << "  --max-tokens <N>      Max tokens to generate (default: 256)\n"
                      << "  --temperature <T>     Sampling temperature (default: 0.8, 0=greedy)\n"
                      << "  --prompt <text>       Prompt string (non-interactive)\n"
                      << "  --prompt-file <path>  Read prompt from file\n"
                      << "  --help                This message\n";
            return 0;
        }
    }

    Tokenizer tokenizer;
    tokenizer.load(tokenizer_path);
    std::cout << "Tokenizer: " << tokenizer.vocab_size() << " tokens" << std::endl;

    ModelConfig cfg;
    cfg.vocab_size = tokenizer.vocab_size();
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

    // Load checkpoint
    {
        std::ifstream f(checkpoint_path, std::ios::binary);
        if (!f.is_open()) {
            std::cerr << "Error: cannot open checkpoint " << checkpoint_path << std::endl;
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
        model.sync_params_to_ssog();
        f.close();
        std::cout << "Checkpoint loaded (step " << step << "): "
                  << checkpoint_path << std::endl;
    }

    std::mt19937_64 rng(std::random_device{}());
    auto& slie = model.slie();
    auto& lssc = model.lssc();
    auto& stre = model.stre();
    auto& ssog = model.ssog();
    Index d_model = cfg.d_model;
    Index d_node = cfg.d_node;
    Index vocab_size = cfg.vocab_size;

    auto generate = [&](const std::vector<Index>& input_ids) -> std::vector<Index> {
        std::vector<Index> tokens = input_ids;
        for (Index gen_step = 0; gen_step < max_tokens; ++gen_step) {
            Index seq_len = tokens.size();

            // SLIE
            Mat embeddings(seq_len, Vec(d_model, 0));
            Vec prev_pos(slie.d_pos(), 0);
            slie.reset_position();
            for (Index t = 0; t < seq_len; ++t) {
                Index token = tokens[t];
                if (token >= vocab_size) token = vocab_size - 1;
                embeddings[t] = slie.forward(token, prev_pos);
                prev_pos = slie.last_position();
            }

            // LSSC
            Mat hidden = lssc.forward(embeddings);

            // STRE
            auto stre_features = stre.forward(hidden, cfg.n_layers);
            for (Index t = 0; t < seq_len; ++t) {
                Vec& h = hidden[t];
                Index node_idx = t < stre_features.size() ? t : stre_features.size() - 1;
                if (node_idx < stre_features.size()) {
                    for (Index i = 0; i < std::min<Index>(d_node, d_model); ++i) {
                        h[i] += 0.1 * stre_features[node_idx][i % d_node];
                    }
                }
            }

            // SSOG on last position only
            const Vec& h_last = hidden[seq_len - 1];
            auto output = ssog.forward(h_last, 0, {});
            Index next = multinomial_sample(output.calibrated_probs, temperature, rng);

            tokens.push_back(next);
            if (next == tokenizer.eos_id()) break;
        }
        return tokens;
    };

    if (interactive) {
        std::cout << "\nAI² Inference ready. Type input (Ctrl+D to exit).\n" << std::endl;
        std::string line;
        while (true) {
            std::cout << "> " << std::flush;
            if (!std::getline(std::cin, line)) break;
            if (line.empty()) continue;

            std::vector<Index> input_ids = tokenizer.encode(line);
            std::cout << "  [encoded " << input_ids.size() << " tokens]" << std::endl;

            auto output_ids = generate(input_ids);
            std::string output_text = tokenizer.decode(output_ids);

            // Trim the input from the output for display
            std::string input_decoded = tokenizer.decode(input_ids);
            std::string generated;
            if (output_text.size() > input_decoded.size())
                generated = output_text.substr(input_decoded.size());
            else
                generated = output_text;

            std::cout << "  " << generated << std::endl;
            std::cout << "  [generated " << (output_ids.size() - input_ids.size())
                      << " tokens, total " << output_ids.size() << "]" << std::endl;
        }
        std::cout << std::endl;
    } else {
        std::vector<Index> input_ids = tokenizer.encode(prompt);
        std::cout << "Input: " << prompt << std::endl;
        std::cout << "  [encoded " << input_ids.size() << " tokens]" << std::endl;

        auto output_ids = generate(input_ids);
        std::string output_text = tokenizer.decode(output_ids);
        std::cout << "Output: " << output_text << std::endl;
        std::cout << "  [generated " << (output_ids.size() - input_ids.size())
                  << " tokens]" << std::endl;
    }

    return 0;
}
