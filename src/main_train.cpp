#include "model.hpp"
#include "dataloader.hpp"
#include "tokenizer.hpp"
#include "trainer.hpp"
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <filesystem>

using namespace ai2;

static std::string find_latest_checkpoint(const std::string& dir,
                                           const std::string& prefix) {
    namespace fs = std::filesystem;
    if (!fs::is_directory(dir)) return {};

    std::string latest;
    Index max_step = 0;

    for (const auto& entry : fs::directory_iterator(dir)) {
        std::string name = entry.path().filename().string();
        for (const char* suffix : {"_step_", "_epoch_"}) {
            std::string match = prefix + suffix;
            if (name.rfind(match, 0) == 0 && name.size() > match.size()) {
                std::string num_part = name.substr(match.size());
                auto dot = num_part.rfind(".bin");
                if (dot != std::string::npos) num_part = num_part.substr(0, dot);
                try {
                    Index step = std::stoull(num_part);
                    if (step > max_step) {
                        max_step = step;
                        latest = entry.path().string();
                    }
                } catch (...) {}
            }
        }
        if (name == prefix + "_final.bin") {
            if (max_step == 0) latest = entry.path().string();
            max_step = Index(-1);
            latest = entry.path().string();
        }
    }
    return latest;
}

static void load_weights_into(Model& model, const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return;
    Index step, n;
    f.read(reinterpret_cast<char*>(&step), sizeof(step));
    f.read(reinterpret_cast<char*>(&n), sizeof(n));
    if (n == (Index)model.param_W_out_.size())
        f.read(reinterpret_cast<char*>(model.param_W_out_.data()), n * sizeof(Real));
    f.read(reinterpret_cast<char*>(&n), sizeof(n));
    if (n == (Index)model.param_b_out_.size())
        f.read(reinterpret_cast<char*>(model.param_b_out_.data()), n * sizeof(Real));
    f.close();
    model.sync_params_to_ssog();
    std::cout << "  Loaded weights from " << path << " (step " << step << ")" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "=== Alien Intelligence (AI²) Training ===" << std::endl;

    std::string data_dir = "data";
    std::string pretrain_file = data_dir + "/pretrain.txt";
    std::string finetune_file = data_dir + "/alpaca_cleaned.txt";
    std::string checkpoint_dir = "checkpoints";
    std::string model_dir = "model";
    bool resume = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--resume" || arg == "-r") resume = true;
    }

    {
        std::ifstream f(pretrain_file);
        if (!f.good()) {
            std::cout << "Datasets not found. Run: bash scripts/download_datasets.sh" << std::endl;
            return 1;
        }
    }

    std::filesystem::create_directories(checkpoint_dir);
    std::filesystem::create_directories(model_dir);

    // --- Tokenizer & ModelConfig ---
    Tokenizer tokenizer;
    {
        std::ifstream f(pretrain_file);
        std::stringstream ss;
        ss << f.rdbuf();
        std::string text = ss.str();
        tokenizer.build_from_chars(text);
        std::cout << "Tokenizer vocab size: " << tokenizer.vocab_size() << std::endl;
        tokenizer.save(data_dir + "/tokenizer.vocab");
    }

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

    std::string model_bin = model_dir + "/model.bin";

    // --- Stage 1: Pretraining ---
    std::string ft_ckpt = find_latest_checkpoint(checkpoint_dir, "ai2_finetune");
    bool skip_pretrain = resume && !ft_ckpt.empty();

    if (skip_pretrain) {
        std::cout << "\nFinetune checkpoint exists, skipping pretrain stage." << std::endl;
    } else {
        std::cout << "\n--- Stage 1: Pretraining ---" << std::endl;
        DataLoader train_loader(pretrain_file, tokenizer, 64, 128);
        DataLoader eval_loader(pretrain_file, tokenizer, 32, 128);

        Model model(cfg);
        std::cout << "Model: d_model=" << cfg.d_model
                  << " d_state=" << cfg.d_state
                  << " vocab=" << cfg.vocab_size
                  << " experts=" << cfg.n_experts << std::endl;

        TrainConfig train_cfg;
        train_cfg.num_epochs = 3;
        train_cfg.log_interval = 50;
        train_cfg.eval_interval = 200;
        train_cfg.save_interval = 100;
        train_cfg.epoch_save_interval = 3;
        train_cfg.lr = 0.001;
        train_cfg.run_name = "ai2_pretrain";
        train_cfg.checkpoint_dir = checkpoint_dir;
        train_cfg.model_path = model_bin;

        Trainer trainer(model, train_loader, &eval_loader, train_cfg);

        if (resume) {
            std::string ckpt = find_latest_checkpoint(checkpoint_dir, train_cfg.run_name);
            if (!ckpt.empty()) {
                std::cout << "  Resuming from " << ckpt << std::endl;
                trainer.load_checkpoint(ckpt);
            }
        }

        trainer.train();
    }

    // --- Stage 2: Finetuning ---
    std::cout << "\n--- Stage 2: Finetuning on Instructions ---" << std::endl;
    {
        std::ifstream f(finetune_file);
        if (!f.good()) {
            std::cout << "Finetuning dataset not found, skipping." << std::endl;
            return 0;
        }

        DataLoader ft_loader(finetune_file, tokenizer, 32, 256);
        DataLoader ft_eval(finetune_file, tokenizer, 16, 256);

        Model ft_model(cfg);

        TrainConfig ft_cfg;
        ft_cfg.num_epochs = 2;
        ft_cfg.log_interval = 20;
        ft_cfg.eval_interval = 50;
        ft_cfg.save_interval = 100;
        ft_cfg.epoch_save_interval = 3;
        ft_cfg.lr = 0.0005;
        ft_cfg.run_name = "ai2_finetune";
        ft_cfg.checkpoint_dir = "checkpoints";
        ft_cfg.model_path = model_bin;

        Trainer ft_trainer(ft_model, ft_loader, &ft_eval, ft_cfg);

        if (resume && !ft_ckpt.empty()) {
            std::cout << "  Resuming finetune from " << ft_ckpt << std::endl;
            ft_trainer.load_checkpoint(ft_ckpt);
        } else if (!resume && std::ifstream(model_bin).good()) {
            // First finetune run: bootstrap from pretrained model.bin
            load_weights_into(ft_model, model_bin);
        } else {
            std::cout << "  Starting finetune from scratch (no pretrained weights)." << std::endl;
        }

        ft_trainer.train();
    }

    std::cout << "\n=== Training Complete! ===" << std::endl;
    std::cout << "Latest model: " << model_bin << std::endl;
    std::cout << "Run inference with: ./build/ai2_infer" << std::endl;

    return 0;
}
