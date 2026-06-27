#include "model.hpp"
#include "dataloader.hpp"
#include "tokenizer.hpp"
#include "trainer.hpp"
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

using namespace ai2;

int main(int /*argc*/, char** /*argv*/) {
    #ifdef _OPENMP
    omp_set_num_threads(omp_get_num_procs());
    std::cout << "OpenMP enabled: " << omp_get_num_procs() << " threads" << std::endl;
    #endif
    std::cout << "=== Alien Intelligence (AI²) Training ===" << std::endl;

    // Configuration
    std::string data_dir = "data";
    std::string pretrain_file = data_dir + "/pretrain.txt";
    std::string finetune_file = data_dir + "/alpaca_cleaned.txt";

    // Check if datasets exist, if not run download script
    {
        std::ifstream f(pretrain_file);
        if (!f.good()) {
            std::cout << "Datasets not found. Run: bash scripts/download_datasets.sh" << std::endl;
            return 1;
        }
    }

    // ========== Stage 1: Pretraining ==========
    std::cout << "\n--- Stage 1: Pretraining ---" << std::endl;

    // Build tokenizer from pretraining data
    Tokenizer tokenizer;
    {
        std::ifstream f(pretrain_file);
        std::stringstream ss;
        ss << f.rdbuf();
        std::string text = ss.str();
        tokenizer.build_from_chars(text);
        std::cout << "Tokenizer vocab size: " << tokenizer.vocab_size() << std::endl;

        // Save tokenizer
        tokenizer.save(data_dir + "/tokenizer.vocab");
    }

    // Create data loaders (large batch for CPU parallelism)
    DataLoader train_loader(pretrain_file, tokenizer, 64, 128);
    DataLoader eval_loader(pretrain_file, tokenizer, 32, 128);

    // Initialize model (bigger model = better CPU utilization)
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
    std::cout << "Model created." << std::endl;
    std::cout << "  d_model=" << cfg.d_model
              << " d_state=" << cfg.d_state
              << " vocab=" << cfg.vocab_size
              << " experts=" << cfg.n_experts
              << std::endl;

    // Run pretraining
    {
        TrainConfig train_cfg;
        train_cfg.num_epochs = 3;
        train_cfg.log_interval = 50;
        train_cfg.eval_interval = 200;
        train_cfg.save_interval = 500;
        train_cfg.lr = 0.001;
        train_cfg.run_name = "ai2_pretrain";
        train_cfg.checkpoint_dir = "checkpoints";

        Trainer trainer(model, train_loader, &eval_loader, train_cfg);
        trainer.train();
    }

    // ========== Stage 2: Finetuning ==========
    std::cout << "\n--- Stage 2: Finetuning on Instructions ---" << std::endl;

    {
        std::ifstream f(finetune_file);
        if (f.good()) {
        DataLoader ft_loader(finetune_file, tokenizer, 32, 256);
        DataLoader ft_eval(finetune_file, tokenizer, 16, 256);

            ModelConfig ft_cfg = cfg;
            ft_cfg.d_model = 256;
            ft_cfg.r_lora = 8;

            Model ft_model(ft_cfg);

            TrainConfig ft_train_cfg;
            ft_train_cfg.num_epochs = 2;
            ft_train_cfg.log_interval = 20;
            ft_train_cfg.eval_interval = 50;
            ft_train_cfg.save_interval = 200;
            ft_train_cfg.lr = 0.0005;
            ft_train_cfg.run_name = "ai2_finetune";
            ft_train_cfg.checkpoint_dir = "checkpoints";

            Trainer ft_trainer(ft_model, ft_loader, &ft_eval, ft_train_cfg);
            ft_trainer.train();
        } else {
            std::cout << "Finetuning dataset not found, skipping." << std::endl;
        }
    }

    std::cout << "\n=== Training Complete! ===" << std::endl;
    std::cout << "Checkpoints saved to checkpoints/" << std::endl;
    std::cout << "Run inference with: ./ai2_infer" << std::endl;

    return 0;
}
