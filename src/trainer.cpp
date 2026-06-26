#include "trainer.hpp"
#include <fstream>
#include <sstream>
#include <ctime>
#include <iomanip>
#include <filesystem>

namespace ai2 {

Trainer::Trainer(Model& model, DataLoader& train_loader,
                 DataLoader* eval_loader, const TrainConfig& cfg)
    : model_(model), train_loader_(train_loader),
      eval_loader_(eval_loader), cfg_(cfg),
      optimizer_(Optimizer::ADAM, cfg.lr, 0.9, 0.999, 1e-8, cfg.weight_decay)
{
    // Register trainable parameters with the optimizer
    // For now, we train the SSOG output projection and expert weights
    auto& ssog = model_.ssog();

    // Experts: flat storage of expert_weights_ and expert_biases_
    // For simplicity in this prototype, we register the output projection
    // In a full implementation, we'd need to flatten and register all trainable params
    (void)ssog;
}

void Trainer::train() {
    std::cout << "\n=== Training Started ===" << std::endl;
    std::cout << "  Run: " << cfg_.run_name << std::endl;
    std::cout << "  Max epochs: " << cfg_.num_epochs << std::endl;
    std::cout << "  Learning rate: " << cfg_.lr << std::endl;
    std::cout << "  Batch size: " << std::endl;  // logged from data loader
    std::cout << "  Checkpoint dir: " << cfg_.checkpoint_dir << std::endl;

    // Create checkpoint directory
    std::filesystem::create_directories(cfg_.checkpoint_dir);

    Index global_step = 0;
    Index tokens_processed = 0;
    auto start_time = std::time(nullptr);

    for (Index epoch = 0; epoch < cfg_.num_epochs; ++epoch) {
        std::cout << "\n--- Epoch " << (epoch + 1) << "/" << cfg_.num_epochs << " ---" << std::endl;
        train_loader_.reset();
        Index epoch_step = 0;

        while (!train_loader_.epoch_done()) {
            // Get batch
            Batch batch = train_loader_.next();
            if (batch.tokens.empty()) break;

            // Forward pass
            TrainingMetrics metrics = model_.forward(batch.tokens, batch.targets);
            metrics_ = {metrics};

            // Log progress
            tokens_processed += batch.batch_size * batch.seq_len;
            global_step++;
            epoch_step++;

            if (global_step % cfg_.log_interval == 1 || epoch_step == 1) {
                log_metrics(global_step, metrics);
            }

            // Evaluate
            if (eval_loader_ && global_step % cfg_.eval_interval == 0) {
                TrainingMetrics eval_metrics = evaluate();
                std::cout << "  [Eval] step=" << global_step
                          << " loss=" << eval_metrics.loss
                          << " ppl=" << eval_metrics.perplexity
                          << " acc=" << (eval_metrics.accuracy * 100) << "%"
                          << std::endl;
            }

            // Save checkpoint
            if (global_step % cfg_.save_interval == 0) {
                std::string ckpt_path = cfg_.checkpoint_dir + "/" + cfg_.run_name
                                      + "_step_" + std::to_string(global_step) + ".bin";
                save_checkpoint(ckpt_path);
                std::cout << "  [Checkpoint] saved to " << ckpt_path << std::endl;
            }

            // Check max steps
            if (cfg_.max_steps > 0 && global_step >= cfg_.max_steps) break;
        }

        // End of epoch evaluation
        TrainingMetrics epoch_metrics = metrics_.empty() ? TrainingMetrics{} : metrics_.back();
        std::cout << "  [Epoch " << (epoch + 1) << "] done. "
                  << "loss=" << epoch_metrics.loss
                  << " ppl=" << epoch_metrics.perplexity
                  << std::endl;

        if (cfg_.max_steps > 0 && global_step >= cfg_.max_steps) break;
    }

    auto elapsed = std::time(nullptr) - start_time;
    std::cout << "\n=== Training Complete ===" << std::endl;
    std::cout << "  Total steps: " << global_step << std::endl;
    std::cout << "  Tokens processed: " << tokens_processed << std::endl;
    std::cout << "  Time elapsed: " << elapsed << "s" << std::endl;

    // Save final checkpoint
    std::string final_path = cfg_.checkpoint_dir + "/" + cfg_.run_name + "_final.bin";
    save_checkpoint(final_path);
    std::cout << "  Final checkpoint: " << final_path << std::endl;
}

void Trainer::train_step(const Batch& batch) {
    // Forward pass already handled in train loop
    // In a full implementation, this would compute gradients
    (void)batch;
}

TrainingMetrics Trainer::evaluate() {
    if (!eval_loader_) return TrainingMetrics{};

    eval_loader_->reset();
    TrainingMetrics total;
    Index count = 0;

    while (!eval_loader_->epoch_done()) {
        Batch batch = eval_loader_->next();
        if (batch.tokens.empty()) break;

        TrainingMetrics metrics = model_.forward(batch.tokens, batch.targets);

        total.loss += metrics.loss;
        total.accuracy += metrics.accuracy;
        total.perplexity += metrics.perplexity;
        count++;
    }

    if (count > 0) {
        total.loss /= count;
        total.accuracy /= count;
        total.perplexity = std::exp(total.loss);
    }

    eval_loader_->reset();
    return total;
}

void Trainer::log_metrics(Index step, const TrainingMetrics& metrics) {
    std::cout << "  [Step " << step << "]"
              << " loss=" << metrics.loss
              << " ppl=" << metrics.perplexity
              << " acc=" << (metrics.accuracy * 100) << "%"
              << " conflict=" << metrics.sheaf_conflict
              << std::endl;
}

Real Trainer::get_lr(Index step) const {
    // Linear warmup then cosine decay
    Index warmup = 100;
    Real lr = cfg_.lr;
    if (step < warmup) {
        lr = cfg_.lr * (Real(step) / warmup);
    } else {
        Real progress = Real(step - warmup) / (cfg_.max_steps - warmup + 1);
        lr = cfg_.lr * 0.5 * (1 + std::cos(progress * PI));
    }
    return std::max(lr, cfg_.lr * cfg_.lr_warmup);
}

void Trainer::save_checkpoint(const std::string& path) {
    // In a full implementation, this would serialize model weights
    // For now, create a minimal marker file
    std::ofstream f(path, std::ios::binary);
    if (f.is_open()) {
        // Save config and step info
        Index step = optimizer_.step();
        f.write(reinterpret_cast<const char*>(&step), sizeof(step));
        f.close();
    }
}

void Trainer::load_checkpoint(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (f.is_open()) {
        Index step;
        f.read(reinterpret_cast<char*>(&step), sizeof(step));
        optimizer_.set_step(step);
        f.close();
        std::cout << "  Loaded checkpoint from " << path << " (step " << step << ")" << std::endl;
    }
}

} // namespace ai2
