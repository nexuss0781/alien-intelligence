#pragma once
#include "types.hpp"
#include "model.hpp"
#include "dataloader.hpp"
#include "optimizer.hpp"
#include <string>
#include <vector>

namespace ai2 {

struct TrainConfig {
    Index num_epochs = 3;
    Index log_interval = 10;
    Index eval_interval = 100;
    Index save_interval = 500;
    Real lr = 0.001;
    Real lr_warmup = 0.1;
    Real weight_decay = 0.01;
    Real grad_clip = 1.0;
    Index max_steps = 0;  // 0 = full epochs
    std::string checkpoint_dir = "checkpoints";
    std::string run_name = "ai2_run";
};

class Trainer {
public:
    Trainer(Model& model, DataLoader& train_loader,
            DataLoader* eval_loader, const TrainConfig& cfg);

    // Run training
    void train();

    // Save checkpoint
    void save_checkpoint(const std::string& path);

    // Load checkpoint
    void load_checkpoint(const std::string& path);

    // Metrics
    const std::vector<TrainingMetrics>& train_metrics() const { return metrics_; }

private:
    Model& model_;
    DataLoader& train_loader_;
    DataLoader* eval_loader_;
    TrainConfig cfg_;
    Optimizer optimizer_;
    std::vector<TrainingMetrics> metrics_;

    // Training loop
    void train_step(const Batch& batch);
    TrainingMetrics evaluate();

    // Logging
    void log_metrics(Index step, const TrainingMetrics& metrics);
    Real get_lr(Index step) const;
};

} // namespace ai2
