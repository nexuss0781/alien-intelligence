#pragma once
#include "types.hpp"
#include "model.hpp"
#include "dataloader.hpp"
#include "optimizer.hpp"
#include "hidden_cache.hpp"
#include <string>
#include <vector>
#include <map>
#include <fstream>

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
    Index max_steps = 0;
    std::string checkpoint_dir = "checkpoints";
    std::string run_name = "ai2_run";
    std::string log_file = "";
};

class Trainer {
public:
    Trainer(Model& model, DataLoader& train_loader,
            DataLoader* eval_loader, const TrainConfig& cfg);

    void train();
    void save_checkpoint(const std::string& path);
    void load_checkpoint(const std::string& path);

    const std::vector<TrainingMetrics>& train_metrics() const { return metrics_; }

private:
    Model& model_;
    DataLoader& train_loader_;
    DataLoader* eval_loader_;
    TrainConfig cfg_;
    Optimizer optimizer_;
    std::vector<TrainingMetrics> metrics_;
    std::ofstream log_stream_;
    HiddenCache hidden_cache_;

    void build_hidden_cache();
    TrainingMetrics train_with_cache();
    TrainingMetrics evaluate();
    void log_metrics(Index step, const TrainingMetrics& metrics,
                     const std::map<std::string, Real>& extra = {});
    void log(const std::string& msg);
    Real get_lr(Index step) const;
};

} // namespace ai2
