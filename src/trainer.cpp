#include "trainer.hpp"
#include <fstream>
#include <sstream>
#include <ctime>
#include <iomanip>
#include <filesystem>
#include <cmath>

namespace ai2 {

Trainer::Trainer(Model& model, DataLoader& train_loader,
                 DataLoader* eval_loader, const TrainConfig& cfg)
    : model_(model), train_loader_(train_loader),
      eval_loader_(eval_loader), cfg_(cfg),
      optimizer_(Optimizer::ADAM, cfg.lr, 0.9, 0.999, 1e-8, cfg.weight_decay)
{
    optimizer_.add_param("W_out", &model_.param_W_out_, &model_.grad_W_out_);
    optimizer_.add_param("b_out", &model_.param_b_out_, &model_.grad_b_out_);

    // Open log file if configured
    if (!cfg_.log_file.empty()) {
        log_stream_.open(cfg_.log_file, std::ios::out);
    }
}

void Trainer::train() {
    // Dynamically calculate total steps
    Index batches_per_epoch = train_loader_.batches_per_epoch();
    Index total_batches = batches_per_epoch * cfg_.num_epochs;
    if (cfg_.max_steps > 0 && cfg_.max_steps < total_batches)
        total_batches = cfg_.max_steps;
    Index batch_size = train_loader_.batch_size();
    Index seq_len = train_loader_.seq_len();
    Index tok_per_step = batch_size * seq_len;
    Real total_tokens = Real(total_batches) * tok_per_step;

    std::cout << "\n=== Training Started ===" << "\n"
              << "  Run: " << cfg_.run_name << "\n"
              << "  Epochs: " << cfg_.num_epochs << "\n"
              << "  Batch size: " << batch_size << "  Seq len: " << seq_len << "\n"
              << "  Tokens/step: " << tok_per_step << "\n"
              << "  Steps/epoch: " << batches_per_epoch << "\n"
              << "  Total steps: " << total_batches << "\n"
              << "  Total tokens: " << std::llround(total_tokens) << "\n"
              << "  Learning rate: " << cfg_.lr << "\n"
              << "  Optimizer params: " << optimizer_.num_params() << "\n"
              << "  Checkpoint dir: " << cfg_.checkpoint_dir << "\n"
              << std::endl;

    log("=== Training Started ===");
    log("  Run: " + cfg_.run_name + "  Steps: " + std::to_string(total_batches) +
        "  Tokens: " + std::to_string(std::llround(total_tokens)));

    std::filesystem::create_directories(cfg_.checkpoint_dir);

    Index global_step = 0;
    Index tokens_processed = 0;
    auto start_time = std::time(nullptr);
    auto step_start = start_time;

    for (Index epoch = 0; epoch < cfg_.num_epochs; ++epoch) {
        train_loader_.reset();
        Index epoch_step = 0;

        std::cout << "--- Epoch " << (epoch + 1) << "/" << cfg_.num_epochs << " ---" << std::endl;
        log("--- Epoch " + std::to_string(epoch + 1) + "/" + std::to_string(cfg_.num_epochs) + " ---");

        while (!train_loader_.epoch_done()) {
            Batch batch = train_loader_.next();
            if (batch.tokens.empty()) break;

            auto batch_start = std::time(nullptr);

            // Forward
            TrainingMetrics metrics = model_.forward(batch.tokens, batch.targets);

            // Backward
            model_.zero_gradients();
            model_.compute_gradients(batch.targets);

            // Gradient norm
            Real grad_norm = 0;
            for (auto& g : model_.grad_W_out_) grad_norm += g * g;
            for (auto& g : model_.grad_b_out_) grad_norm += g * g;
            grad_norm = std::sqrt(grad_norm);

            // Gradient clipping
            if (grad_norm > cfg_.grad_clip) {
                Real scale = cfg_.grad_clip / (grad_norm + EPS);
                for (auto& g : model_.grad_W_out_) g *= scale;
                for (auto& g : model_.grad_b_out_) g *= scale;
            }

            // Optimizer step
            optimizer_.set_lr(get_lr(global_step));
            optimizer_.step();
            model_.sync_params_to_ssog();

            // Update counters
            tokens_processed += batch.batch_size * batch.seq_len;
            global_step++;
            epoch_step++;

            // Log every interval
            if (global_step % cfg_.log_interval == 1 || epoch_step == 1) {
                auto now = std::time(nullptr);
                Real elapsed = std::difftime(now, start_time);
                Real step_time = std::difftime(now, batch_start);
                Real tok_per_sec = tok_per_step / std::max(step_time, 1.0);
                Real eta = (total_batches - global_step) * step_time;
                Real lr_now = optimizer_.lr();
                Real epoch_progress = 100.0 * Real(epoch_step) / batches_per_epoch;

                log_metrics(global_step, metrics, {
                    {"lr", lr_now},
                    {"grad_norm", grad_norm},
                    {"epoch_progress", epoch_progress},
                    {"tok/s", tok_per_sec},
                    {"eta_s", eta},
                    {"elapsed_s", elapsed},
                    {"total", Real(total_batches)},
                    {"epoch_step", Real(epoch_step)},
                    {"epoch", Real(epoch + 1)}
                });
            }

            // Eval
            if (eval_loader_ && global_step % cfg_.eval_interval == 0) {
                TrainingMetrics eval_metrics = evaluate();
                auto now = std::time(nullptr);
                Real elapsed = std::difftime(now, start_time);
                std::ostringstream ss;
                ss << "  [Eval step=" << global_step << "]"
                   << " loss=" << std::fixed << std::setprecision(4) << eval_metrics.loss
                   << " ppl=" << std::setprecision(4) << eval_metrics.perplexity
                   << " acc=" << std::setprecision(2) << (eval_metrics.accuracy * 100) << "%"
                   << " elapsed=" << elapsed << "s";
                std::cout << ss.str() << std::endl;
                log(ss.str());
            }

            // Checkpoint
            if (global_step % cfg_.save_interval == 0) {
                std::string ckpt_path = cfg_.checkpoint_dir + "/" + cfg_.run_name
                                      + "_step_" + std::to_string(global_step) + ".bin";
                save_checkpoint(ckpt_path);
            }

            if (cfg_.max_steps > 0 && global_step >= cfg_.max_steps) break;
        }

        // Epoch summary
        TrainingMetrics epoch_metrics = metrics_.empty() ? TrainingMetrics{} : metrics_.back();
        auto now = std::time(nullptr);
        Real elapsed = std::difftime(now, start_time);
        std::ostringstream ss;
        ss << "  [Epoch " << (epoch + 1) << "] done."
           << " loss=" << std::fixed << std::setprecision(4) << epoch_metrics.loss
           << " ppl=" << std::setprecision(4) << epoch_metrics.perplexity
           << " acc=" << std::setprecision(2) << (epoch_metrics.accuracy * 100) << "%"
           << " elapsed=" << elapsed << "s";
        std::cout << ss.str() << std::endl;
        log(ss.str());

        if (cfg_.max_steps > 0 && global_step >= cfg_.max_steps) break;
    }

    auto end_time = std::time(nullptr);
    Real total_elapsed = std::difftime(end_time, start_time);
    Real tok_per_sec = tokens_processed / std::max(total_elapsed, 1.0);

    std::cout << "\n=== Training Complete ===" << "\n"
              << "  Total steps: " << global_step << "/" << total_batches << "\n"
              << "  Tokens processed: " << tokens_processed << "\n"
              << "  Time elapsed: " << total_elapsed << "s" << "\n"
              << "  Throughput: " << std::llround(tok_per_sec) << " tok/s\n"
              << std::endl;

    {
        std::ostringstream ss;
        ss << "=== Training Complete ==="
           << " steps=" << global_step << "/" << total_batches
           << " tokens=" << tokens_processed
           << " time=" << total_elapsed << "s"
           << " throughput=" << std::llround(tok_per_sec) << " tok/s";
        log(ss.str());
    }

    std::string final_path = cfg_.checkpoint_dir + "/" + cfg_.run_name + "_final.bin";
    save_checkpoint(final_path);
    std::cout << "  Final checkpoint: " << final_path << std::endl;
}

void Trainer::train_step(const Batch& batch) {
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

void Trainer::log_metrics(Index step, const TrainingMetrics& metrics,
                          const std::map<std::string, Real>& extra) {
    auto get = [&](const std::string& k, Real dflt = 0) {
        auto it = extra.find(k);
        return it != extra.end() ? it->second : dflt;
    };

    Real total = get("total", 1);
    Real epoch_step = get("epoch_step", 1);
    Real epoch_progress = get("epoch_progress", 0);
    Real lr_now = get("lr", 0);
    Real grad_norm = get("grad_norm", 0);
    Real tok_per_sec = get("tok/s", 0);
    Real eta_s = get("eta_s", 0);
    Real elapsed_s = get("elapsed_s", 0);
    Real epoch = get("epoch", 1);

    // Format ETA
    std::string eta_str;
    if (eta_s > 3600) {
        eta_str = std::to_string(int(eta_s / 3600)) + "h"
                + std::to_string(int(int(eta_s) % 3600 / 60)) + "m";
    } else if (eta_s > 60) {
        eta_str = std::to_string(int(eta_s / 60)) + "m"
                + std::to_string(int(eta_s) % 60) + "s";
    } else {
        eta_str = std::to_string(int(eta_s)) + "s";
    }

    std::cout << "  [Step " << step << "/" << int(total) << "]"
              << " ep=" << int(epoch) << "/" << cfg_.num_epochs
              << " " << std::fixed << std::setprecision(1) << epoch_progress << "%"
              << " loss=" << std::setprecision(4) << metrics.loss
              << " ppl=" << std::setprecision(4) << metrics.perplexity
              << " acc=" << std::setprecision(2) << (metrics.accuracy * 100) << "%"
              << " conf=" << std::setprecision(2) << metrics.sheaf_conflict
              << " lr=" << std::scientific << std::setprecision(2) << lr_now
              << " |g|=" << std::fixed << std::setprecision(2) << grad_norm
              << " " << std::llround(tok_per_sec) << "tok/s"
              << " eta=" << eta_str
              << std::endl;

    // Log to file (CSV format)
    if (log_stream_.is_open()) {
        log_stream_ << step << ","
                    << int(epoch) << ","
                    << epoch_progress << ","
                    << metrics.loss << ","
                    << metrics.perplexity << ","
                    << metrics.accuracy << ","
                    << metrics.sheaf_conflict << ","
                    << lr_now << ","
                    << grad_norm << ","
                    << tok_per_sec << ","
                    << elapsed_s
                    << std::endl;
    }
}

void Trainer::log(const std::string& msg) {
    if (log_stream_.is_open()) {
        log_stream_ << "# " << msg << std::endl;
    }
}

Real Trainer::get_lr(Index step) const {
    Index warmup = 100;
    Real lr = cfg_.lr;
    if (step < warmup) {
        lr = cfg_.lr * (Real(step) / warmup);
    } else {
        Index total = train_loader_.batches_per_epoch() * cfg_.num_epochs;
        if (cfg_.max_steps > 0) total = cfg_.max_steps;
        Real progress = Real(step - warmup) / (total - warmup + 1);
        if (total > warmup) {
            lr = cfg_.lr * 0.5 * (1 + std::cos(progress * PI));
        }
    }
    return std::max(lr, cfg_.lr * cfg_.lr_warmup);
}

void Trainer::save_checkpoint(const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (f.is_open()) {
        Index step = optimizer_.current_step();
        f.write(reinterpret_cast<const char*>(&step), sizeof(step));

        Index n = model_.param_W_out_.size();
        f.write(reinterpret_cast<const char*>(&n), sizeof(n));
        f.write(reinterpret_cast<const char*>(model_.param_W_out_.data()), n * sizeof(Real));
        n = model_.param_b_out_.size();
        f.write(reinterpret_cast<const char*>(&n), sizeof(n));
        f.write(reinterpret_cast<const char*>(model_.param_b_out_.data()), n * sizeof(Real));
        f.close();
        std::cout << "  [Checkpoint] saved " << path << std::endl;
    }
}

void Trainer::load_checkpoint(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (f.is_open()) {
        Index step;
        f.read(reinterpret_cast<char*>(&step), sizeof(step));
        optimizer_.set_step(step);

        Index n;
        f.read(reinterpret_cast<char*>(&n), sizeof(n));
        if (n == (Index)model_.param_W_out_.size()) {
            f.read(reinterpret_cast<char*>(model_.param_W_out_.data()), n * sizeof(Real));
        }
        f.read(reinterpret_cast<char*>(&n), sizeof(n));
        if (n == (Index)model_.param_b_out_.size()) {
            f.read(reinterpret_cast<char*>(model_.param_b_out_.data()), n * sizeof(Real));
        }
        model_.sync_params_to_ssog();
        f.close();
        std::cout << "  Loaded checkpoint from " << path << " (step " << step << ")" << std::endl;
    }
}

} // namespace ai2
