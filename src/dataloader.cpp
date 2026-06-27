#include "dataloader.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>

namespace ai2 {

DataLoader::DataLoader(const std::string& data_path, const Tokenizer& tokenizer,
                       Index batch_size, Index seq_len,
                       Index shuffle_buffer)
    : tokenizer_(&tokenizer), batch_size_(batch_size),
      seq_len_(seq_len), pos_(0), done_(false)
{
    load_data(data_path);
    (void)shuffle_buffer;  // reserved for future use
}

void DataLoader::load_data(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "Error: could not open " << path << std::endl;
        return;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    std::string text = ss.str();

    data_ = tokenizer_->encode(text);
    total_tokens_ = data_.size();

    // Compute batches per epoch
    Index tokens_per_batch = batch_size_ * seq_len_;
    batches_per_epoch_ = total_tokens_ / tokens_per_batch;
    if (batches_per_epoch_ == 0) batches_per_epoch_ = 1;

    std::cout << "  Loaded " << total_tokens_ << " tokens from " << path << std::endl;
    std::cout << "  Vocab size: " << tokenizer_->vocab_size() << std::endl;
    std::cout << "  Batches per epoch: " << batches_per_epoch_ << std::endl;
}

Batch DataLoader::next() {
    if (done_) return {};

    Index batch_size = batch_size_;
    Index seq_len = seq_len_;

    Mat tokens(batch_size, Vec(seq_len, tokenizer_->pad_id()));
    Mat targets(batch_size, Vec(seq_len, tokenizer_->pad_id()));

    for (Index b = 0; b < batch_size; ++b) {
        for (Index t = 0; t < seq_len; ++t) {
            if (pos_ + t >= data_.size()) {
                done_ = true;
                break;
            }
            tokens[b][t] = data_[pos_ + t];
            if (pos_ + t + 1 < data_.size()) {
                targets[b][t] = data_[pos_ + t + 1];
            } else {
                targets[b][t] = tokenizer_->eos_id();
                done_ = true;
            }
        }
        pos_ += seq_len;
        if (pos_ >= data_.size()) {
            done_ = true;
            break;
        }
    }

    return {tokens, targets, batch_size, seq_len};
}

void DataLoader::reset() {
    pos_ = 0;
    done_ = false;
}

} // namespace ai2
