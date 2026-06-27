#pragma once
#include "types.hpp"
#include "tokenizer.hpp"
#include <string>
#include <vector>
#include <deque>

namespace ai2 {

struct Batch {
    Mat tokens;       // [batch_size x seq_len] input token IDs
    Mat targets;      // [batch_size x seq_len] target token IDs (next token)
    Index batch_size;
    Index seq_len;
};

class DataLoader {
public:
    DataLoader(const std::string& data_path, const Tokenizer& tokenizer,
               Index batch_size = 8, Index seq_len = 128,
               Index shuffle_buffer = 10000);

    // Get next batch
    Batch next();

    // Check if epoch is done
    bool epoch_done() const { return done_; }

    // Reset for new epoch
    void reset();

    // Total number of tokens in dataset
    Index total_tokens() const { return total_tokens_; }

    // Number of batches per epoch
    Index batches_per_epoch() const { return batches_per_epoch_; }

    Index batch_size() const { return batch_size_; }
    Index seq_len() const { return seq_len_; }

private:
    const Tokenizer* tokenizer_;
    Index batch_size_;
    Index seq_len_;

    std::vector<Index> data_;         // tokenized corpus
    Index pos_ = 0;                   // current position in data
    Index total_tokens_ = 0;
    Index batches_per_epoch_ = 0;
    bool done_ = false;

    void load_data(const std::string& path);
};

} // namespace ai2
