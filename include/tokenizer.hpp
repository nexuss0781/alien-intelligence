#pragma once
#include "types.hpp"
#include <string>
#include <unordered_map>
#include <vector>

namespace ai2 {

class Tokenizer {
public:
    Tokenizer();

    void train(const std::string& text, Index min_freq = 2);
    void build_from_chars(const std::string& text);

    std::vector<Index> encode(const std::string& text) const;
    std::string decode(const std::vector<Index>& tokens) const;

    Index vocab_size() const { return vocab_size_; }
    Index pad_id() const { return pad_id_; }
    Index bos_id() const { return bos_id_; }
    Index eos_id() const { return eos_id_; }
    Index unk_id() const { return unk_id_; }

    void save(const std::string& path) const;
    void load(const std::string& path);

private:
    std::unordered_map<std::string, Index> stoi_;
    std::unordered_map<Index, std::string> itos_;
    Index vocab_size_ = 0;
    Index pad_id_ = 0;
    Index bos_id_ = 1;
    Index eos_id_ = 2;
    Index unk_id_ = 3;
    Index next_id_ = 4;
};

} // namespace ai2
