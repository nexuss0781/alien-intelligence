#include "tokenizer.hpp"
#include <fstream>
#include <sstream>
#include <set>
#include <algorithm>

namespace ai2 {

Tokenizer::Tokenizer() {
    // Reserve special tokens
    stoi_["<PAD>"] = pad_id_;
    stoi_["<BOS>"] = bos_id_;
    stoi_["<EOS>"] = eos_id_;
    stoi_["<UNK>"] = unk_id_;
    itos_[pad_id_] = "<PAD>";
    itos_[bos_id_] = "<BOS>";
    itos_[eos_id_] = "<EOS>";
    itos_[unk_id_] = "<UNK>";
}

void Tokenizer::build_from_chars(const std::string& text) {
    std::set<char> chars;
    for (char c : text) {
        chars.insert(c);
    }
    // Add newline and space explicitly
    chars.insert('\n');
    chars.insert(' ');

    for (char c : chars) {
        std::string s(1, c);
        stoi_[s] = next_id_;
        itos_[next_id_] = s;
        next_id_++;
    }
    vocab_size_ = next_id_;
}

void Tokenizer::train(const std::string& text, Index min_freq) {
    // Count character frequencies
    std::unordered_map<char, Index> freq;
    for (char c : text) freq[c]++;

    for (auto& [c, f] : freq) {
        if (f >= min_freq) {
            std::string s(1, c);
            if (stoi_.find(s) == stoi_.end()) {
                stoi_[s] = next_id_;
                itos_[next_id_] = s;
                next_id_++;
            }
        }
    }
    vocab_size_ = next_id_;
}

std::vector<Index> Tokenizer::encode(const std::string& text) const {
    std::vector<Index> tokens;
    tokens.push_back(bos_id_);
    for (char c : text) {
        std::string s(1, c);
        auto it = stoi_.find(s);
        if (it != stoi_.end()) {
            tokens.push_back(it->second);
        } else {
            tokens.push_back(unk_id_);
        }
    }
    tokens.push_back(eos_id_);
    return tokens;
}

std::string Tokenizer::decode(const std::vector<Index>& tokens) const {
    std::string result;
    for (Index id : tokens) {
        if (id == pad_id_ || id == bos_id_ || id == eos_id_) continue;
        auto it = itos_.find(id);
        if (it != itos_.end()) {
            result += it->second;
        }
    }
    return result;
}

void Tokenizer::save(const std::string& path) const {
    std::ofstream f(path);
    for (auto& [s, id] : stoi_) {
        f << id << " " << s << "\n";
    }
}

void Tokenizer::load(const std::string& path) {
    stoi_.clear();
    itos_.clear();
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        auto pos = line.find(' ');
        if (pos != std::string::npos) {
            Index id = std::stoul(line.substr(0, pos));
            std::string s = line.substr(pos + 1);
            stoi_[s] = id;
            itos_[id] = s;
        }
    }
    vocab_size_ = stoi_.size();
    next_id_ = vocab_size_;
}

} // namespace ai2
