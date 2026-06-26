#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="data"
mkdir -p "$DATA_DIR"

echo "=== Downloading pretraining dataset ==="

# Tiny Shakespeare (small, character-level, ~1MB)
if [ ! -f "$DATA_DIR/tiny_shakespeare.txt" ]; then
    echo "Downloading Tiny Shakespeare..."
    wget -q -O "$DATA_DIR/tiny_shakespeare.txt" \
        https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    echo "  Done (pretraining)"
else
    echo "  Pretraining data already exists, skipping"
fi

# Alpaca-style instruction finetuning dataset (small sample)
if [ ! -f "$DATA_DIR/alpaca_cleaned.txt" ]; then
    echo "Downloading finetuning dataset (Alpaca sample)..."
    python3 -c "
import json, urllib.request, random

# Download a small sample of instruction data
url = 'https://raw.githubusercontent.com/tloen/alpaca-lora/main/alpaca_data_cleaned_archive.json'
try:
    with urllib.request.urlopen(url) as f:
        data = json.load(f)
    random.seed(42)
    sample = random.sample(data, min(500, len(data)))
    lines = []
    for item in sample:
        lines.append('### Instruction: ' + item.get('instruction', ''))
        if item.get('input', '').strip():
            lines.append('### Input: ' + item['input'])
        lines.append('### Response: ' + item.get('output', ''))
        lines.append('')
    with open('$DATA_DIR/alpaca_cleaned.txt', 'w') as f:
        f.write('\n'.join(lines))
    print(f'  Downloaded {len(sample)} instruction examples')
except Exception as e:
    print(f'  Skipping Alpaca download ({e})')
    # Fallback: create a simple instruction-format text
    with open('$DATA_DIR/alpaca_cleaned.txt', 'w') as f:
        f.write('### Instruction: What is AI?\n### Response: Artificial Intelligence is the simulation of human intelligence by machines.\n\n')
        f.write('### Instruction: Explain machine learning.\n### Response: Machine learning is a subset of AI where systems learn from data.\n\n')
    print('  Created minimal finetuning sample')
"
else
    echo "  Finetuning data already exists, skipping"
fi

# Combine into a unified corpus (for pretraining)
if [ ! -f "$DATA_DIR/pretrain.txt" ]; then
    echo "Creating unified pretraining corpus..."
    cat "$DATA_DIR/tiny_shakespeare.txt" > "$DATA_DIR/pretrain.txt"
    if [ -f "$DATA_DIR/alpaca_cleaned.txt" ]; then
        echo "" >> "$DATA_DIR/pretrain.txt"
        cat "$DATA_DIR/alpaca_cleaned.txt" >> "$DATA_DIR/pretrain.txt"
    fi
    echo "  Done"
fi

echo ""
echo "=== Dataset summary ==="
for f in "$DATA_DIR"/*.txt; do
    lines=$(wc -l < "$f")
    chars=$(wc -c < "$f")
    echo "  $(basename "$f"): ${lines} lines, ${chars} chars"
done
echo "=== All datasets ready ==="
