=== Alien Intelligence (AI²) Training ===
[1/3] Downloading datasets (if needed)...
  Data already exists, skipping download
[2/3] Building...
-- Configuring done (0.0s)
-- Generating done (0.0s)
-- Build files have been written to: /content/alien-intelligence/build
[  6%] Building CXX object CMakeFiles/ai2_core.dir/src/slie.cpp.o
[ 13%] Building CXX object CMakeFiles/ai2_core.dir/src/lssc.cpp.o
[ 20%] Building CXX object CMakeFiles/ai2_core.dir/src/stre.cpp.o
[ 26%] Building CXX object CMakeFiles/ai2_core.dir/src/uq.cpp.o
[ 33%] Building CXX object CMakeFiles/ai2_core.dir/src/ataa.cpp.o
[ 40%] Building CXX object CMakeFiles/ai2_core.dir/src/ssog.cpp.o
[ 46%] Linking CXX static library libai2_core.a
[ 46%] Built target ai2_core
[ 53%] Building CXX object CMakeFiles/ai2_train_lib.dir/src/tokenizer.cpp.o
[ 60%] Building CXX object CMakeFiles/ai2_train_lib.dir/src/dataloader.cpp.o
[ 66%] Building CXX object CMakeFiles/ai2_train_lib.dir/src/model.cpp.o
[ 73%] Building CXX object CMakeFiles/ai2_train_lib.dir/src/optimizer.cpp.o
[ 80%] Building CXX object CMakeFiles/ai2_train_lib.dir/src/trainer.cpp.o
[ 86%] Linking CXX static library libai2_train_lib.a
[ 86%] Built target ai2_train_lib
[ 93%] Building CXX object CMakeFiles/ai2_train.dir/src/main_train.cpp.o
[100%] Linking CXX executable ai2_train
[100%] Built target ai2_train
[3/3] Starting training (3 epochs ~ 7638 steps) ...
=== Alien Intelligence (AI²) Training ===

--- Stage 1: Pretraining ---
Tokenizer vocab size: 132
  Loaded 1303653 tokens from data/pretrain.txt
  Vocab size: 132
  Batches per epoch: 2546
  Loaded 1303653 tokens from data/pretrain.txt
  Vocab size: 132
  Batches per epoch: 5092
Model created.
  d_model=64 d_state=32 vocab=132 experts=16

=== Training Started ===
  Run: ai2_pretrain
  Epochs: 3
  Batch size: 8  Seq len: 64
  Tokens/step: 512
  Steps/epoch: 2546
  Total steps: 7638
  Total tokens: 3910656
  Learning rate: 0.001
  Optimizer params: 8580
  Checkpoint dir: checkpoints

--- Epoch 1/3 ---
  [Step 1/7638] ep=1/3 0.0% loss=4.8766 ppl=131.1785 acc=1.37% conf=2.14 lr=1.00e-04 |g|=237.96 512tok/s eta=0s
  [Step 6/7638] ep=1/3 0.2% loss=4.8873 ppl=132.5996 acc=2.34% conf=3.49 lr=1.00e-04 |g|=255.38 512tok/s eta=0s
  [Step 11/7638] ep=1/3 0.4% loss=4.8708 ppl=130.4227 acc=1.56% conf=1.55 lr=1.00e-04 |g|=233.30 512tok/s eta=0s
  [Step 16/7638] ep=1/3 0.6% loss=4.8810 ppl=131.7662 acc=2.15% conf=2.09 lr=1.50e-04 |g|=277.58 512tok/s eta=0s
  [Eval step=20] loss=4.8748 ppl=130.9423 acc=1.94% elapsed=104.00s
  [Step 21/7638] ep=1/3 0.8% loss=4.8802 ppl=131.6606 acc=1.37% conf=2.33 lr=2.00e-04 |g|=250.89 512tok/s eta=0s
  [Step 26/7638] ep=1/3 1.0% loss=4.8711 ppl=130.4696 acc=1.76% conf=2.11 lr=2.50e-04 |g|=273.41 512tok/s eta=0s
  [Step 31/7638] ep=1/3 1.2% loss=4.8640 ppl=129.5392 acc=2.54% conf=2.10 lr=3.00e-04 |g|=256.06 512tok/s eta=0s
  [Step 36/7638] ep=1/3 1.4% loss=4.8702 ppl=130.3493 acc=2.54% conf=2.77 lr=3.50e-04 |g|=226.57 512tok/s eta=0s
  [Eval step=40] loss=4.8573 ppl=128.6707 acc=3.63% elapsed=206.00s
  [Step 41/7638] ep=1/3 1.6% loss=4.8652 ppl=129.6943 acc=3.32% conf=2.95 lr=4.00e-04 |g|=241.98 512tok/s eta=0s
  [Step 46/7638] ep=1/3 1.8% loss=4.8493 ppl=127.6507 acc=4.88% conf=2.66 lr=4.50e-04 |g|=270.50 512tok/s eta=0s
  [Step 51/7638] ep=1/3 2.0% loss=4.8348 ppl=125.8126 acc=5.66% conf=1.79 lr=5.00e-04 |g|=247.35 512tok/s eta=0s
  [Step 56/7638] ep=1/3 2.2% loss=4.8401 ppl=126.4861 acc=4.49% conf=2.47 lr=5.50e-04 |g|=237.73 512tok/s eta=0s

