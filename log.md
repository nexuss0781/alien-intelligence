=== Alien Intelligence (AI²) Training ===
[1/3] Downloading datasets (if needed)...
  Data already exists, skipping download
[2/3] Building...
-- Found OpenMP_CXX: -fopenmp (found version "4.5")
-- Found OpenMP: TRUE (found version "4.5")
-- Configuring done (0.2s)
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
[ 60%] Building CXX object CMakeFiles/ai2_train_lib.dir/src/dataloader.cpp.o
[ 60%] Building CXX object CMakeFiles/ai2_train_lib.dir/src/tokenizer.cpp.o
[ 66%] Building CXX object CMakeFiles/ai2_train_lib.dir/src/model.cpp.o
[ 73%] Building CXX object CMakeFiles/ai2_train_lib.dir/src/optimizer.cpp.o
[ 80%] Building CXX object CMakeFiles/ai2_train_lib.dir/src/trainer.cpp.o
[ 86%] Linking CXX static library libai2_train_lib.a
[ 86%] Built target ai2_train_lib
[ 93%] Building CXX object CMakeFiles/ai2_train.dir/src/main_train.cpp.o
[100%] Linking CXX executable ai2_train
[100%] Built target ai2_train
[3/3] Starting training (OMP_NUM_THREADS=2) ...
OpenMP enabled: 2 threads
=== Alien Intelligence (AI²) Training ===

--- Stage 1: Pretraining ---
Tokenizer vocab size: 132
  Loaded 1303653 tokens from data/pretrain.txt
  Vocab size: 132
  Batches per epoch: 159
  Loaded 1303653 tokens from data/pretrain.txt
  Vocab size: 132
  Batches per epoch: 318
Model created.
  d_model=256 d_state=128 vocab=132 experts=32

=== Training Started ===
  Run: ai2_pretrain
  Epochs: 3
  Batch size: 64  Seq len: 128
  Tokens/step: 8192
  Steps/epoch: 159
  Total steps: 477
  Total tokens: 3907584
  Learning rate: 0.001
  Optimizer params: 33924
  Checkpoint dir: checkpoints

--- Epoch 1/3 ---
  [Step 1/477] ep=1/3 0.6% loss=4.8831 ppl=132.0353 acc=0.46% conf=1.38 lr=1.00e-04 |g|=2289.01 1365tok/s eta=47m36s

