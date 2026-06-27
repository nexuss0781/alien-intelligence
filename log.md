=== Alien Intelligence (AI²) Training ===
[1/3] Downloading datasets (if needed)...
  Data already exists, skipping download
[2/3] Building with CUDA...
-- Configuring done (0.0s)
-- Generating done (0.0s)
-- Build files have been written to: /content/alien-intelligence/build
[ 11%] Built target ai2_gpu
[ 50%] Built target ai2_core
[ 55%] Building CXX object CMakeFiles/ai2_train_lib.dir/src/trainer.cpp.o
[ 61%] Building CXX object CMakeFiles/ai2_train_lib.dir/src/hidden_cache.cpp.o
/content/alien-intelligence/src/trainer.cpp: In member function ‘ai2::TrainingMetrics ai2::Trainer::train_with_cache()’:
/content/alien-intelligence/src/trainer.cpp:134:5: error: ‘Log’ was not declared in this scope; did you mean ‘log’?
  134 |     Log("=== Training (GPU + Cache) Started ===");
      |     ^~~
      |     log
gmake[3]: *** [CMakeFiles/ai2_train_lib.dir/build.make:135: CMakeFiles/ai2_train_lib.dir/src/trainer.cpp.o] Error 1
gmake[3]: *** Waiting for unfinished jobs....
gmake[2]: *** [CMakeFiles/Makefile2:182: CMakeFiles/ai2_train_lib.dir/all] Error 2
gmake[1]: *** [CMakeFiles/Makefile2:455: CMakeFiles/ai2_train.dir/rule] Error 2
gmake: *** [Makefile:264: ai2_train] Error 2
[3/3] Starting training (GPU) ...
  Found checkpoint: checkpoints/ai2_pretrain_step_100.bin
  Cache file: checkpoints/*_cache.bin (loaded automatically)
  Resuming...
=== Alien Intelligence (AI²) Training ===

--- Stage 1: Pretraining ---
Tokenizer vocab size: 132
  Loaded 1303653 tokens from data/pretrain.txt
  Vocab size: 132
  Batches per epoch: 159
  Loaded 1303653 tokens from data/pretrain.txt
  Vocab size: 132
  Batches per epoch: 318
  GPU: Tesla T4  14912MB  SMs: 40
  GPU acceleration enabled (full forward+backward)
Model created.
  d_model=256 d_state=128 vocab=132 experts=32
Resuming from checkpoint: checkpoints/ai2_pretrain_step_100.bin
  Loaded checkpoint from checkpoints/ai2_pretrain_step_100.bin (step 100)

=== Building Hidden State Cache ===
  Pre-computing hidden states for 159 batches (1302528 positions, 1272MB)...
    Batch 1/159  cache: ~8MB so far
^C

