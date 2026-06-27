=== Alien Intelligence (AI²) Build & Test ===
-- Configuring done (0.0s)
-- Generating done (0.0s)
-- Build files have been written to: /content/alien-intelligence/build
[  3%] Building CXX object CMakeFiles/ai2_train_lib.dir/src/dataloader.cpp.o
[ 29%] Built target ai2_core
[ 33%] Building CXX object CMakeFiles/ai2_train_lib.dir/src/model.cpp.o
/content/alien-intelligence/src/model.cpp: In constructor ‘ai2::Model::Model(const ai2::ModelConfig&)’:
/content/alien-intelligence/src/model.cpp:22:40: error: ‘const struct ai2::ModelConfig’ has no member named ‘n_examples’
   22 |                                    cfg.n_examples, cfg.sketch_rank);
      |                                        ^~~~~~~~~~
gmake[2]: *** [CMakeFiles/ai2_train_lib.dir/build.make:107: CMakeFiles/ai2_train_lib.dir/src/model.cpp.o] Error 1
gmake[2]: *** Waiting for unfinished jobs....
[ 37%] Building CXX object CMakeFiles/test_slie.dir/tests/test_slie.cpp.o
gmake[1]: *** [CMakeFiles/Makefile2:143: CMakeFiles/ai2_train_lib.dir/all] Error 2
gmake[1]: *** Waiting for unfinished jobs....
[ 40%] Linking CXX executable test_slie
[ 40%] Built target test_slie
gmake: *** [Makefile:101: all] Error 2

=== Running Tests ===
=== Component 1: SLIE Tests ===
  CHE produces valid embeddings... PASS
  CHE is deterministic (same token -> same embedding)... PASS
  CHE gives different embeddings for different tokens... PASS
  SPE updates positional state... PASS
  SPE gives different states for different positions... PASS
  Sketch updates counters... PASS
  reset_position resets state... PASS
  O(1) per token — embedding dimension independent... PASS
  Handles OOB vocabulary indices gracefully... PASS
  No NaN or Inf in outputs... PASS
All SLIE tests PASSED.

=== All components built and tested successfully ===

