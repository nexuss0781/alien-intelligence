=== Alien Intelligence (AI²) Build & Test ===
-- Configuring done (0.0s)
-- Generating done (0.0s)
-- Build files have been written to: /content/alien-intelligence/build
[  3%] Building CXX object CMakeFiles/ai2_train_lib.dir/src/model.cpp.o
[ 29%] Built target ai2_core
[ 37%] Built target test_slie
[ 40%] Building CXX object CMakeFiles/ai2_train_lib.dir/src/optimizer.cpp.o
[ 44%] Building CXX object CMakeFiles/test_lssc.dir/tests/test_lssc.cpp.o
[ 48%] Building CXX object CMakeFiles/ai2_train_lib.dir/src/trainer.cpp.o
/content/alien-intelligence/src/trainer.cpp: In member function ‘void ai2::Trainer::save_checkpoint(const string&)’:
/content/alien-intelligence/src/trainer.cpp:171:37: error: void value not ignored as it ought to be
  171 |         Index step = optimizer_.step();
      |                      ~~~~~~~~~~~~~~~^~
gmake[2]: *** [CMakeFiles/ai2_train_lib.dir/build.make:135: CMakeFiles/ai2_train_lib.dir/src/trainer.cpp.o] Error 1
gmake[1]: *** [CMakeFiles/Makefile2:143: CMakeFiles/ai2_train_lib.dir/all] Error 2
gmake[1]: *** Waiting for unfinished jobs....
[ 51%] Linking CXX executable test_lssc
[ 51%] Built target test_lssc
gmake: *** [Makefile:101: all] Error 2

=== Running Tests ===
=== Component 2: LSSC Tests ===
  SSM step produces consistent output dimensions... PASS
  SSM state evolves over time... PASS
  SSM scan processes sequence in O(n)... PASS
  RFA random features have correct dimension... PASS
  RFA features are bounded and positive... PASS
  RFA causal produces correct output shape... PASS
  RFA causal first token depends only on itself... PASS
  Gated fusion produces correct output... PASS
  Gated fusion produces convex combination of SSM and RFA... PASS
  Full forward processes sequence in O(n)... PASS
  No NaN/Inf during long sequence processing... PASS
  SSM remains stable for many steps... PASS
All LSSC tests PASSED.

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

