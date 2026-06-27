=== Alien Intelligence (AI²) Build & Test ===
-- Configuring done (0.0s)
-- Generating done (0.0s)
-- Build files have been written to: /content/alien-intelligence/build
[  3%] Building CXX object CMakeFiles/ai2_core.dir/src/slie.cpp.o
[  7%] Building CXX object CMakeFiles/ai2_train_lib.dir/src/tokenizer.cpp.o
[ 11%] Building CXX object CMakeFiles/ai2_train_lib.dir/src/dataloader.cpp.o
[ 14%] Building CXX object CMakeFiles/ai2_core.dir/src/lssc.cpp.o
/content/alien-intelligence/src/dataloader.cpp: In member function ‘ai2::Batch ai2::DataLoader::next()’:
/content/alien-intelligence/src/dataloader.cpp:48:41: error: ‘pad_id’ was not declared in this scope
   48 |     Mat tokens(batch_size, Vec(seq_len, pad_id()));
      |                                         ^~~~~~
/content/alien-intelligence/src/dataloader.cpp:61:33: error: ‘eos_id’ was not declared in this scope
   61 |                 targets[b][t] = eos_id();
      |                                 ^~~~~~
gmake[2]: *** [CMakeFiles/ai2_train_lib.dir/build.make:93: CMakeFiles/ai2_train_lib.dir/src/dataloader.cpp.o] Error 1
gmake[1]: *** [CMakeFiles/Makefile2:143: CMakeFiles/ai2_train_lib.dir/all] Error 2
gmake[1]: *** Waiting for unfinished jobs....
[ 18%] Building CXX object CMakeFiles/ai2_core.dir/src/stre.cpp.o
[ 22%] Building CXX object CMakeFiles/ai2_core.dir/src/uq.cpp.o
[ 25%] Building CXX object CMakeFiles/ai2_core.dir/src/ataa.cpp.o
[ 29%] Building CXX object CMakeFiles/ai2_core.dir/src/ssog.cpp.o
[ 33%] Linking CXX static library libai2_core.a
[ 33%] Built target ai2_core
gmake: *** [Makefile:101: all] Error 2

=== Running Tests ===
=== All components built and tested successfully ===

