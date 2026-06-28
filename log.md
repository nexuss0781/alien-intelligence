-- The CXX compiler identification is GNU 11.4.0
-- The CUDA compiler identification is NVIDIA 12.8.93 with host compiler GNU 11.4.0
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Detecting CUDA compiler ABI info
-- Detecting CUDA compiler ABI info - done
-- Check for working CUDA compiler: /usr/local/cuda/bin/nvcc - skipped
-- Detecting CUDA compile features
-- Detecting CUDA compile features - done
-- Configuring done (3.9s)
-- Generating done (0.0s)
-- Build files have been written to: /content/alien-intelligence/build
[  2%] Building CXX object CMakeFiles/ai2_core.dir/src/slie.cpp.o
[  5%] Building CUDA object CMakeFiles/ai2_gpu.dir/src/gpu_backend.cu.o
[  8%] Building CXX object CMakeFiles/ai2_core.dir/src/lssc.cpp.o
[ 11%] Building CXX object CMakeFiles/ai2_core.dir/src/stre.cpp.o
[ 14%] Linking CUDA static library libai2_gpu.a
[ 14%] Built target ai2_gpu
[ 17%] Building CXX object CMakeFiles/ai2_core.dir/src/uq.cpp.o
[ 20%] Building CXX object CMakeFiles/ai2_core.dir/src/ataa.cpp.o
[ 23%] Building CXX object CMakeFiles/ai2_core.dir/src/ssog.cpp.o
[ 26%] Linking CXX static library libai2_core.a
[ 26%] Built target ai2_core
[ 32%] Building CXX object CMakeFiles/ai2_train_lib.dir/src/tokenizer.cpp.o
[ 32%] Building CXX object CMakeFiles/test_slie.dir/tests/test_slie.cpp.o
[ 35%] Linking CXX executable test_slie
[ 35%] Built target test_slie
[ 38%] Building CXX object CMakeFiles/ai2_train_lib.dir/src/dataloader.cpp.o
[ 41%] Building CXX object CMakeFiles/ai2_train_lib.dir/src/model.cpp.o
[ 44%] Building CXX object CMakeFiles/ai2_train_lib.dir/src/optimizer.cpp.o
[ 47%] Building CXX object CMakeFiles/test_lssc.dir/tests/test_lssc.cpp.o
[ 50%] Building CXX object CMakeFiles/ai2_train_lib.dir/src/trainer.cpp.o
[ 52%] Linking CXX executable test_lssc
[ 52%] Built target test_lssc
[ 55%] Building CXX object CMakeFiles/ai2_train_lib.dir/src/hidden_cache.cpp.o
[ 58%] Building CXX object CMakeFiles/test_stre.dir/tests/test_stre.cpp.o
[ 61%] Linking CXX static library libai2_train_lib.a
[ 61%] Built target ai2_train_lib
[ 64%] Building CXX object CMakeFiles/test_uq.dir/tests/test_uq.cpp.o
[ 67%] Linking CXX executable test_stre
[ 67%] Built target test_stre
[ 70%] Building CXX object CMakeFiles/test_ataa.dir/tests/test_ataa.cpp.o
[ 73%] Linking CXX executable test_uq
[ 73%] Built target test_uq
[ 76%] Building CXX object CMakeFiles/test_ssog.dir/tests/test_ssog.cpp.o
[ 79%] Linking CXX executable test_ataa
[ 79%] Built target test_ataa
[ 82%] Building CXX object CMakeFiles/test_math.dir/tests/test_math.cpp.o
[ 85%] Linking CXX executable test_ssog
[ 85%] Built target test_ssog
[ 88%] Building CXX object CMakeFiles/ai2_train.dir/src/main_train.cpp.o
[ 91%] Linking CXX executable ai2_train
[ 91%] Built target ai2_train
[ 94%] Building CXX object CMakeFiles/ai2_infer.dir/src/infer_main.cpp.o
/content/alien-intelligence/src/infer_main.cpp: In lambda function:
/content/alien-intelligence/src/infer_main.cpp:143:39: error: could not convert ‘{tokens}’ from ‘<brace-enclosed initializer list>’ to ‘ai2::Mat’ {aka ‘std::vector<std::vector<double> >’}
  143 |             Mat batch_tokens = {tokens};
      |                                       ^
      |                                       |
      |                                       <brace-enclosed initializer list>
/content/alien-intelligence/src/infer_main.cpp:144:64: error: could not convert ‘{std::vector<long unsigned int>(seq_len, 0, std::allocator<long unsigned int>())}’ from ‘<brace-enclosed initializer list>’ to ‘ai2::Mat’ {aka ‘std::vector<std::vector<double> >’}
  144 |             Mat dummy_targets = {std::vector<Index>(seq_len, 0)};
      |                                                                ^
      |                                                                |
      |                                                                <brace-enclosed initializer list>
[ 97%] Linking CXX executable test_math
gmake[2]: *** [CMakeFiles/ai2_infer.dir/build.make:79: CMakeFiles/ai2_infer.dir/src/infer_main.cpp.o] Error 1
gmake[1]: *** [CMakeFiles/Makefile2:485: CMakeFiles/ai2_infer.dir/all] Error 2
gmake[1]: *** Waiting for unfinished jobs....
[ 97%] Built target test_math
gmake: *** [Makefile:101: all] Error 2

