-- The CXX compiler identification is GNU 11.4.0
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Configuring done (0.2s)
-- Generating done (0.0s)
-- Build files have been written to: /content/alien-intelligence/build
[  3%] Building CXX object CMakeFiles/ai2_core.dir/src/slie.cpp.o
[  6%] Building CXX object CMakeFiles/ai2_core.dir/src/lssc.cpp.o
[ 10%] Building CXX object CMakeFiles/ai2_core.dir/src/stre.cpp.o
[ 13%] Building CXX object CMakeFiles/ai2_core.dir/src/uq.cpp.o
[ 17%] Building CXX object CMakeFiles/ai2_core.dir/src/ataa.cpp.o
[ 20%] Building CXX object CMakeFiles/ai2_core.dir/src/ssog.cpp.o
[ 24%] Linking CXX static library libai2_core.a
[ 24%] Built target ai2_core
[ 27%] Building CXX object CMakeFiles/test_slie.dir/tests/test_slie.cpp.o
[ 31%] Building CXX object CMakeFiles/ai2_train_lib.dir/src/tokenizer.cpp.o
[ 34%] Linking CXX executable test_slie
[ 34%] Built target test_slie
[ 37%] Building CXX object CMakeFiles/ai2_train_lib.dir/src/dataloader.cpp.o
[ 41%] Building CXX object CMakeFiles/ai2_train_lib.dir/src/model.cpp.o
[ 44%] Building CXX object CMakeFiles/ai2_train_lib.dir/src/optimizer.cpp.o
[ 48%] Building CXX object CMakeFiles/ai2_train_lib.dir/src/trainer.cpp.o
[ 51%] Building CXX object CMakeFiles/test_lssc.dir/tests/test_lssc.cpp.o
[ 55%] Linking CXX executable test_lssc
[ 55%] Built target test_lssc
[ 58%] Building CXX object CMakeFiles/test_stre.dir/tests/test_stre.cpp.o
[ 62%] Linking CXX static library libai2_train_lib.a
[ 62%] Built target ai2_train_lib
[ 65%] Building CXX object CMakeFiles/test_uq.dir/tests/test_uq.cpp.o
[ 68%] Linking CXX executable test_stre
[ 68%] Built target test_stre
[ 72%] Building CXX object CMakeFiles/test_ataa.dir/tests/test_ataa.cpp.o
[ 75%] Linking CXX executable test_uq
[ 75%] Built target test_uq
[ 79%] Building CXX object CMakeFiles/test_ssog.dir/tests/test_ssog.cpp.o
[ 82%] Linking CXX executable test_ataa
[ 82%] Built target test_ataa
[ 86%] Building CXX object CMakeFiles/test_math.dir/tests/test_math.cpp.o
[ 89%] Linking CXX executable test_ssog
[ 89%] Built target test_ssog
[ 93%] Building CXX object CMakeFiles/ai2_train.dir/src/main_train.cpp.o
/content/alien-intelligence/tests/test_math.cpp: In function ‘void ai2::test::test_gradient_math()’:
/content/alien-intelligence/tests/test_math.cpp:773:64: error: ‘std::vector<std::vector<std::vector<double> > > ai2::Model::logits_’ is private within this context
  773 |     for (Index i = 0; i < 8 && i < 16; ++i) std::cout << model.logits_[0][0][i] << ",";
      |                                                                ^~~~~~~
In file included from /content/alien-intelligence/tests/test_math.cpp:8:
/content/alien-intelligence/include/model.hpp:100:35: note: declared private here
  100 |     std::vector<std::vector<Vec>> logits_;    // [batch x seq_len x vocab]
      |                                   ^~~~~~~
/content/alien-intelligence/tests/test_math.cpp:784:64: error: ‘std::vector<std::vector<std::vector<double> > > ai2::Model::logits_’ is private within this context
  784 |     for (Index i = 0; i < 8 && i < 16; ++i) std::cout << model.logits_[0][0][i] << ",";
      |                                                                ^~~~~~~
In file included from /content/alien-intelligence/tests/test_math.cpp:8:
/content/alien-intelligence/include/model.hpp:100:35: note: declared private here
  100 |     std::vector<std::vector<Vec>> logits_;    // [batch x seq_len x vocab]
      |                                   ^~~~~~~
gmake[2]: *** [CMakeFiles/test_math.dir/build.make:79: CMakeFiles/test_math.dir/tests/test_math.cpp.o] Error 1
gmake[1]: *** [CMakeFiles/Makefile2:371: CMakeFiles/test_math.dir/all] Error 2
gmake[1]: *** Waiting for unfinished jobs....
[ 96%] Linking CXX executable ai2_train
[ 96%] Built target ai2_train
gmake: *** [Makefile:101: all] Error 2

