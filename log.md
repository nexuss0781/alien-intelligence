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
[ 27%] Building CXX object CMakeFiles/ai2_train_lib.dir/src/tokenizer.cpp.o
[ 31%] Building CXX object CMakeFiles/test_slie.dir/tests/test_slie.cpp.o
[ 34%] Linking CXX executable test_slie
[ 34%] Built target test_slie
[ 37%] Building CXX object CMakeFiles/test_lssc.dir/tests/test_lssc.cpp.o
[ 41%] Building CXX object CMakeFiles/ai2_train_lib.dir/src/dataloader.cpp.o
[ 44%] Building CXX object CMakeFiles/ai2_train_lib.dir/src/model.cpp.o
[ 48%] Linking CXX executable test_lssc
[ 48%] Built target test_lssc
[ 51%] Building CXX object CMakeFiles/test_stre.dir/tests/test_stre.cpp.o
[ 55%] Linking CXX executable test_stre
[ 55%] Built target test_stre
[ 58%] Building CXX object CMakeFiles/ai2_train_lib.dir/src/optimizer.cpp.o
[ 62%] Building CXX object CMakeFiles/ai2_train_lib.dir/src/trainer.cpp.o
[ 65%] Building CXX object CMakeFiles/test_uq.dir/tests/test_uq.cpp.o
[ 68%] Linking CXX executable test_uq
[ 68%] Built target test_uq
[ 72%] Building CXX object CMakeFiles/test_ataa.dir/tests/test_ataa.cpp.o
[ 75%] Linking CXX static library libai2_train_lib.a
[ 75%] Built target ai2_train_lib
[ 79%] Building CXX object CMakeFiles/test_ssog.dir/tests/test_ssog.cpp.o
[ 82%] Linking CXX executable test_ataa
[ 82%] Built target test_ataa
[ 86%] Building CXX object CMakeFiles/test_math.dir/tests/test_math.cpp.o
[ 89%] Linking CXX executable test_ssog
[ 89%] Built target test_ssog
[ 93%] Building CXX object CMakeFiles/ai2_train.dir/src/main_train.cpp.o
[ 96%] Linking CXX executable ai2_train
[ 96%] Built target ai2_train
[100%] Linking CXX executable test_math
[100%] Built target test_math
=== Alien Intelligence (AI²) Comprehensive Mathematical Tests ===

[Types/Utilities Mathematical Tests]

[SLIE Mathematical Tests]
    [debug] token0_1 L2^2=0.0970324
    [debug] same-token L2 diff=0 max_diff[0]=0 a[i]=0.00813751 b[i]=0.00813751
    [debug] pos_diff L2^2=0.003907
    [debug] e_pos0[0..7]=0.0248757,-0.0077751,-0.00222176,0.00908489,-0.00269023,0.0119504,0.0620854,0.036686, e_pos1[0..7]=0.0246679,0.00313438,0.03809,0.0443968,-0.0176159,0.0328002,0.0519693,0.0241902,
    [debug] sketch_features size=16 expected=16

[LSSC Mathematical Tests]

[STRE Mathematical Tests]
    [debug] n_nodes=1 lap[0][0]=0 total=0

[UQ Mathematical Tests]

[ATAA Mathematical Tests]

[SSOG Mathematical Tests]

[Optimizer Mathematical Tests]

[Gradient Mathematical Tests]
    [debug] grad_norm=2.56843 grad_max=1.14419 param_W[0]=0.371773 param_b[0]=0
    [debug] BEFORE step: logits.size=1 seq[0].size=3 vocab=16 -3.05902 -2.61769 -3.38977 -2.51318 -2.84074 -3.22619 -2.39403 -3.27072 target=2
    [debug] AFTER  step: logits.size=1 seq[0].size=3 vocab=16 -3.08014 -2.63895 -3.34119 -2.41472 -2.71839 -3.24935 -2.41583 -3.29279 target=2
 target=2
    [debug] loss before=3.0184 after=2.96455 change=0.0538492 param_W[0]=0.37635 param_b[0]=-0.017206

[Pipeline Integration Test]
    [debug] step=1 loss=2.954995 |g|=13.3267 b_out[0]: 0.0000 -> -0.0100
    [debug] step=2 loss=2.945895 |g|=13.2730 b_out[0]: -0.0100 -> -0.0200
    [debug] step=3 loss=2.936833 |g|=13.2168 b_out[0]: -0.0200 -> -0.0300
    [debug] step=10 loss=2.877144 |g|=12.8191 b_out[0]: -0.0897 -> -0.0996
  Loss trajectory: 2.9550 2.9459 2.9368 2.9279 2.9192 2.9105 2.9020 2.8936 2.8853 2.8771 

=== Results ===
  Total: 277
  Passed: 277
  Failed: 0
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
test_slie: /content/alien-intelligence/include/types.hpp:66: ai2::Vec ai2::mat_vec(const Mat&, const Vec&): Assertion `m > 0 && A[0].size() == x.size()' failed.
run.txt: line 1:  3378 Aborted                 (core dumped) ./build/test_slie

