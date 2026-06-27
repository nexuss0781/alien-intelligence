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
[ 41%] Building CXX object CMakeFiles/test_lssc.dir/tests/test_lssc.cpp.o
[ 44%] Building CXX object CMakeFiles/ai2_train_lib.dir/src/model.cpp.o
[ 48%] Linking CXX executable test_lssc
[ 48%] Built target test_lssc
[ 51%] Building CXX object CMakeFiles/ai2_train_lib.dir/src/optimizer.cpp.o
[ 55%] Building CXX object CMakeFiles/test_stre.dir/tests/test_stre.cpp.o
[ 58%] Building CXX object CMakeFiles/ai2_train_lib.dir/src/trainer.cpp.o
[ 62%] Linking CXX executable test_stre
[ 62%] Built target test_stre
[ 65%] Building CXX object CMakeFiles/test_uq.dir/tests/test_uq.cpp.o
[ 68%] Linking CXX static library libai2_train_lib.a
[ 68%] Built target ai2_train_lib
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
/content/alien-intelligence/tests/test_math.cpp:734:33: warning: unused variable ‘g_min’ [-Wunused-variable]
  734 |     Real g_norm = 0, g_max = 0, g_min = 0;
      |                                 ^~~~~
[ 96%] Linking CXX executable ai2_train
[ 96%] Built target ai2_train
[100%] Linking CXX executable test_math
[100%] Built target test_math
=== Alien Intelligence (AI²) Comprehensive Mathematical Tests ===

[Types/Utilities Mathematical Tests]

[SLIE Mathematical Tests]
  FAIL: Different token IDs should yield different embeddings
    [debug] same-token L2 diff=0 max_diff[0]=0 a[i]=0.00813751 b[i]=0.00813751
  FAIL: Positional encoding: same token at different positions should differ (or positional component is additive)
    [debug] sketch_features size=16 expected=16

[LSSC Mathematical Tests]

[STRE Mathematical Tests]
    [debug] n_nodes=1 n_edges=0
  FAIL: Sheaf Laplacian of constant vector should be near zero, got 1.000000 got 1 expected 0 (diff=1)

[UQ Mathematical Tests]

[ATAA Mathematical Tests]

[SSOG Mathematical Tests]

[Optimizer Mathematical Tests]

[Gradient Mathematical Tests]
    [debug] grad_norm=2.52491 grad_max=1.07206 param_W[0]=0.371773 param_b[0]=0
    [debug] loss before=2.77259 after=2.77259 change=0 param_W[0]=0.377146 param_b[0]=-0.01875
  FAIL: Loss should change after SGD step, was 2.772589 now 2.772589 (grad_norm=2.524915)

[Pipeline Integration Test]
    [debug] step=1 loss=2.772589 |g|=11.8163 b_out[0]: 0.0000 -> -0.0100
    [debug] step=2 loss=2.772589 |g|=11.8163 b_out[0]: -0.0100 -> -0.0200
    [debug] step=3 loss=2.772589 |g|=11.8163 b_out[0]: -0.0200 -> -0.0300
    [debug] step=10 loss=2.772589 |g|=11.8163 b_out[0]: -0.0900 -> -0.1000
  FAIL: Loss should change during training. All values = 2.772589
  Loss trajectory: 2.7726 2.7726 2.7726 2.7726 2.7726 2.7726 2.7726 2.7726 2.7726 2.7726 

=== Results ===
  Total: 276
  Passed: 271
  Failed: 5

