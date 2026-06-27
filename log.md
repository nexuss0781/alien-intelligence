-- Configuring done (0.0s)
-- Generating done (0.0s)
-- Build files have been written to: /content/alien-intelligence/build
[ 24%] Built target ai2_core
[ 31%] Built target test_slie
[ 34%] Building CXX object CMakeFiles/ai2_train_lib.dir/src/dataloader.cpp.o
[ 41%] Built target test_lssc
[ 44%] Building CXX object CMakeFiles/ai2_train_lib.dir/src/trainer.cpp.o
/content/alien-intelligence/src/trainer.cpp: In member function ‘void ai2::Trainer::train()’:
/content/alien-intelligence/src/trainer.cpp:59:10: warning: unused variable ‘step_start’ [-Wunused-variable]
   59 |     auto step_start = start_time;
      |          ^~~~~~~~~~
/content/alien-intelligence/src/trainer.cpp: In member function ‘void ai2::Trainer::log_metrics(ai2::Index, const ai2::TrainingMetrics&, const std::map<std::__cxx11::basic_string<char>, double>&)’:
/content/alien-intelligence/src/trainer.cpp:235:10: warning: unused variable ‘epoch_step’ [-Wunused-variable]
  235 |     Real epoch_step = get("epoch_step", 1);
      |          ^~~~~~~~~~
[ 48%] Building CXX object CMakeFiles/test_stre.dir/tests/test_stre.cpp.o
[ 51%] Linking CXX static library libai2_train_lib.a
[ 62%] Built target ai2_train_lib
[ 68%] Built target test_uq
[ 75%] Built target test_ataa
[ 82%] Built target test_ssog
[ 86%] Building CXX object CMakeFiles/test_math.dir/tests/test_math.cpp.o
[ 89%] Linking CXX executable test_stre
[ 89%] Built target test_stre
[ 93%] Building CXX object CMakeFiles/ai2_train.dir/src/main_train.cpp.o
[ 96%] Linking CXX executable ai2_train
[ 96%] Built target ai2_train
[100%] Linking CXX executable test_math
[100%] Built target test_math
=== Alien Intelligence (AI²) Comprehensive Mathematical Tests ===

[Types/Utilities Mathematical Tests]
  FAIL: Sigmoid should be in (0,1) for v=100.000000

[SLIE Mathematical Tests]
  FAIL: Same token should give same embedding got 0.0638242 expected 0 (diff=0.0638242)
  FAIL: Sketch features should be sketch_width * sketch_depth

[LSSC Mathematical Tests]

[STRE Mathematical Tests]
  FAIL: Sheaf Laplacian of constant vector should be near zero got 1 expected 0 (diff=1)

[UQ Mathematical Tests]

[ATAA Mathematical Tests]

[SSOG Mathematical Tests]

[Optimizer Mathematical Tests]

[Gradient Mathematical Tests]
  FAIL: Loss should change after SGD step, was 2.772589 now 2.772589

[Pipeline Integration Test]
  Loss trajectory: 2.7726 2.7726 2.7726 2.7726 2.7726 2.7726 2.7726 2.7726 2.7726 2.7726 

=== Results ===
  Total: 259
  Passed: 254
  Failed: 5

