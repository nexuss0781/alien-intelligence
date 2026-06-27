-- Configuring done (0.0s)
-- Generating done (0.0s)
-- Build files have been written to: /content/alien-intelligence/build
[  6%] Building CXX object CMakeFiles/ai2_core.dir/src/stre.cpp.o
[  6%] Building CXX object CMakeFiles/ai2_core.dir/src/ssog.cpp.o
[ 10%] Linking CXX static library libai2_core.a
[ 24%] Built target ai2_core
[ 27%] Linking CXX executable test_slie
[ 31%] Building CXX object CMakeFiles/ai2_train_lib.dir/src/model.cpp.o
[ 34%] Built target test_slie
[ 37%] Linking CXX executable test_lssc
[ 41%] Built target test_lssc
[ 44%] Building CXX object CMakeFiles/ai2_train_lib.dir/src/trainer.cpp.o
[ 48%] Building CXX object CMakeFiles/test_stre.dir/tests/test_stre.cpp.o
[ 51%] Linking CXX static library libai2_train_lib.a
[ 62%] Built target ai2_train_lib
[ 65%] Linking CXX executable test_uq
[ 68%] Built target test_uq
[ 72%] Linking CXX executable test_ataa
[ 75%] Built target test_ataa
[ 79%] Building CXX object CMakeFiles/test_ssog.dir/tests/test_ssog.cpp.o
/content/alien-intelligence/tests/test_stre.cpp: In function ‘void test_restriction_maps()’:
/content/alien-intelligence/tests/test_stre.cpp:64:10: error: ‘class ai2::STRE’ has no member named ‘init_restriction_maps’; did you mean ‘std::vector<std::vector<std::vector<std::vector<double> > > > ai2::STRE::restriction_maps_’? (not accessible from this context)
   64 |     stre.init_restriction_maps(42);
      |          ^~~~~~~~~~~~~~~~~~~~~
In file included from /content/alien-intelligence/tests/test_stre.cpp:1:
/content/alien-intelligence/include/stre.hpp:92:35: note: declared private here
   92 |     std::vector<std::vector<Mat>> restriction_maps_; // [node_idx][neighbor_idx] ∈ R^{d_node × d_node}
      |                                   ^~~~~~~~~~~~~~~~~
gmake[2]: *** [CMakeFiles/test_stre.dir/build.make:79: CMakeFiles/test_stre.dir/tests/test_stre.cpp.o] Error 1
gmake[1]: *** [CMakeFiles/Makefile2:242: CMakeFiles/test_stre.dir/all] Error 2
gmake[1]: *** Waiting for unfinished jobs....
[ 82%] Linking CXX executable test_ssog
[ 82%] Built target test_ssog
gmake: *** [Makefile:101: all] Error 2

