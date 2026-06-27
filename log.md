=== Alien Intelligence (AI²) Training ===
[1/3] Downloading datasets (if needed)...
  Data already exists, skipping download
[2/3] Building with CUDA...
-- Configuring done (0.0s)
-- Generating done (0.0s)
-- Build files have been written to: /content/alien-intelligence/build
[  5%] Building CUDA object CMakeFiles/ai2_gpu.dir/src/gpu_backend.cu.o
nvcc warning : Support for offline compilation for architectures prior to '<compute/sm/lto>_75' will be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).
[ 11%] Building CXX object CMakeFiles/ai2_core.dir/src/slie.cpp.o
In file included from /content/alien-intelligence/include/slie.hpp:2,
                 from /content/alien-intelligence/src/slie.cpp:1:
/content/alien-intelligence/include/types.hpp:38: warning: ignoring ‘#pragma omp parallel’ [-Wunknown-pragmas]
   38 |     #pragma omp parallel for reduction(+:s)
      | 
/content/alien-intelligence/include/types.hpp:72: warning: ignoring ‘#pragma omp parallel’ [-Wunknown-pragmas]
   72 |     #pragma omp parallel for
      | 
/content/alien-intelligence/include/types.hpp:83: warning: ignoring ‘#pragma omp parallel’ [-Wunknown-pragmas]
   83 |     #pragma omp parallel for
      | 
[ 16%] Building CXX object CMakeFiles/ai2_core.dir/src/lssc.cpp.o
In file included from /content/alien-intelligence/include/lssc.hpp:2,
                 from /content/alien-intelligence/src/lssc.cpp:1:
/content/alien-intelligence/include/types.hpp:38: warning: ignoring ‘#pragma omp parallel’ [-Wunknown-pragmas]
   38 |     #pragma omp parallel for reduction(+:s)
      | 
/content/alien-intelligence/include/types.hpp:72: warning: ignoring ‘#pragma omp parallel’ [-Wunknown-pragmas]
   72 |     #pragma omp parallel for
      | 
/content/alien-intelligence/include/types.hpp:83: warning: ignoring ‘#pragma omp parallel’ [-Wunknown-pragmas]
   83 |     #pragma omp parallel for
      | 
[ 22%] Building CXX object CMakeFiles/ai2_core.dir/src/stre.cpp.o
In file included from /content/alien-intelligence/include/stre.hpp:2,
                 from /content/alien-intelligence/src/stre.cpp:1:
/content/alien-intelligence/include/types.hpp:38: warning: ignoring ‘#pragma omp parallel’ [-Wunknown-pragmas]
   38 |     #pragma omp parallel for reduction(+:s)
      | 
/content/alien-intelligence/include/types.hpp:72: warning: ignoring ‘#pragma omp parallel’ [-Wunknown-pragmas]
   72 |     #pragma omp parallel for
      | 
/content/alien-intelligence/include/types.hpp:83: warning: ignoring ‘#pragma omp parallel’ [-Wunknown-pragmas]
   83 |     #pragma omp parallel for
      | 
[ 27%] Linking CUDA static library libai2_gpu.a
[ 27%] Built target ai2_gpu
[ 33%] Building CXX object CMakeFiles/ai2_core.dir/src/uq.cpp.o
In file included from /content/alien-intelligence/include/uq.hpp:2,
                 from /content/alien-intelligence/src/uq.cpp:1:
/content/alien-intelligence/include/types.hpp:38: warning: ignoring ‘#pragma omp parallel’ [-Wunknown-pragmas]
   38 |     #pragma omp parallel for reduction(+:s)
      | 
/content/alien-intelligence/include/types.hpp:72: warning: ignoring ‘#pragma omp parallel’ [-Wunknown-pragmas]
   72 |     #pragma omp parallel for
      | 
/content/alien-intelligence/include/types.hpp:83: warning: ignoring ‘#pragma omp parallel’ [-Wunknown-pragmas]
   83 |     #pragma omp parallel for
      | 
[ 38%] Building CXX object CMakeFiles/ai2_core.dir/src/ataa.cpp.o
In file included from /content/alien-intelligence/include/ataa.hpp:2,
                 from /content/alien-intelligence/src/ataa.cpp:1:
/content/alien-intelligence/include/types.hpp:38: warning: ignoring ‘#pragma omp parallel’ [-Wunknown-pragmas]
   38 |     #pragma omp parallel for reduction(+:s)
      | 
/content/alien-intelligence/include/types.hpp:72: warning: ignoring ‘#pragma omp parallel’ [-Wunknown-pragmas]
   72 |     #pragma omp parallel for
      | 
/content/alien-intelligence/include/types.hpp:83: warning: ignoring ‘#pragma omp parallel’ [-Wunknown-pragmas]
   83 |     #pragma omp parallel for
      | 
[ 44%] Building CXX object CMakeFiles/ai2_core.dir/src/ssog.cpp.o
In file included from /content/alien-intelligence/include/ssog.hpp:2,
                 from /content/alien-intelligence/src/ssog.cpp:1:
/content/alien-intelligence/include/types.hpp:38: warning: ignoring ‘#pragma omp parallel’ [-Wunknown-pragmas]
   38 |     #pragma omp parallel for reduction(+:s)
      | 
/content/alien-intelligence/include/types.hpp:72: warning: ignoring ‘#pragma omp parallel’ [-Wunknown-pragmas]
   72 |     #pragma omp parallel for
      | 
/content/alien-intelligence/include/types.hpp:83: warning: ignoring ‘#pragma omp parallel’ [-Wunknown-pragmas]
   83 |     #pragma omp parallel for
      | 
[ 50%] Linking CXX static library libai2_core.a
[ 50%] Built target ai2_core
[ 55%] Building CXX object CMakeFiles/ai2_train_lib.dir/src/tokenizer.cpp.o
[ 61%] Building CXX object CMakeFiles/ai2_train_lib.dir/src/dataloader.cpp.o
In file included from /content/alien-intelligence/include/tokenizer.hpp:2,
                 from /content/alien-intelligence/src/tokenizer.cpp:1:
/content/alien-intelligence/include/types.hpp:38: warning: ignoring ‘#pragma omp parallel’ [-Wunknown-pragmas]
   38 |     #pragma omp parallel for reduction(+:s)
      | 
In file included from /content/alien-intelligence/include/dataloader.hpp:2,
                 from /content/alien-intelligence/src/dataloader.cpp:1:
/content/alien-intelligence/include/types.hpp:38: warning: ignoring ‘#pragma omp parallel’ [-Wunknown-pragmas]
   38 |     #pragma omp parallel for reduction(+:s)
      | 
/content/alien-intelligence/include/types.hpp:72: warning: ignoring ‘#pragma omp parallel’ [-Wunknown-pragmas]
   72 |     #pragma omp parallel for
      | 
/content/alien-intelligence/include/types.hpp:72: warning: ignoring ‘#pragma omp parallel’ [-Wunknown-pragmas]
   72 |     #pragma omp parallel for
      | 
/content/alien-intelligence/include/types.hpp:83: warning: ignoring ‘#pragma omp parallel’ [-Wunknown-pragmas]
   83 |     #pragma omp parallel for
      | 
/content/alien-intelligence/include/types.hpp:83: warning: ignoring ‘#pragma omp parallel’ [-Wunknown-pragmas]
   83 |     #pragma omp parallel for
      | 
[ 66%] Building CXX object CMakeFiles/ai2_train_lib.dir/src/model.cpp.o
In file included from /content/alien-intelligence/include/model.hpp:2,
                 from /content/alien-intelligence/src/model.cpp:1:
/content/alien-intelligence/include/types.hpp:38: warning: ignoring ‘#pragma omp parallel’ [-Wunknown-pragmas]
   38 |     #pragma omp parallel for reduction(+:s)
      | 
/content/alien-intelligence/include/types.hpp:72: warning: ignoring ‘#pragma omp parallel’ [-Wunknown-pragmas]
   72 |     #pragma omp parallel for
      | 
/content/alien-intelligence/include/types.hpp:83: warning: ignoring ‘#pragma omp parallel’ [-Wunknown-pragmas]
   83 |     #pragma omp parallel for
      | 
[ 72%] Building CXX object CMakeFiles/ai2_train_lib.dir/src/optimizer.cpp.o
In file included from /content/alien-intelligence/include/optimizer.hpp:2,
                 from /content/alien-intelligence/src/optimizer.cpp:1:
/content/alien-intelligence/include/types.hpp:38: warning: ignoring ‘#pragma omp parallel’ [-Wunknown-pragmas]
   38 |     #pragma omp parallel for reduction(+:s)
      | 
/content/alien-intelligence/include/types.hpp:72: warning: ignoring ‘#pragma omp parallel’ [-Wunknown-pragmas]
   72 |     #pragma omp parallel for
      | 
/content/alien-intelligence/include/types.hpp:83: warning: ignoring ‘#pragma omp parallel’ [-Wunknown-pragmas]
   83 |     #pragma omp parallel for
      | 
/content/alien-intelligence/src/model.cpp: In constructor ‘ai2::Model::Model(const ai2::ModelConfig&)’:
/content/alien-intelligence/src/model.cpp:38:33: error: ‘std::vector<std::vector<std::vector<double> > > ai2::SSOG::expert_weights_’ is private within this context
   38 |                          ssog_->expert_weights_, ssog_->expert_biases_,
      |                                 ^~~~~~~~~~~~~~~
In file included from /content/alien-intelligence/include/model.hpp:8,
                 from /content/alien-intelligence/src/model.cpp:1:
/content/alien-intelligence/include/ssog.hpp:91:22: note: declared private here
   91 |     std::vector<Mat> expert_weights_;
      |                      ^~~~~~~~~~~~~~~
/content/alien-intelligence/src/model.cpp:38:57: error: ‘std::vector<std::vector<double> > ai2::SSOG::expert_biases_’ is private within this context
   38 |                          ssog_->expert_weights_, ssog_->expert_biases_,
      |                                                         ^~~~~~~~~~~~~~
In file included from /content/alien-intelligence/include/model.hpp:8,
                 from /content/alien-intelligence/src/model.cpp:1:
/content/alien-intelligence/include/ssog.hpp:94:22: note: declared private here
   94 |     std::vector<Vec> expert_biases_;
      |                      ^~~~~~~~~~~~~~
gmake[3]: *** [CMakeFiles/ai2_train_lib.dir/build.make:107: CMakeFiles/ai2_train_lib.dir/src/model.cpp.o] Error 1
gmake[3]: *** Waiting for unfinished jobs....
gmake[2]: *** [CMakeFiles/Makefile2:182: CMakeFiles/ai2_train_lib.dir/all] Error 2
gmake[1]: *** [CMakeFiles/Makefile2:455: CMakeFiles/ai2_train.dir/rule] Error 2
gmake: *** [Makefile:264: ai2_train] Error 2
[3/3] Starting training (GPU) ...
train.txt: line 21: ./build/ai2_train: No such file or directory

