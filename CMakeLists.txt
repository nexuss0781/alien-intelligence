cmake_minimum_required(VERSION 3.18)
project(alien_reasoning LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release: optimized
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O2 -DNDEBUG")
# Debug: for tests
set(CMAKE_CXX_FLAGS_DEBUG "-Og -g -Wall -Wextra -Wpedantic")
set(CMAKE_CUDA_FLAGS_DEBUG "-G")
# Default
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -O2")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2 -Wno-deprecated-gpu-targets")

include_directories(include)

# GPU backend (CUDA)
add_library(ai2_gpu STATIC
    src/gpu_backend.cu
)

add_library(ai2_core STATIC
    src/slie.cpp
    src/lssc.cpp
    src/stre.cpp
    src/uq.cpp
    src/ataa.cpp
    src/ssog.cpp
)

add_library(ai2_train_lib STATIC
    src/tokenizer.cpp
    src/dataloader.cpp
    src/model.cpp
    src/optimizer.cpp
    src/trainer.cpp
    src/hidden_cache.cpp
)
target_link_libraries(ai2_train_lib ai2_core ai2_gpu)

foreach(COMPONENT slie lssc stre uq ataa ssog)
    add_executable(test_${COMPONENT} tests/test_${COMPONENT}.cpp)
    target_link_libraries(test_${COMPONENT} ai2_core ai2_gpu)
endforeach()

add_executable(test_math tests/test_math.cpp)
target_link_libraries(test_math ai2_train_lib ai2_gpu)

add_executable(ai2_train src/main_train.cpp)
target_link_libraries(ai2_train ai2_train_lib ai2_gpu)

enable_testing()
