cmake_minimum_required(VERSION 3.14)
project(alien_reasoning LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release: optimized with OpenMP for fast training
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -march=native -fopenmp -DNDEBUG")
# Debug: for tests
set(CMAKE_CXX_FLAGS_DEBUG "-Og -g -Wall -Wextra -Wpedantic")
# Default: release-like for training speed
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -O2 -march=native -fopenmp")

include_directories(include)
find_package(OpenMP REQUIRED)

add_library(ai2_core STATIC
    src/slie.cpp
    src/lssc.cpp
    src/stre.cpp
    src/uq.cpp
    src/ataa.cpp
    src/ssog.cpp
)
target_link_libraries(ai2_core OpenMP::OpenMP_CXX)

add_library(ai2_train_lib STATIC
    src/tokenizer.cpp
    src/dataloader.cpp
    src/model.cpp
    src/optimizer.cpp
    src/trainer.cpp
)
target_link_libraries(ai2_train_lib ai2_core OpenMP::OpenMP_CXX)

foreach(COMPONENT slie lssc stre uq ataa ssog)
    add_executable(test_${COMPONENT} tests/test_${COMPONENT}.cpp)
    target_link_libraries(test_${COMPONENT} ai2_core OpenMP::OpenMP_CXX)
    add_test(NAME ${COMPONENT} COMMAND test_${COMPONENT})
endforeach()

add_executable(test_math tests/test_math.cpp)
target_link_libraries(test_math ai2_train_lib OpenMP::OpenMP_CXX)
add_test(NAME math COMMAND test_math)

add_executable(ai2_train src/main_train.cpp)
target_link_libraries(ai2_train ai2_train_lib OpenMP::OpenMP_CXX)

enable_testing()
