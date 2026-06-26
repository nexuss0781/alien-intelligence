cmake_minimum_required(VERSION 3.14)
project(alien_reasoning LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -O2 -DNDEBUG")

include_directories(include)

add_library(ai2_core STATIC
    src/slie.cpp
    src/lssc.cpp
    src/stre.cpp
    src/uq.cpp
    src/ataa.cpp
    src/ssog.cpp
)

add_library(ai2_train STATIC
    src/tokenizer.cpp
    src/dataloader.cpp
    src/model.cpp
    src/optimizer.cpp
    src/trainer.cpp
)

foreach(COMPONENT slie lssc stre uq ataa ssog)
    add_executable(test_${COMPONENT} tests/test_${COMPONENT}.cpp)
    target_link_libraries(test_${COMPONENT} ai2_core)
    add_test(NAME ${COMPONENT} COMMAND test_${COMPONENT})
endforeach()

add_executable(ai2_train src/main_train.cpp)
target_link_libraries(ai2_train ai2_core ai2_train)
target_link_libraries(ai2_train stdc++fs)

enable_testing()
