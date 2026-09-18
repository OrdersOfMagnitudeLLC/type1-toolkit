cmake_minimum_required(VERSION 3.20)
project(nsinfer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(nsinfer nsinfer_bench.cpp)
target_include_directories(nsinfer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(nsinfer PRIVATE -O3 -march=native)
