cmake_minimum_required(VERSION 3.20)
project(nsgraph CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(nsgraph nsgraph_bench.cpp)
target_include_directories(nsgraph PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(nsgraph PRIVATE -O3 -march=native)
