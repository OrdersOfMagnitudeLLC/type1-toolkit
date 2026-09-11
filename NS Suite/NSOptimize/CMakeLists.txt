cmake_minimum_required(VERSION 3.20)
project(nsoptimize CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(nsoptimize optimize_bench.cpp)
target_include_directories(nsoptimize PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(nsoptimize PRIVATE -O3 -march=native)
