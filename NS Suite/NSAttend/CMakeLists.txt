cmake_minimum_required(VERSION 3.20)
project(nsattend CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(nsattend nsattend_bench.cpp)
target_include_directories(nsattend PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(nsattend PRIVATE -O3 -march=native)

option(NS_CUDA "Build CUDA variant" OFF)
if(NS_CUDA)
    enable_language(CUDA)
    add_executable(nsattend_cuda nsattend_cuda.cu)
    target_compile_options(nsattend_cuda PRIVATE -O3 -march=native)
endif()
