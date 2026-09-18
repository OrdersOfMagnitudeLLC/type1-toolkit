cmake_minimum_required(VERSION 3.20)
project(nskvcache CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(kv_alloc kv_alloc.cpp)
target_compile_options(kv_alloc PRIVATE -O3 -march=native)

add_executable(kv_box kv_box.cpp)
target_compile_options(kv_box PRIVATE -O3 -march=native)
