cmake_minimum_required(VERSION 3.20)
project(nscache CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(nscache bench_nscache.cpp)
target_include_directories(nscache PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(nscache PRIVATE -O3 -march=native)
