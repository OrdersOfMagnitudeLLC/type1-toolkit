cmake_minimum_required(VERSION 3.20)
project(nsfft CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(nsfft nsfft_bench.cpp)
target_include_directories(nsfft PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nsfft PRIVATE fftw3)
target_compile_options(nsfft PRIVATE -O3 -march=native)
