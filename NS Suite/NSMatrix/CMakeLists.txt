cmake_minimum_required(VERSION 3.20)
project(nsmatrix CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(nsmatrix bench_nsmatrix.cpp)
target_include_directories(nsmatrix PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nsmatrix PRIVATE hwy)
target_compile_options(nsmatrix PRIVATE -O3 -march=native)
