cmake_minimum_required(VERSION 3.20)
project(nsmesh CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(nsmesh nsmesh_bench.cpp)
target_include_directories(nsmesh PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nsmesh PRIVATE z)
target_compile_options(nsmesh PRIVATE -O3 -march=native)
