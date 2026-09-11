cmake_minimum_required(VERSION 3.20)
project(nsstringindex CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_executable(nsstringindex nsstringindex_bench.cpp)
target_include_directories(nsstringindex PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nsstringindex PRIVATE OpenMP::OpenMP_CXX hwy)
target_compile_options(nsstringindex PRIVATE -O3 -march=native)
