cmake_minimum_required(VERSION 3.20)
project(nsbvh CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(nsbvh nsbvh_bench.cpp)
target_include_directories(nsbvh PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nsbvh PRIVATE embree4 hwy)
target_compile_options(nsbvh PRIVATE -O3 -march=native)

add_executable(nsbvh_simple nsbvh_simple.cpp)
target_include_directories(nsbvh_simple PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../NSHash)
target_link_libraries(nsbvh_simple PRIVATE embree4 hwy hwy_contrib)
target_compile_options(nsbvh_simple PRIVATE -O3 -march=native)
