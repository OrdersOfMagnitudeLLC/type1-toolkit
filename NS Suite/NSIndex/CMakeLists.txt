cmake_minimum_required(VERSION 3.20)
project(nsindex CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(nsindex nsindex_bench.cpp)
target_include_directories(nsindex PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/pgm ${CMAKE_CURRENT_SOURCE_DIR}/third_party/alex)
target_compile_options(nsindex PRIVATE -O3 -march=native)

add_executable(nsindex_server nsindex_server.cpp)
target_include_directories(nsindex_server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(nsindex_server PRIVATE -O3 -march=native)
