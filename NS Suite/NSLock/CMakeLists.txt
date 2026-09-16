cmake_minimum_required(VERSION 3.20)
project(nslock CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

add_executable(nslock lock_bench.cpp)
target_include_directories(nslock PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nslock PRIVATE Threads::Threads)
target_compile_options(nslock PRIVATE -O3 -march=native)
