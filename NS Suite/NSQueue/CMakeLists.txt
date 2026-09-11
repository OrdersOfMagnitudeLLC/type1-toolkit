cmake_minimum_required(VERSION 3.20)
project(nsqueue CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

add_executable(nsqueue queue_bench.cpp)
target_include_directories(nsqueue PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nsqueue PRIVATE Threads::Threads hwy)
target_compile_options(nsqueue PRIVATE -O3 -march=native)
