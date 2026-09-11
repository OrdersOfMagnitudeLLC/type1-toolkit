cmake_minimum_required(VERSION 3.20)
project(nshash CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(benchmark REQUIRED)
find_package(absl REQUIRED)

add_executable(nshash bench.cpp)
target_include_directories(nshash PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
target_link_libraries(nshash PRIVATE benchmark::benchmark benchmark::benchmark_main absl::flat_hash_map pthread hwy)
target_compile_options(nshash PRIVATE -O3 -march=native)

add_executable(nshash_server nshash_server.cpp)
target_include_directories(nshash_server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(nshash_server PRIVATE -O3 -march=native)
