cmake_minimum_required(VERSION 3.20)
project(nscomp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(nscomp_bench nscomp_bench.cpp nscomp.cpp)
target_include_directories(nscomp_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nscomp_bench PRIVATE zstd lz4)
target_compile_options(nscomp_bench PRIVATE -O3 -march=native)

add_executable(nscomp_server nscomp_server.cpp nscomp.cpp)
target_include_directories(nscomp_server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nscomp_server PRIVATE ssl crypto zstd lz4)
target_compile_options(nscomp_server PRIVATE -O3 -march=native)
