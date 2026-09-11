cmake_minimum_required(VERSION 3.20)
project(nssort CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(nssort NSSort.cpp nssort_bench.cpp)
target_include_directories(nssort PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/ips4o)
target_compile_definitions(nssort PRIVATE USE_IPS4O BENCH_MODE IPS4O_PARALLEL)
target_link_libraries(nssort PRIVATE Threads::Threads OpenMP::OpenMP_CXX tbb atomic sodium hwy)
target_compile_options(nssort PRIVATE -O3 -march=native -mavx2)
target_link_options(nssort PRIVATE -Wl,--allow-multiple-definition)

add_executable(bench_vqsort bench_vqsort.cpp)
target_compile_options(bench_vqsort PRIVATE -O3 -march=native)
target_link_libraries(bench_vqsort PRIVATE hwy hwy_contrib)
