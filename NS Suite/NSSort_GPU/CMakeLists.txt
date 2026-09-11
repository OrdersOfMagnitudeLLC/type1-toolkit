cmake_minimum_required(VERSION 3.20)
project(nssort_gpu)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

option(NS_CUDA OFF)

if(NS_CUDA)
    find_package(CUDAToolkit REQUIRED)
    enable_language(CUDA)
    add_executable(nssort_gpu NSSort_GPU.cu)
    target_compile_options(nssort_gpu PRIVATE -O3 -march=native)
    target_link_libraries(nssort_gpu PRIVATE CUDA::cudart)
endif()
