cmake_minimum_required(VERSION 3.10)

include(FetchContent)
FetchContent_Declare(
  highway
  GIT_REPOSITORY https://github.com/google/highway.git
  GIT_TAG        1.2.0
  GIT_SHALLOW    TRUE
)
set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(highway)

project(NSMace LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Highway compiles to best available SIMD on the build machine.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -fopenmp")

# FP64 reference build option.
option(NSMACE_FP64 "Use double precision for reference build" OFF)
if(NSMACE_FP64)
    add_compile_definitions(NSMACE_FP64)
endif()

# Highway lives under ./hwy -> vendor/highway/hwy
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${highway_SOURCE_DIR})

add_executable(NSMace
    core/NSMaceEngine.cpp
    core/NeighborList.cpp
    core/WeightLoader.cpp
)

target_link_libraries(NSMace PRIVATE gomp)

# FP64 reference binary, built when -DNSMACE_FP64=ON is supplied.
if(NOT NSMACE_FP64)
    add_executable(NSMace_fp64
        core/NSMaceEngine.cpp
        core/NeighborList.cpp
        core/WeightLoader.cpp
    )
    target_compile_definitions(NSMace_fp64 PRIVATE NSMACE_FP64)
    target_link_libraries(NSMace_fp64 PRIVATE gomp)
endif()

enable_testing()

# Parity test on the built-in H2O benchmark.
add_test(NAME parity_h2o_fp32 COMMAND NSMace)
set_tests_properties(parity_h2o_fp32 PROPERTIES
    PASS_REGULAR_EXPRESSION "Delta:     0 eV")

# Multi-run timing harness.
add_test(NAME timing_h2o COMMAND bash ${CMAKE_SOURCE_DIR}/tests/benchmark.sh $<TARGET_FILE:NSMace>)
