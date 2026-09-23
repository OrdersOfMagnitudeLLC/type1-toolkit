cmake_minimum_required(VERSION 3.20)
project(nsgemm CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Highway: system package, vendored copy, or fetch ---
if(NOT TARGET hwy)
    find_package(HWY QUIET)
endif()
if(NOT TARGET hwy AND TARGET hwy::hwy)
    add_library(hwy INTERFACE)
    target_link_libraries(hwy INTERFACE hwy::hwy)
endif()
if(NOT TARGET hwy)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/highway/CMakeLists.txt")
        add_subdirectory(third_party/highway EXCLUDE_FROM_ALL)
    else()
        find_library(HWY_LIB NAMES hwy libhwy)
        find_path(HWY_INCLUDE_DIR NAMES hwy/highway.h)
        if(HWY_LIB AND HWY_INCLUDE_DIR)
            add_library(hwy UNKNOWN IMPORTED)
            set_target_properties(hwy PROPERTIES
                IMPORTED_LOCATION "${HWY_LIB}"
                INTERFACE_INCLUDE_DIRECTORIES "${HWY_INCLUDE_DIR}")
        else()
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
        endif()
    endif()
endif()
# --- end Highway ---

if(NOT EIGEN3_INCLUDE_DIR)
    find_package(Eigen3 QUIET)
    if(Eigen3_FOUND AND Eigen3_INCLUDE_DIRS)
        set(EIGEN3_INCLUDE_DIR "${Eigen3_INCLUDE_DIRS}")
    elseif(EXISTS "/usr/include/eigen3")
        set(EIGEN3_INCLUDE_DIR "/usr/include/eigen3")
    else()
        message(FATAL_ERROR "Eigen3 not found; set EIGEN3_INCLUDE_DIR or install libeigen3-dev")
    endif()
endif()

add_executable(nsgemm bench_nsgemm.cpp)
target_include_directories(nsgemm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${EIGEN3_INCLUDE_DIR})
target_link_libraries(nsgemm PRIVATE hwy)
target_compile_options(nsgemm PRIVATE -O3 -march=native -DUSE_EIGEN)
