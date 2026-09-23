cmake_minimum_required(VERSION 3.20)
project(nsnet CXX)

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

add_executable(nsnet bench_nsnet.cpp)
target_include_directories(nsnet PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nsnet PRIVATE hwy)
target_compile_options(nsnet PRIVATE -O3 -march=native)
