cmake_minimum_required(VERSION 3.20)
project(nsfix CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(nsfix bench_nsfix.cpp)
target_include_directories(nsfix PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(nsfix PRIVATE -O3 -march=native)

add_executable(nsfix_full bench_nsfix_full.cpp)
target_include_directories(nsfix_full PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../NSHash)
target_compile_options(nsfix_full PRIVATE -O3 -march=native)

add_executable(nsfix_bare bench_nsfix_bare.cpp)
target_include_directories(nsfix_bare PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(nsfix_bare PRIVATE -O3 -march=native)

option(NS_FIX_ALL "Build all NSFix targets" OFF)
if(NS_FIX_ALL)
    add_executable(nsfix_batch bench_batch.cpp)
    target_include_directories(nsfix_batch PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_options(nsfix_batch PRIVATE -O3 -march=native)

    add_executable(nsfix_diagnostic bench_diagnostic.cpp)
    target_include_directories(nsfix_diagnostic PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_options(nsfix_diagnostic PRIVATE -O3 -march=native)

    add_executable(nsfix_parse_only bench_parse_only.cpp)
    target_include_directories(nsfix_parse_only PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_options(nsfix_parse_only PRIVATE -O3 -march=native)

    add_executable(nsfix_quickfix bench_quickfix.cpp)
    target_include_directories(nsfix_quickfix PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    set_target_properties(nsfix_quickfix PROPERTIES CXX_STANDARD 14)
    target_compile_options(nsfix_quickfix PRIVATE -O3 -march=native)
    find_library(QUICKFIX_LIB NAMES quickfix libquickfix)
if(QUICKFIX_LIB)
    target_link_libraries(nsfix_quickfix PRIVATE ${QUICKFIX_LIB})
else()
    message(WARNING "quickfix not found - nsfix_quickfix may fail")
endif()

    add_executable(nsfix_scalar bench_scalar.cpp)
    target_include_directories(nsfix_scalar PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_options(nsfix_scalar PRIVATE -O3 -march=native)

    add_executable(nsfix_benchmark fix_benchmark.cpp)
    target_include_directories(nsfix_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_options(nsfix_benchmark PRIVATE -O3 -march=native)

    add_executable(nsfix_hash_benchmark fix_hash_benchmark.cpp)
    target_include_directories(nsfix_hash_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_options(nsfix_hash_benchmark PRIVATE -O3 -march=native)

    add_executable(nsfix_server nsfix_server.cpp)
    target_include_directories(nsfix_server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_options(nsfix_server PRIVATE -O3 -march=native)

    add_executable(nsfix_test test.cpp)
    target_include_directories(nsfix_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_options(nsfix_test PRIVATE -O3 -march=native)
endif()
