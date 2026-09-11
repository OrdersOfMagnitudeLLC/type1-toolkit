cmake_minimum_required(VERSION 3.20)
project(nsquant CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

set(NSRUN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../../deprecated/NSRun/src)
set(NSRUN_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../../deprecated/NSRun/include)

add_executable(nsquant
    nsquant.cpp
    ${NSRUN_SRC}/gguf_parser.cpp
    ${NSRUN_SRC}/tokenizer.cpp
    ${NSRUN_SRC}/nsm_loader.cpp
)
target_include_directories(nsquant PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${NSRUN_SRC}
    ${NSRUN_INCLUDE}
)
target_link_libraries(nsquant PRIVATE z Threads::Threads OpenMP::OpenMP_CXX)
target_compile_options(nsquant PRIVATE -O3 -march=native -fopenmp)
