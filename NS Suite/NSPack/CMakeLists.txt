cmake_minimum_required(VERSION 3.20)
project(NSPack VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# --- Highway: system package, vendored copy, or bare library ---
# Same pattern as other NS Suite products. Produces a `hwy` target and
# sets NSPACK_HAVE_HWY when a real library was found.
set(NSPACK_HAVE_HWY OFF)
if(NOT TARGET hwy)
    find_package(HWY QUIET)
endif()
if(NOT TARGET hwy AND TARGET hwy::hwy)
    add_library(hwy INTERFACE)
    target_link_libraries(hwy INTERFACE hwy::hwy)
endif()
if(TARGET hwy::hwy)
    set(NSPACK_HAVE_HWY ON)
endif()
if(NOT TARGET hwy)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/highway/CMakeLists.txt")
        add_subdirectory(third_party/highway EXCLUDE_FROM_ALL)
        set(NSPACK_HAVE_HWY ON)
    else()
        find_library(HWY_LIB NAMES hwy libhwy
                     PATHS ${HWY_ROOT} ${HWY_ROOT}/lib ${HWY_ROOT}/lib64)
        find_path(HWY_INCLUDE_DIR NAMES hwy/highway.h
                  PATHS ${HWY_ROOT} ${HWY_ROOT}/include ${HWY_ROOT}/src)
        if(HWY_LIB AND HWY_INCLUDE_DIR)
            add_library(hwy UNKNOWN IMPORTED)
            set_target_properties(hwy PROPERTIES
                IMPORTED_LOCATION "${HWY_LIB}"
                INTERFACE_INCLUDE_DIRECTORIES "${HWY_INCLUDE_DIR}")
            set(NSPACK_HAVE_HWY ON)
        else()
            message(STATUS "Highway (hwy) not found - building scalar fallbacks only")
            add_library(hwy INTERFACE)
        endif()
    endif()
endif()
# --- end Highway ---

find_package(ZSTD QUIET)
find_library(ZSTD_LIB NAMES zstd libzstd REQUIRED)
find_library(LZ4_LIB NAMES lz4 liblz4 REQUIRED)
find_library(Z_LIB NAMES z libz REQUIRED)

set(NSPACK_CODEC_LIBS ${ZSTD_LIB} ${LZ4_LIB} ${Z_LIB})
if(UNIX AND NOT APPLE)
    list(APPEND NSPACK_CODEC_LIBS dl)
endif()
if(UNIX)
    list(APPEND NSPACK_CODEC_LIBS pthread m)
endif()

# NSPack core binary
add_executable(nspack nspack.cpp NSComp/nscomp.cpp)
target_include_directories(nspack PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/NSComp)
target_link_libraries(nspack PRIVATE ${NSPACK_CODEC_LIBS})
if(NSPACK_HAVE_HWY)
    target_sources(nspack PRIVATE nspack_hwy.cpp)
    target_link_libraries(nspack PRIVATE hwy)
    target_compile_definitions(nspack PRIVATE NSPACK_HIGHWAY=1)
endif()

# NSPack FUSE filesystem (Linux/Mac only)
if(NOT WIN32)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(FUSE3 fuse3)
    endif()
    add_executable(nspack_fs nspack_fs.cpp NSComp/nscomp.cpp)
    target_include_directories(nspack_fs PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/NSComp)
    target_compile_definitions(nspack_fs PRIVATE NSPACK_NO_MAIN)
    if(FUSE3_FOUND)
        target_include_directories(nspack_fs PRIVATE ${FUSE3_INCLUDE_DIRS})
        target_link_libraries(nspack_fs PRIVATE ${FUSE3_LIBRARIES})
        target_link_directories(nspack_fs PRIVATE ${FUSE3_LIBRARY_DIRS})
    else()
        target_link_libraries(nspack_fs PRIVATE fuse3)
    endif()
    target_link_libraries(nspack_fs PRIVATE ${NSPACK_CODEC_LIBS})
    if(NSPACK_HAVE_HWY)
        target_sources(nspack_fs PRIVATE nspack_hwy.cpp)
        target_link_libraries(nspack_fs PRIVATE hwy)
        target_compile_definitions(nspack_fs PRIVATE NSPACK_HIGHWAY=1)
    endif()
endif()

# NSPack Windows FUSE (WinFSP -- Windows only)
if(WIN32)
    set(WINFSP_DIR "C:/Program Files (x86)/WinFsp" CACHE PATH "WinFSP install dir")
    add_executable(nspack_fs nspack_fs_win.cpp NSComp/nscomp.cpp)
    target_include_directories(nspack_fs PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/NSComp "${WINFSP_DIR}/inc")
    target_compile_definitions(nspack_fs PRIVATE NSPACK_NO_MAIN)
    target_link_libraries(nspack_fs PRIVATE "${WINFSP_DIR}/lib/winfsp-x64.lib" ${NSPACK_CODEC_LIBS})
    if(NSPACK_HAVE_HWY)
        target_sources(nspack_fs PRIVATE nspack_hwy.cpp)
        target_link_libraries(nspack_fs PRIVATE hwy)
        target_compile_definitions(nspack_fs PRIVATE NSPACK_HIGHWAY=1)
    endif()
endif()

# Cross-check the Windows sources on POSIX hosts without the WinFSP SDK.
if(NOT WIN32)
    add_custom_target(check-win
        COMMAND ${CMAKE_CXX_COMPILER} -std=c++17 -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/NSComp -D_WIN32 -fsyntax-only ${CMAKE_CURRENT_SOURCE_DIR}/nspack_fs_win.cpp
        COMMAND ${CMAKE_CXX_COMPILER} -std=c++17 -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/NSComp -D_WIN32 -fsyntax-only ${CMAKE_CURRENT_SOURCE_DIR}/nspack.cpp
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Type-checking Windows sources with -D_WIN32")
endif()
