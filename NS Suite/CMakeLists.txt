cmake_minimum_required(VERSION 3.20)
project(NSSuite)

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

# Add only products that use Highway
add_subdirectory(NSHash)
add_subdirectory(NSFix)
add_subdirectory(NSQueue)
add_subdirectory(NSLock)
add_subdirectory(NSIndex)
add_subdirectory(NSSort)
add_subdirectory(NSBVH)
add_subdirectory(NSStringIndex)
add_subdirectory(NSAttend)
add_subdirectory(NSMatrix)
add_subdirectory(NSNet)
add_subdirectory(NSGemm)
add_subdirectory(NSCrypt)
