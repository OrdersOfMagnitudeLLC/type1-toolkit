cmake_minimum_required(VERSION 3.16)
project(lacuna C)

# Lacuna ships as prebuilt stripped binaries — no source compilation.
# Link your binary against the prebuilt liblacuna.a.

find_package(PkgConfig REQUIRED)
pkg_check_modules(SODIUM REQUIRED libsodium)

add_library(lacuna STATIC IMPORTED)
set_target_properties(lacuna PROPERTIES
    IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/lib/liblacuna.a
    INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}
    INTERFACE_LINK_LIBRARIES ${SODIUM_LIBRARIES}
)
target_link_directories(lacuna INTERFACE ${SODIUM_LIBRARY_DIRS})

# Example: link your binary against lacuna
# add_executable(my_app main.c)
# target_link_libraries(my_app PRIVATE lacuna)
