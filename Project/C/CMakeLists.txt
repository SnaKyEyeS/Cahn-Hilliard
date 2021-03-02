# Minimum version
cmake_minimum_required(VERSION 3.12.2)

# Project name
project(LMECA2300 C)

# Exec name
set(EXEC_NAME project)

# CMake module
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

# Math
set(CMAKE_REQUIRED_LIBRARIES m)
unset(CMAKE_REQUIRED_LIBRARIES)
set(LIBM_LIBRARIES m)

# FFTW3
find_package(FFTW REQUIRED)
include_directories(${FFTW_INCLUDE_DIRS})

# OpenGL
find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIR})

# BOV
add_subdirectory(${PROJECT_SOURCE_DIR}/deps/bov)

# Project files
# include_directories(${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/inc)
file(GLOB C_SRCS "${PROJECT_SOURCE_DIR}/src/spectral.c")

# Compile/link src files & libs
add_executable(${EXEC_NAME} ${C_SRCS})
target_link_libraries(
    ${EXEC_NAME}
    ${LIBM_LIBRARIES}
    ${FFTW_LIBRARIES}
    ${OPENGL_gl_LIBRARY}
    bov
)
