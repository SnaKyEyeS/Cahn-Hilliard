# Minimum version
cmake_minimum_required(VERSION 3.12.2)

# Project name
project(LMECA2300 C)

# Exec name
set(EXEC_NAME project)
set(CMAKE_C_FLAGS "-O3")

# CMake module
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

# FFTW3
find_package(FFTW REQUIRED)
include_directories(${FFTW_INCLUDE_DIRS})

# GLEW
find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIRS})

# GLFW3
find_package(GLFW3 REQUIRED)
include_directories(${GLFW3_INCLUDE_DIR})

# OpenGL
find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIR})

# Project files
include_directories(${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/inc)
file(GLOB C_SRCS "${PROJECT_SOURCE_DIR}/src/*.c")

# Compile/link src files & libs
add_executable(${EXEC_NAME} ${C_SRCS})
target_link_libraries(
    ${EXEC_NAME}
    ${FFTW_LIBRARIES}
    ${GLEW_LIBRARIES}
    ${GLFW3_LIBRARY}
    ${OPENGL_LIBRARY}
    m   # LIBM
)
