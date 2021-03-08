# Minimum version
cmake_minimum_required(VERSION 3.12.2)

# Project name
project(LMECA2300 C)

# Exec name
set(EXEC_NAME project)
set(CMAKE_C_FLAGS "-O3")

# CMake module
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

# OpenMP
option(USE_OPENMP "Compile parallel code, using OpenMP." OFF)
if (USE_OPENMP)

    message(STATUS "Compiling with OpenMP...")
    find_package(OpenMP REQUIRED)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    include_directories(${OPENMP_C_INCLUDE_DIRS})
    set(EXT_LIBS ${EXT_LIBS} ${OPENMP_C_LIBRARIES})
    add_definitions(-DUSE_OPENMP)

endif (USE_OPENMP)

# FFT libraries - MKL or FFTW3 (if MKL not found)
find_package(MKL)
if (MKL_FOUND)          # Intel's MKL

    include_directories(${MKL_INCLUDE_DIR})
    set(EXT_LIBS ${EXT_LIBS} ${MKL_LIBRARIES})

else (MKL_FOUND)        # FFTW3

    message(STATUS "Intel's MKL not found -- proceeding with FFTW3.")
    find_package(FFTW REQUIRED)
    include_directories(${FFTW_INCLUDE_DIRS})
    set(EXT_LIBS ${EXT_LIBS} ${FFTW_LIBRARIES})

endif (MKL_FOUND)

# GLEW
find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIRS})
set(EXT_LIBS ${EXT_LIBS} ${GLEW_LIBRARIES})

# GLFW3
find_package(GLFW3 REQUIRED)
include_directories(${GLFW3_INCLUDE_DIR})
set(EXT_LIBS ${EXT_LIBS} ${GLFW3_LIBRARY})

# OpenGL
find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIR})
set(EXT_LIBS ${EXT_LIBS} ${OPENGL_LIBRARY})

# Project files
include_directories(${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/inc)
file(GLOB C_SRCS "${PROJECT_SOURCE_DIR}/src/*.c")

# Compile/link src files & libs
add_executable(${EXEC_NAME} ${C_SRCS})
target_link_libraries(${EXEC_NAME} ${EXT_LIBS} m dl pthread)
