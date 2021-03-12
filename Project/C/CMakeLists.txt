# Minimum version
cmake_minimum_required(VERSION 3.17)

# Project name
project(LMECA2300 LANGUAGES C CUDA)

# Exec name
set(EXEC_NAME project)
set(CMAKE_C_FLAGS "-O3")

# CMake module
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

# Project files
include_directories(${PROJECT_SOURCE_DIR}/inc)
file(GLOB C_SRCS    "${PROJECT_SOURCE_DIR}/src/*.c")
file(GLOB CUDA_SRCS "${PROJECT_SOURCE_DIR}/src/*.cu")

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

# CUDA
option(USE_CUDA "Compile GPU code, using CUDA." OFF)
if (USE_CUDA)
    # cuFFT & co
    find_package(CUDA REQUIRED)
    find_package(CUDAToolkit REQUIRED)
    set(EXT_LIBS ${EXT_LIBS} CUDA::cufft)
    add_definitions(-DUSE_CUDA)

    # Compile *.cu source files
    cuda_compile(CUDA_O ${CUDA_SRCS})
endif (USE_CUDA)

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

# Compile/link src files & libs
add_executable(${EXEC_NAME} ${C_SRCS} ${CUDA_O})
target_link_libraries(${EXEC_NAME} ${EXT_LIBS} m dl pthread)

# Target properties
set_property(TARGET ${EXEC_NAME} PROPERTY CUDA_ARCHITECTURES OFF)
