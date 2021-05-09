# Minimum version - 3.17
# Required for find_package(CUDAToolkit)
# Can be lowered if not using CUDA.
cmake_minimum_required(VERSION 3.17)

# Project name
project(LMECA2300 LANGUAGES C)

# Exec name
set(EXEC_NAME project)
set(CMAKE_C_FLAGS "-O3")

# CMake module
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

# Project files
include_directories(${PROJECT_SOURCE_DIR}/inc)
file(GLOB C_SRCS "${PROJECT_SOURCE_DIR}/src/*.c")

# Compile options
option(USE_CUDA "Compile GPU code, using CUDA." OFF)
option(MOBILITY "Mobility coefficient model." OFF)
option(N_DISCR "Discretisation grid to use." OFF)
option(SOLVER "Solver to use." OFF)

# Number of discretisation points
if (NOT N_DISCR)
    set(N_DISCR 128)
endif (NOT N_DISCR)
message(STATUS "Using a ${N_DISCR}x${N_DISCR} grid.")
add_definitions(-DN_DISCR=${N_DISCR})

# Solver to use
if (NOT SOLVER)
    set(SOLVER ETDRK4)
endif (NOT SOLVER)
message(STATUS "Using the ${SOLVER} solver.")
add_definitions(-DSOLVER=${SOLVER})

# Constant or Variable mobility ?
if (NOT MOBILITY OR MOBILITY STREQUAL CONSTANT)
    message(STATUS "Simulating constant mobility.")
    set(TIMESTEP 1e-6)
    set(KAPPA 1e-4)
    add_definitions(-DCONSTANT_MOBILITY)

elseif (MOBILITY STREQUAL VARIABLE)
    message(STATUS "Simulating variable mobility.")
    set(TIMESTEP 1)
    set(KAPPA 1)
    add_definitions(-DVARIABLE_MOBILITY)

    if (NOT SOLVER STREQUAL ETDRK4 OR NOT USE_CUDA)
        message(FATAL_ERROR "Variable mobility is only implemented for ETDRK4 using CUDA.")
    endif ()

else ()
    message(FATAL_ERROR "Unknown mobility model type.")
endif ()

add_definitions(-DTIMESTEP=${TIMESTEP})
add_definitions(-DKAPPA=${KAPPA})

# CPU or GPU (CUDA)
if (USE_CUDA)
    # Enable CUDA
    message(STATUS "Compiling with CUDA...")
    enable_language(CUDA)

    # cuFFT & co
    find_package(CUDA REQUIRED)
    find_package(CUDAToolkit REQUIRED)
    set(EXT_LIBS ${EXT_LIBS} CUDA::cufft)
    add_definitions(-DUSE_CUDA)

    # GPU-specific files
    include_directories(${PROJECT_SOURCE_DIR}/inc/gpu)
    file(GLOB CUDA_SRCS "${PROJECT_SOURCE_DIR}/src/gpu/*.cu")
    file(GLOB C_SRCS ${C_SRCS} "${PROJECT_SOURCE_DIR}/src/gpu/*.c")

    # Compile *.cu source files
    cuda_compile(CUDA_O ${CUDA_SRCS})

else ()
    # CPU-specific files
    include_directories(${PROJECT_SOURCE_DIR}/inc/cpu)
    file(GLOB C_SRCS ${C_SRCS} "${PROJECT_SOURCE_DIR}/src/cpu/*.c")

    # FFT libraries - MKL or FFTW3 (if MKL not found)
    find_package(MKL)
    if (MKL_FOUND)          # Intel's MKL
        include_directories(${MKL_INCLUDE_DIR})
        set(EXT_LIBS ${EXT_LIBS} ${MKL_LIBRARIES})
    else ()        # FFTW3
        message(STATUS "Intel's MKL not found -- proceeding with FFTW3.")
        find_package(FFTW REQUIRED)
        include_directories(${FFTW_INCLUDE_DIRS})
        set(EXT_LIBS ${EXT_LIBS} ${FFTW_LIBRARIES})
    endif ()
endif ()

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
