# Minimum version
cmake_minimum_required(VERSION 3.12.2)

# Project name
project(Project CXX)

# Exec name
set(EXEC_NAME project)

# CMake module
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

# Math
set(CMAKE_REQUIRED_LIBRARIES m)
unset(CMAKE_REQUIRED_LIBRARIES)
set(LIBM_LIBRARIES m)

# Python
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

# FFTW3
find_package(FFTW REQUIRED)
include_directories(${FFTW_INCLUDE_DIRS})

# Deps
include_directories(${PROJECT_SOURCE_DIR}/deps)

# Project files
include_directories(${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/inc)
file(GLOB C_SRCS
    "${PROJECT_SOURCE_DIR}/src/*.c"
    "${PROJECT_SOURCE_DIR}/src/*.cpp"
)

# Use G++ for compiling/linking the source files
set_source_files_properties(${C_SRCS} PROPERTIES LANGUAGE CXX)
add_executable(${EXEC_NAME} ${C_SRCS})
target_link_libraries(${EXEC_NAME} ${LIBM_LIBRARIES} ${PYTHON_LIBRARIES} ${FFTW_LIBRARIES})
