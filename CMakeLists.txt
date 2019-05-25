cmake_minimum_required(VERSION 3.0)

project(heat_solver_3d VERSION 1.0.0)
set(PROJECT_DESCRIPTION "library for solving 3D heat/diffusion equations")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

option(OPTIMIZE_FOR_NATIVE "Build with -march=native" OFF)
if(OPTIMIZE_FOR_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

find_package(MPI REQUIRED)

include_directories(${MPI_INCLUDE_PATH})
set(HEAT_SOLVER_3D_INCLUDE_PATH "include")
get_filename_component(HEAT_SOLVER_3D_INCLUDE_PATH "${CMAKE_CURRENT_LIST_DIR}/include" ABSOLUTE)
include_directories(${HEAT_SOLVER_3D_INCLUDE_PATH})
file(GLOB HEADERS include/heat_solver_3d/*.hpp)
file(GLOB SOURCES src/*.cpp)
add_library(${PROJECT_NAME} SHARED ${HEADERS} ${SOURCES})
target_link_libraries(${PROJECT_NAME} ${MPI_LIBRARIES})

if(MPI_COMPILE_FLAGS)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

option(BUILD_EXAMPLES "Build examples as well" OFF)
if(BUILD_EXAMPLES)
    add_subdirectory(examples/quasilinear)
    add_subdirectory(examples/plasma_jet)
endif(BUILD_EXAMPLES)

option(BUILD_TOOLS "Build tools as well" ON)
if(BUILD_TOOLS)
    add_subdirectory(tools/results_reader)
endif(BUILD_TOOLS)
