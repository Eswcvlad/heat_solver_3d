cmake_minimum_required(VERSION 3.0)

project(example_quasilinear VERSION 1.0.0 DESCRIPTION "an example of using the heat_solver_3d library for solving quasilinear equations")

find_package(MPI REQUIRED)

include_directories(${MPI_INCLUDE_PATH})
include_directories(${HEAT_SOLVER_3D_INCLUDE_PATH})
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ${MPI_LIBRARIES} heat_solver_3d)

if(MPI_COMPILE_FLAGS)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()
