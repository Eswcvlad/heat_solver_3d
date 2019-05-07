cmake_minimum_required(VERSION 3.0)

project(results_reader VERSION 1.0.0)
set(PROJECT_DESCRIPTION "a small tool for reading binary results of the heat solver")

add_executable(${PROJECT_NAME} main.c)
target_link_libraries(${PROJECT_NAME} m)
