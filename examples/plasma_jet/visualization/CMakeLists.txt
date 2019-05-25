cmake_minimum_required(VERSION 3.10)

project(plasma_jet_visualizer VERSION 1.0.0)
set(PROJECT_DESCRIPTION "OpenGL based visualization for the plasma jet example")

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

include_directories(${GLUT_INCLUDE_DIRS})
add_executable(${PROJECT_NAME} main.c)
target_link_libraries(${PROJECT_NAME} m ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})
