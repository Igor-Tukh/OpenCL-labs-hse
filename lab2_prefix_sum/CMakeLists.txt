cmake_minimum_required(VERSION 3.14)
project(lab2_prefix_sum)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})

add_executable(lab2_prefix_sum main.cpp)
target_link_libraries(lab2_prefix_sum ${OpenCL_LIBRARY})