cmake_minimum_required(VERSION 3.13)
project(lab1_convolution)
set(CMAKE_CXX_STANDARD 11)

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})

add_executable(lab1_convolution main.cpp)
target_link_libraries(lab1_convolution ${OpenCL_LIBRARY})
