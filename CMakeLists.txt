cmake_minimum_required(VERSION 3.10)

project(mnn-yolov7)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

find_package(OpenCV REQUIRED)

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/mnn/include)

link_directories(${CMAKE_SOURCE_DIR}/mnn/libs)

add_executable(mnn-yolov7 main.cpp yolov7.cpp)
target_link_libraries(mnn-yolov7 MNN ${OpenCV_LIBS})
