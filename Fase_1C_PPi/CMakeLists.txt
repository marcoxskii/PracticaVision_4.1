cmake_minimum_required(VERSION 3.10)
project(Fase1C_Comparison)

set(CMAKE_CXX_STANDARD 14)

# Como instalamos OpenCV en /usr/local, find_package lo encontrará automático
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV found: ${OpenCV_DIR}")
message(STATUS "OpenCV Libs: ${OpenCV_LIBS}")

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(gpu_benchmark main.cpp)
target_link_libraries(gpu_benchmark ${OpenCV_LIBS})