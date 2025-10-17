cmake_minimum_required(VERSION 3.15)

project(RISCVGpuSim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB SOURCES src/*.cpp)

add_library(cxxopts INTERFACE)
target_include_directories(cxxopts INTERFACE include)

add_library(elfio INTERFACE)
target_include_directories(elfio INTERFACE include/elfio)

find_package(PkgConfig REQUIRED)
pkg_check_modules(CAPSTONE REQUIRED capstone)

include_directories(${CAPSTONE_INCLUDE_DIRS})
link_directories(${CAPSTONE_LIBRARY_DIRS})

add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(RISCVGpuSim 
                        PRIVATE cxxopts
                        PRIVATE elfio
                        PUBLIC ${CAPSTONE_LIBRARIES})