cmake_minimum_required(VERSION 3.15)
project(RISCVGpuSim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB SOURCES CONFIGURE_DEPENDS src/*.cpp)

add_library(cxxopts INTERFACE)
target_include_directories(cxxopts INTERFACE ${CMAKE_SOURCE_DIR}/include)

add_library(elfio INTERFACE)
target_include_directories(elfio INTERFACE ${CMAKE_SOURCE_DIR}/include/elfio)

find_package(PkgConfig REQUIRED)
pkg_check_modules(CAPSTONE REQUIRED capstone)

add_executable(${PROJECT_NAME} ${SOURCES})
target_include_directories(${PROJECT_NAME} PRIVATE ${CAPSTONE_INCLUDE_DIRS})
target_link_directories(${PROJECT_NAME} PRIVATE ${CAPSTONE_LIBRARY_DIRS})

target_link_libraries(${PROJECT_NAME}
    PRIVATE cxxopts
    PRIVATE elfio
    PRIVATE ${CAPSTONE_LIBRARIES}
)