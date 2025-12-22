cmake_minimum_required(VERSION 3.15)
project(RISCVGpuSim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS src/*.cpp)

# --- LLVM via llvm-config ---
find_program(LLVM_CONFIG NAMES llvm-config PATHS /usr/local/bin ENV PATH)
if(NOT LLVM_CONFIG)
    message(FATAL_ERROR "llvm-config not found. Ensure LLVM is installed and in PATH.")
endif()

# Get LLVM flags
execute_process(
    COMMAND ${LLVM_CONFIG} --cxxflags
    OUTPUT_VARIABLE LLVM_CXXFLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
    COMMAND ${LLVM_CONFIG} --ldflags
    OUTPUT_VARIABLE LLVM_LDFLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
    COMMAND ${LLVM_CONFIG} --libs all
    OUTPUT_VARIABLE LLVM_LIBS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
    COMMAND ${LLVM_CONFIG} --system-libs
    OUTPUT_VARIABLE LLVM_SYSTEM_LIBS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Convert flags into CMake lists
separate_arguments(LLVM_CXXFLAGS_LIST NATIVE_COMMAND "${LLVM_CXXFLAGS}")
separate_arguments(LLVM_LDFLAGS_LIST NATIVE_COMMAND "${LLVM_LDFLAGS}")
separate_arguments(LLVM_LIBS_LIST NATIVE_COMMAND "${LLVM_LIBS}")
separate_arguments(LLVM_SYSTEM_LIBS_LIST NATIVE_COMMAND "${LLVM_SYSTEM_LIBS}")

# --- Dependencies ---
add_library(cxxopts INTERFACE)
target_include_directories(cxxopts INTERFACE ${CMAKE_SOURCE_DIR}/include)

add_library(elfio INTERFACE)
target_include_directories(elfio INTERFACE ${CMAKE_SOURCE_DIR}/include/elfio)

# --- Build executable ---
add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(
    ${PROJECT_NAME} 
    PRIVATE 
    ${CMAKE_SOURCE_DIR}/src
)

target_compile_options(${PROJECT_NAME} PRIVATE ${LLVM_CXXFLAGS_LIST})
target_compile_options(${PROJECT_NAME} PRIVATE -fexceptions)
target_link_options(${PROJECT_NAME} PRIVATE ${LLVM_LDFLAGS_LIST})

target_link_libraries(${PROJECT_NAME}
    PRIVATE cxxopts
    PRIVATE elfio
    PRIVATE ${LLVM_LIBS_LIST}
    PRIVATE ${LLVM_SYSTEM_LIBS_LIST}
)