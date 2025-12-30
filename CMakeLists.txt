cmake_minimum_required(VERSION 3.15)
project(RISCVGpuSim LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- LLVM stuff ---
find_package(LLVM REQUIRED CONFIG)
set(LLVM_COMPONENTS
    Core
    Support
    MC
    MCParser
    MCDisassembler
    Object
    Target
    RISCV
)
llvm_map_components_to_libnames(LLVM_LIBS ${LLVM_COMPONENTS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND "${LLVM_DEFINITIONS}")
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "LLVM include dirs: ${LLVM_INCLUDE_DIRS}")
message(STATUS "LLVM libraries: ${LLVM_LIBS}")

# --- Source files ---
file(GLOB_RECURSE LIB_SOURCES CONFIGURE_DEPENDS src/*.cpp)
list(FILTER LIB_SOURCES EXCLUDE REGEX ".*main.cpp$")
set(MAIN_SOURCE src/main.cpp)

# --- Dependencies ---
add_library(cxxopts INTERFACE)
target_include_directories(cxxopts INTERFACE ${CMAKE_SOURCE_DIR}/include)

add_library(elfio INTERFACE)
target_include_directories(elfio INTERFACE ${CMAKE_SOURCE_DIR}/include/elfio)

# --- Library Target ---
add_library(gpu_sim_lib ${LIB_SOURCES})
target_include_directories(gpu_sim_lib
    PUBLIC ${CMAKE_SOURCE_DIR}/src
    PUBLIC ${LLVM_INCLUDE_DIRS}
)
target_compile_options(gpu_sim_lib PRIVATE -fexceptions)
target_compile_definitions(gpu_sim_lib PUBLIC ${LLVM_DEFINITIONS_LIST})
target_link_libraries(gpu_sim_lib
    PUBLIC cxxopts
    PUBLIC elfio
    PUBLIC ${LLVM_LIBS}
)

# --- Main Executable ---
add_executable(${PROJECT_NAME} ${MAIN_SOURCE})
target_compile_definitions(${PROJECT_NAME} PRIVATE ${LLVM_DEFINITIONS_LIST})
target_link_libraries(${PROJECT_NAME} PRIVATE gpu_sim_lib)

# --- Unit Tests ---
enable_testing()
add_executable(unit_tests
    tests/unit_tests.cpp
    tests/test_memory.cpp
    tests/test_host.cpp
    tests/test_pipeline.cpp
    tests/test_pipeline_scheduler.cpp
    tests/test_pipeline_execute.cpp
)
target_compile_definitions(${PROJECT_NAME} PRIVATE ${LLVM_DEFINITIONS_LIST})
target_link_libraries(unit_tests PRIVATE gpu_sim_lib)
