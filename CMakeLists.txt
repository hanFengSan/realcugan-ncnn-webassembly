project(realcugan-ncnn-webassembly)

cmake_minimum_required(VERSION 3.10)

option(LOCAL_DEBUG "debug" OFF)

set(CMAKE_BUILD_TYPE release)

if(NOT WASM_FEATURE)
    message(FATAL_ERROR "You must pass cmake option -DWASM_FEATURE and possible values are basic, simd, threads and simd-threads")
endif()

add_subdirectory(fmt)
add_subdirectory(ncnn)

if (NOT LOCAL_DEBUG)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -s FORCE_FILESYSTEM=1  -sALLOW_MEMORY_GROWTH -s EXIT_RUNTIME=1")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s FORCE_FILESYSTEM=1  -sALLOW_MEMORY_GROWTH -s EXIT_RUNTIME=1")
    set(CMAKE_EXECUTBLE_LINKER_FLAGS "${CMAKE_EXECUTBLE_LINKER_FLAGS} -s FORCE_FILESYSTEM=1  -sALLOW_MEMORY_GROWTH -s EXIT_RUNTIME=1")

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -sEXPORTED_FUNCTIONS=['_process_image','_free'] --preload-file ${CMAKE_CURRENT_SOURCE_DIR}/models@. --use-preload-cache")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -sEXPORTED_FUNCTIONS=['_process_image','_free'] --preload-file ${CMAKE_CURRENT_SOURCE_DIR}/models@. --use-preload-cache")
    set(CMAKE_EXECUTBLE_LINKER_FLAGS "${CMAKE_EXECUTBLE_LINKER_FLAGS} -sEXPORTED_FUNCTIONS=['_process_image','_free'] --preload-file ${CMAKE_CURRENT_SOURCE_DIR}/models@. --use-preload-cache")

    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif()

if(${WASM_FEATURE} MATCHES "threads")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -pthread -s USE_PTHREADS=1 -s PTHREAD_POOL_SIZE=navigator.hardwareConcurrency")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -pthread -s USE_PTHREADS=1 -s PTHREAD_POOL_SIZE=navigator.hardwareConcurrency")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fopenmp -pthread -s USE_PTHREADS=1 -s PTHREAD_POOL_SIZE=navigator.hardwareConcurrency")
endif()

add_executable(realcugan-ncnn-webassembly-${WASM_FEATURE} main.cpp realcugan.cpp)
set_property(TARGET realcugan-ncnn-webassembly-${WASM_FEATURE} PROPERTY CXX_STANDARD 11)
target_link_libraries(realcugan-ncnn-webassembly-${WASM_FEATURE} ncnn fmt)
