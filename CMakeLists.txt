cmake_minimum_required(VERSION 3.29)

include(CheckLanguage)

check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    add_definitions(-DUSE_CUDA)    

    include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
else()
    message("Failed to find CUDA Compiler")
endif()

project(CUDA_PATH_TRACER LANGUAGES CUDA CXX)

# Finds the required Dependencies
find_package(GLEW REQUIRED)
find_package(FreeGLUT REQUIRED)
find_package(OpenGL REQUIRED)
find_package(cxxopts REQUIRED)


# Includes the required dependencies
include_directories(${GLEW_INCLUDE_DIR})
include_directories(${FreeGLUT_INCLUDE_DIR})
include_directories(${OPENGL_INCLUDE_DIR})
include_directories(${CXXOPTS_INCLUDE_DIR})

set(HEADERS
    src/Array.h
    src/BVH.h
    src/BVHNode.h
    src/Camera.h
    src/CudaBVH.h
    src/CudaRenderKernel.h
    src/Geometry.h
    src/HDRloader.h
    src/MouseKeyboardInput.h
    src/Mesh.h
    src/SceneLoader.h
    src/Sort.h
    src/SplitBVHBuilder.h
    src/Util.h
    src/cutil_math.h
    src/linear_math.h
    src/Application.h
    src/handlerror.h
)

set(SOURCES
    src/BVH.cpp
    src/BVHNode.cpp
    src/Camera.cpp
    src/CudaBVH.cpp
    src/HDRloader.cpp
    src/main.cpp
    src/renderkernel.cu
    src/SceneLoader.cpp
    src/Sort.cpp
    src/SplitBVHBuilder.cpp
    src/Util.cpp
    src/Application.cpp
)

add_executable(CUDA_PT ${HEADERS} ${SOURCES})
set_target_properties(CUDA_PT PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_features(CUDA_PT PUBLIC cxx_std_20)
target_compile_features(CUDA_PT PUBLIC cuda_std_20)

target_link_libraries(CUDA_PT
    ${GLEW_LIBRARIES}
    ${FreeGLUT_INCLUDE_DIR}
    $<IF:$<TARGET_EXISTS:FreeGLUT::freeglut>,FreeGLUT::freeglut,FreeGLUT::freeglut_static>
    ${OPENGL_LIBRARIES}
    cxxopts::cxxopts
)

add_custom_command(
    TARGET CUDA_PT POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/data
            ${CMAKE_CURRENT_BINARY_DIR}/data
)
