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

# find_package(CUDA REQUIRED)
find_package(GLEW REQUIRED)
find_package(FreeGLUT REQUIRED)
find_package(OpenGL REQUIRED)


# include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${GLEW_INCLUDE_DIR})
include_directories(${FreeGLUT_INCLUDE_DIR})
include_directories(${OPENGL_INCLUDE_DIR})

set(HEADERS
    Array.h
    BVH.h
    BVHNode.h
    Camera.h
    CudaBVH.h
    CudaRenderKernel.h
    Geometry.h
    HDRloader.h
    MouseKeyboardInput.h
    Scene.h
    SceneLoader.h
    Sort.h
    SplitBVHBuilder.h
    Util.h
    cutil_math.h
    linear_math.h
)

set(SOURCES
    BVH.cpp
    BVHNode.cpp
    Camera.cpp
    CudaBVH.cpp
    HDRloader.cpp
    main.cpp
    renderkernel.cu
    SceneLoader.cpp
    Sort.cpp
    SplitBVHBuilder.cpp
    Util.cpp
)

add_executable(CUDA_PT ${HEADERS} ${SOURCES})
set_target_properties(CUDA_PT PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_features(CUDA_PT PUBLIC cxx_std_20)

target_link_libraries(CUDA_PT
#    ${CUDA_LIBRARIES}
    ${GLEW_LIBRARIES}
    ${FreeGLUT_INCLUDE_DIR}
    $<IF:$<TARGET_EXISTS:FreeGLUT::freeglut>,FreeGLUT::freeglut,FreeGLUT::freeglut_static>
    ${OPENGL_LIBRARIES}
)

add_custom_command(
    TARGET CUDA_PT POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/data
            ${CMAKE_CURRENT_BINARY_DIR}/data
)
