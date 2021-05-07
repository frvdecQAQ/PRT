cmake_minimum_required(VERSION 3.16)

project(PRT)

include_directories(include)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Release")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -lcufft)

aux_source_directory(src DIR_SRCS)

find_package(CUDA REQUIRED)

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("Found OpenMP")
    set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS})
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS})
    set(CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS})
endif()

find_package(CUDA)

find_package(OpenGL REQUIRED)
message("${OPENGL_LIBRARIES}")
if(OPENGL_FOUND)
    message("Found OpenGL")
endif()
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenCV REQUIRED)
find_package(assimp REQUIRED)

set(AntTweakBar_INCLUDE_DIR "/home/aglove/tools/AntTweakBar/include/")
set(AntTweakBar_LIBRARIES "/home/aglove/tools/AntTweakBar/lib/libAntTweakBar.so")

include_directories(${OPENGL_INCLUDE_DIRS})
include_directories(${GLUT_INCLUDE_DIRS})
include_directories(${GLEW_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${AntTweakBar_INCLUDE_DIR})

cuda_add_executable(${PROJECT_NAME} ${DIR_SRCS})

CUDA_ADD_CUFFT_TO_TARGET(${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${GLEW_LIBRARIES} 
${OpenCV_LIBRARIES} ${AntTweakBar_LIBRARIES} ${ASSIMP_LIBRARIES} fftw3 fftw3f glfw)




