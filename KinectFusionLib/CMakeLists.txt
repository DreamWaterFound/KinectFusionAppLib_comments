cmake_minimum_required(VERSION 3.0)

project(KinectFusionLib)

# Use modern C++
set(CMAKE_CXX_STANDARD 14)

# Setting output paths
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

# ------------------------------------------------
#                 Dependencies
# ------------------------------------------------
## CUDA
find_package(CUDA 8.0 REQUIRED)
IF (CUDA_FOUND)
    include_directories("${CUDA_INCLUDE_DIRS}")
    message(STATUS "==> KinectFusionLib Found CUDA： " ${CUDA_INCLUDE_DIRS})
    # Optional: Specify the arch of your CUDA hardware here
    # NOTICE 需要修改arch为自己显卡的架构代码；--expt-relaxed-constexpr来启用Eigen对CUDA的兼容
    SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-std=c++11 -gencode arch=compute_61,code=sm_61 --expt-relaxed-constexpr)
ENDIF ()

## OpenCV
# Optional: Set OpenCV_DIR if you want to use a custom version of OpenCV
SET("OpenCV_DIR" "/home/guoqing/libraries/opencv_320/share/OpenCV")

find_package(OpenCV 3.0 REQUIRED)
if (OpenCV_INCLUDE_DIRS)
    include_directories("${OpenCV_INCLUDE_DIRS}")
    message(STATUS "==> KinectFusionLib Found OpenCV： " ${OpenCV_INCLUDE_DIRS})
    link_directories(${OpenCV_DIR}/lib)
endif (OpenCV_INCLUDE_DIRS)

## Eigen3
# find_package(Eigen3 REQUIRED)
# NOTICE
# 注意选择的Eigen库还是使用编译的时候带CUDA支持的吧否则指定报错
SET("EIGEN3_INCLUDE_DIR" "/home/guoqing/libraries/eigen_333_cuda8/include/eigen3")
if (EIGEN3_INCLUDE_DIR)
    include_directories("${EIGEN3_INCLUDE_DIR}")
    message(STATUS "==> KinectFusionLib Found Eigen3： " ${EIGEN3_INCLUDE_DIR})    
endif (EIGEN3_INCLUDE_DIR)
# ------------------------------------------------

SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Weffc++")

# Targets
set(PROJECT_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(PROJECT_SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)

include_directories("${PROJECT_INCLUDE_DIR}" "${PROJECT_SOURCE_DIR}/cuda/include")
file(GLOB KinectFusion_SRCS "${PROJECT_SOURCE_DIR}/cuda/*.cu" "${PROJECT_SOURCE_DIR}/*.cpp")
# file(GLOB KinectFusion_SRCS  "${PROJECT_SOURCE_DIR}/*.cpp" "${PROJECT_SOURCE_DIR}/cuda/*.cu")

cuda_add_library(KinectFusion STATIC "${KinectFusion_SRCS}")
target_link_libraries(KinectFusion "${OpenCV_LIBS}")
