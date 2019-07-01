cmake_minimum_required(VERSION 3.0)

# NOTICE 请使用CUDA8来编译！

project(KinectFusionApp)

# Use modern C++
set(CMAKE_CXX_STANDARD 14)

set(PROJECT_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(PROJECT_SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)

# ------------------------------------------------
#                 Dependencies
# ------------------------------------------------
## CUDA
find_package(CUDA 8.0 REQUIRED)
IF (CUDA_FOUND)
    include_directories("${CUDA_INCLUDE_DIRS}")
    message(STATUS "==> KinectFusionApp Found CUDA： " ${CUDA_INCLUDE_DIRS})
    # Optional: Specify the arch of your CUDA hardware here
    # SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-std=c++11 -gencode arch=compute_20,code=sm_21)
    SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-std=c++11 -gencode arch=compute_61,code=sm_61 --expt-relaxed-constexpr)
ENDIF ()

## OpenCV
# Optional: Set OpenCV_DIR if you want to use a custom version of OpenCV
# SET("OpenCV_DIR" "/opt/opencv/share/OpenCV")
SET("OpenCV_DIR" "/home/guoqing/libraries/opencv_320/share/OpenCV")

find_package(OpenCV 3.0 REQUIRED)
if (OpenCV_INCLUDE_DIRS)
    include_directories("${OpenCV_INCLUDE_DIRS}")
    message(STATUS "==> KinectFusionApp Found OpenCV： " ${OpenCV_INCLUDE_DIRS})
    link_directories(${OpenCV_DIR}/lib)
endif (OpenCV_INCLUDE_DIRS)

## Eigen3
SET("EIGEN3_INCLUDE_DIR" "/home/guoqing/libraries/eigen_333_cuda8/include/eigen3")
# find_package(Eigen3 REQUIRED)
if (EIGEN3_INCLUDE_DIR)
    include_directories("${EIGEN3_INCLUDE_DIR}")
    message(STATUS "==> KinectFusionApp Found Eigen3： " ${EIGEN3_INCLUDE_DIR})    
endif (EIGEN3_INCLUDE_DIR)

## OpenNI2
find_package(openni2)
if (OPENNI2_FOUND)
    include_directories("${OPENNI2_INCLUDE_DIR}")
    message(STATUS "==> KinectFusionApp Found OpenNI2： " ${OPENNI2_INCLUDE_DIR})    
endif (OPENNI2_FOUND)

## Intel RealSense
find_package(realsense2 REQUIRED)
if (realsense2_FOUND)
    set("realsense_INCLUDE_DIR"   "/usr/include/librealsense2")
    include_directories("${realsense_INCLUDE_DIR}")
    message(STATUS "==> KinectFusionApp Found realsense2 " ${realsense_INCLUDE_DIR})    
endif (realsense2_FOUND)
# ------------------------------------------------

include_directories(${PROJECT_INCLUDE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/KinectFusionLib/include)

file(GLOB KinectFusionApp_SRCS ${PROJECT_SOURCE_DIR}/*.cpp)

add_executable(KinectFusionApp ${KinectFusionApp_SRCS})
add_dependencies(KinectFusionApp KinectFusion)
target_link_libraries(KinectFusionApp ${OpenCV_LIBS}  ${OPENNI2_LIBRARY} ${realsense2_LIBRARY} KinectFusion )


