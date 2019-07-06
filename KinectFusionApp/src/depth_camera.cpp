
/**
 * @file depth_camera.cpp
 * @author guoqing (1337841346@qq.com)
 * @brief 实现深度相机接口类
 * @version 0.1
 * @date 2019-07-04
 * 
 * @copyright Copyright (c) 2019
 * 
 */

// 自己的头文件
#include <depth_camera.h>

// C++ STD LIB
#include <iostream>
#include <fstream>
#include <iomanip>

// 临时禁用警告
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Weffc++"
#include <PS1080.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv/cv.hpp>
#pragma GCC diagnostic pop

// =============================================### Pseudo ###
// 构造函数,参数是深度图也是配置文件的存放路径
PseudoCamera::PseudoCamera(const std::string& _data_path) :
        // C++11 中的花括号初始化
        data_path{_data_path}, cam_params{}, 
        current_index{0}                                        //注意这个成员变量将会被初始化成为0
{
    // 从已经写死的文件中读取相机参数,包括:
    // * 图像大小    w  h
    // * 焦距       fx fy
    // * 光心偏差    cx cy
    std::ifstream cam_params_stream { data_path + "seq_cparam.txt" };
    if (!cam_params_stream.is_open())
        throw std::runtime_error{"Camera parameters could not be read"};
    cam_params_stream >> cam_params.image_width >> cam_params.image_height;
    cam_params_stream >> cam_params.focal_x >> cam_params.focal_y;
    cam_params_stream >> cam_params.principal_x >> cam_params.principal_y;
};

// 虚拟相机获取图像
InputFrame PseudoCamera::grab_frame () const
{
    // step 1 生成访问路径
    std::stringstream depth_file;
    depth_file << data_path << "seq_depth" << std::setfill('0') << std::setw(5) << current_index << ".png";
    std::stringstream color_file;
    color_file << data_path << "seq_color" << std::setfill('0') << std::setw(5) << current_index << ".png";

    // step 2 获取深度图像并且检查
    InputFrame frame {};
    cv::imread(depth_file.str(), -1).convertTo(frame.depth_map, CV_32FC1);
    // 如果为空说明已经到头了,然后从0重新生成
    if (frame.depth_map.empty()) {  // When this happens, we reached the end of the recording and have to
                                    // start at 0 again
        current_index = 0;
        depth_file = std::stringstream {};
        color_file = std::stringstream {};
        depth_file << data_path << "seq_depth" << std::setfill('0') << std::setw(5) << current_index << ".png";
        color_file << data_path << "seq_color" << std::setfill('0') << std::setw(5) << current_index << ".png";
        frame.depth_map = cv::imread(depth_file.str(), -1);
    }

    // step 3 获取彩色图像
    frame.color_map = cv::imread(color_file.str());

    // step 4 修改状态并且返回
    ++current_index;

    return frame;
}

// 获取相机参数
CameraParameters PseudoCamera::get_parameters() const
{
    return cam_params;
}

// =============================================### Asus Xtion PRO LIVE ###=============================================
// 手上没有设备,也没有机会接触,所以注释简略一点

// 构造函数,生成Xtion深度相机对象
XtionCamera::XtionCamera() :
        device{}, depthStream{}, colorStream{}, depthFrame{},
        colorFrame{}, cam_params{}
{
    // step 1 初始化设备并枚举设备
    openni::OpenNI::initialize();

    openni::Array<openni::DeviceInfo> deviceInfoList;
    openni::OpenNI::enumerateDevices(&deviceInfoList);

    std::cout << deviceInfoList.getSize() << std::endl;
    for (int i = 0; i < deviceInfoList.getSize(); ++i) {
        std::cout << deviceInfoList[i].getName() << ", "
                  << deviceInfoList[i].getVendor() << ", "
                  << deviceInfoList[i].getUri() << ", "
                  << std::endl;
    }

    // step 2 尝试打开其中的任何一个设备
    auto ret = device.open(openni::ANY_DEVICE);
    if (ret != openni::STATUS_OK)
        throw std::runtime_error{"OpenNI device could not be opened"};

    // step 3 创建深度图像流和彩色图像流
    // 创建图像流之前首先要生成视频模式的配置结构体
    openni::VideoMode depthMode;
    depthMode.setResolution(640, 480);
    depthMode.setFps(30);
    depthMode.setPixelFormat(openni::PIXEL_FORMAT_DEPTH_1_MM);

    openni::VideoMode colorMode;
    colorMode.setResolution(640, 480);
    colorMode.setFps(30);
    colorMode.setPixelFormat(openni::PIXEL_FORMAT_RGB888);

    // 创建流
    depthStream.create(device, openni::SENSOR_DEPTH);
    depthStream.setVideoMode(depthMode);
    depthStream.start();

    colorStream.create(device, openni::SENSOR_COLOR);
    colorStream.setVideoMode(colorMode);

    // step 4 配置摄像头,使能自动曝光控制和自动白平衡控制,操作后配置立即生效
    openni::CameraSettings *cameraSettings = colorStream.getCameraSettings();
    cameraSettings->setAutoExposureEnabled(true);
    cameraSettings->setAutoWhiteBalanceEnabled(true);

    // step 5 重新获取当前摄像头的配置信息
    cameraSettings = colorStream.getCameraSettings();
    if (cameraSettings != nullptr) {
        std::cout << "Camera Settings" << std::endl;
        std::cout << " Auto Exposure Enabled      : " << cameraSettings->getAutoExposureEnabled() << std::endl;
        std::cout << " Auto WhiteBalance Enabled  : " << cameraSettings->getAutoWhiteBalanceEnabled() << std::endl;
        std::cout << " Exposure                   : " << cameraSettings->getExposure() << std::endl;
        std::cout << " Gain                       : " << cameraSettings->getGain() << std::endl;
    }

    // step 6 启用彩色图像数据流,并且使能两项功能:
    // 1. 彩色图和深度图进行时间同步
    // 2. 彩色图和深度图进行配准
    colorStream.start();

    if (device.setDepthColorSyncEnabled(true) != openni::STATUS_OK) {
        std::cout << "setDepthColorSyncEnabled is disabled" << std::endl;
    }
    if (device.setImageRegistrationMode(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR) != openni::STATUS_OK) {
        std::cout << "setImageRegistrationMode is disabled" << std::endl;
    }

    // step 7 计算相机内参,主要是计算焦距

    double pixelSize;
    // ? XN_STREAM_PROPERTY_ZERO_PLANE_PIXEL_SIZE
    depthStream.getProperty<double>(XN_STREAM_PROPERTY_ZERO_PLANE_PIXEL_SIZE, &pixelSize);

    // pixel size @ VGA = pixel size @ SXGA x 2
    pixelSize *= 2.0; // in mm

    // focal length of IR camera in pixels for VGA resolution
    int zeroPlaneDistance; // in mm
    depthStream.getProperty(XN_STREAM_PROPERTY_ZERO_PLANE_DISTANCE, &zeroPlaneDistance);

    // ? 感觉 baseline 拿到之后并没有起到什么作用啊
    double baseline;
    depthStream.getProperty<double>(XN_STREAM_PROPERTY_EMITTER_DCMOS_DISTANCE, &baseline);
    baseline *= 10.0;

    // focal length from mm -> pixels (valid for 640x480)
    double depthFocalLength_VGA = (int) (static_cast<double>(zeroPlaneDistance) / pixelSize);

    CameraParameters cp {};
    cp.image_width = depthStream.getVideoMode().getResolutionX();
    cp.image_height = depthStream.getVideoMode().getResolutionY();
    cp.focal_x = cp.focal_y = (float) depthFocalLength_VGA;
    cp.principal_x = cp.image_width / 2 - 0.5f;
    cp.principal_y = cp.image_height / 2 - 0.5f;

    cam_params = cp;
}

// 获取 RGB-D 图像对
InputFrame XtionCamera::grab_frame() const
{
    // step 1 从深度图像流和彩色图像流中获取帧对象,并且检查其有效性
    depthStream.readFrame(&depthFrame);
    colorStream.readFrame(&colorFrame);

    if (!depthFrame.isValid() || depthFrame.getData() == nullptr ||
        !colorFrame.isValid() || colorFrame.getData() == nullptr) {
        throw std::runtime_error{"Frame data retrieval error"};
    } else {
        // step 2 生成彩色图和深度图, 并调整格式
        // 深度图
        cv::Mat depthImg16U { depthStream.getVideoMode().getResolutionY(),
                              depthStream.getVideoMode().getResolutionX(),
                              CV_16U,
                              static_cast<char*>(const_cast<void*>(depthFrame.getData())) };
        cv::Mat depth_image;
        // 转换为 CV_32FC1 格式, 并且把图像翻转一下
        depthImg16U.convertTo(depth_image, CV_32FC1);
        cv::flip(depth_image, depth_image, 1);

        // 彩色图
        cv::Mat color_image { colorStream.getVideoMode().getResolutionY(),
                              colorStream.getVideoMode().getResolutionX(),
                              CV_8UC3,
                              static_cast<char*>(const_cast<void*>(colorFrame.getData())) };
        // 转换成为 RGB 颜色通道编码, 并且把图像翻转一下
        cv::cvtColor(color_image, color_image, cv::COLOR_BGR2RGB);
        cv::flip(color_image, color_image, 1);

        // step 3 返回 RGB-D 图像对
        return InputFrame { depth_image, color_image };
    }
}

// 获取相机内参
CameraParameters XtionCamera::get_parameters() const
{
    return cam_params;
}

// =============================================### Intel RealSense ###=============================================
// 无参构造函数, 用于生成一个真实的 RealSence 实例
RealSenseCamera::RealSenseCamera() : pipeline{}
{
    // Explicitly enable depth and color stream, with these constraints:
    // Same dimensions and color stream has format BGR 8bit

    // step 1 生成配置对象
    rs2::config configuration {};
    configuration.disable_all_streams();
    // 原来的配置不支持我们实验室使用的D435
    // configuration.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_BGR8, 30);
    // configuration.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_Z16, 30);

    // 这个能够适配我们实验室使用的D435
    configuration.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    configuration.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);

    // step 2 获取设备列表并且选择默认设备
    // Use the first detected device, if any
    rs2::context ctx {};
    auto devices = ctx.query_devices();

    if(devices.size() == 0)
        throw std::runtime_error { "No RealSense device detected" };

    {
        auto device = devices[0]; // The device handle defined here is invalid after pipeline.start()

        // Print info about the device
        std::cout << "Using this RealSense device:" << std::endl;
        for ( int info_idx = 0; info_idx < static_cast<int>(RS2_CAMERA_INFO_COUNT); ++info_idx ) {
            auto info_type = static_cast<rs2_camera_info>(info_idx);
            std::cout << "  " << std::left << std::setw(20) << info_type << " : ";
            if ( device.supports(info_type))
                std::cout << device.get_info(info_type) << std::endl;
            else
                std::cout << "Not supported" << std::endl;
        }
    }

    // step 3 配置驱动的管线 pipeline
    pipeline.start(configuration);


    // step 4 获取相机内参 -- 指的是深度相机的内参
    // Get depth sensor intrinsics
    auto streams = pipeline.get_active_profile().get_streams();
    for(const auto& stream : streams) {
        if(stream.stream_type() == RS2_STREAM_DEPTH) {
            auto intrinsics = stream.as<rs2::video_stream_profile>().get_intrinsics();
            cam_params.focal_x = intrinsics.fx;
            cam_params.focal_y = intrinsics.fy;
            cam_params.image_height = intrinsics.height;
            cam_params.image_width = intrinsics.width;
            cam_params.principal_x = intrinsics.ppx;
            cam_params.principal_y = intrinsics.ppy;

            // // DEBUG
            // std::cout<<"====================================="<<std::endl;
            // std::cout<<"focal_x="<<cam_params.focal_x<<std::endl;
            // std::cout<<"focal_y="<<cam_params.focal_y<<std::endl;
            // std::cout<<"image_height="<<cam_params.image_height<<std::endl;
            // std::cout<<"image_width="<<cam_params.image_width<<std::endl;
            // std::cout<<"principal_x="<<cam_params.principal_x<<std::endl;
            // std::cout<<"principal_y="<<cam_params.principal_y<<std::endl;
            // std::cout<<"====================================="<<std::endl;

        }
    }

    // step 5 获取深度值和真实尺度之间的缩放倍数
    // Get depth scale which is used to convert the measurements into millimeters
    depth_scale = pipeline.get_active_profile().get_device().first<rs2::depth_sensor>().get_depth_scale();
}

// 有参数的构造函数, 给定的参数是"回放文件"的路径(类似于数据集)
RealSenseCamera::RealSenseCamera(const std::string& filename) : pipeline{}
{
    // step 1 从回放文件生成配置对象并启动管线 pipeline
    // ? enable_device_from_file
    rs2::config configuration {};
    configuration.disable_all_streams();
    configuration.enable_device_from_file(filename);
    pipeline.start(configuration);

    // step 2 获取相机内参
    auto streams = pipeline.get_active_profile().get_streams();
    for(const auto& stream : streams) {
        if(stream.stream_type() == RS2_STREAM_DEPTH) {
            auto intrinsics = stream.as<rs2::video_stream_profile>().get_intrinsics();
            cam_params.focal_x = intrinsics.fx;
            cam_params.focal_y = intrinsics.fy;
            cam_params.image_height = intrinsics.height;
            cam_params.image_width = intrinsics.width;
            cam_params.principal_x = intrinsics.ppx;
            cam_params.principal_y = intrinsics.ppy;
        }
    }

    // step 3 获取深度值和真实尺度之间的缩放倍数
    // Get depth scale which is used to convert the measurements into millimeters
    depth_scale = pipeline.get_active_profile().get_device().first<rs2::depth_sensor>().get_depth_scale();
}

// 获取RGB-D图像对
InputFrame RealSenseCamera::grab_frame() const
{
    // step 1 获取深度图像帧和彩色图像帧
    auto data = pipeline.wait_for_frames();
    auto depth = data.get_depth_frame();
    auto color = data.get_color_frame();

    // step 2 从深度图像帧和彩色图像帧中读取深度图像和彩色图像,然后进行格式调整
    // 深度图
    cv::Mat depth_image { cv::Size { cam_params.image_width,
                                     cam_params.image_height },
                          CV_16UC1,                                         // 原始数据的格式是这样的
                          const_cast<void*>(depth.get_data()),
                          cv::Mat::AUTO_STEP};

    cv::Mat converted_depth_image;
    depth_image.convertTo(converted_depth_image, CV_32FC1, depth_scale * 1000.f);

    // 彩色图
    cv::Mat color_image { cv::Size { cam_params.image_width,
                                     cam_params.image_height },
                          CV_8UC3,
                          const_cast<void*>(color.get_data()),
                          cv::Mat::AUTO_STEP};

    return InputFrame {
            converted_depth_image,
            color_image
    };
}

// 获取相机的内参数结构体
CameraParameters RealSenseCamera::get_parameters() const
{
    return cam_params;
}



// ### Kinect ###
/*
KinectCamera::KinectCamera()
{

}

KinectCamera::~KinectCamera()
{

}

InputFrame KinectCamera::grab_frame() const
{

}

CameraParameters KinectCamera::get_parameters() const
{

}
*/
