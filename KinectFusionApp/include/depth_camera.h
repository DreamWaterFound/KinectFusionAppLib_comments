/** @brief 相机模型类和配置类的声明 */

#ifndef KINECTFUSION_CAMERA_H
#define KINECTFUSION_CAMERA_H

/*
 * Camera class declarations. Add your own camera handler by deriving from DepthCamera.
 * Author: Christian Diller
 */

// 这里包含了这个头文件,主要是因为下面的程序中使用到了结构体 CameraParameters
#include <data_types.h>

// 禁止警告
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wunused-variable"
#include <OpenNI.h>
#pragma GCC diagnostic pop

// libRealSense 支持
#include <librealsense2/rs.hpp>

using kinectfusion::CameraParameters;

/**
 * @brief 
 * Represents a single input frame
 * Packages a depth map with the corresponding RGB color map
 * The depth map is expected to hold float values, the color map 8 bit RGB values
 */
struct InputFrame {
    cv::Mat_<float> depth_map;
    cv::Mat_<cv::Vec3b> color_map;
};

/**
 * @brief Models the interface to a device that provides raw depth images
 */
class DepthCamera {
public:
    virtual ~DepthCamera() = default;

    virtual InputFrame grab_frame() const = 0;
    virtual CameraParameters get_parameters() const = 0;
};

/**
 * @brief For testing purposes. This camera simply loads depth frames stored on disk.
 */
class PseudoCamera : public DepthCamera {
public:
    // explicit 关键字用于只有一个实际必须的参数的类的构造函数,用于避免隐式的类型转换
    /**
     * @brief 构造函数
     * @param[in] _data_path 深度图，也是配置文件的存放路径
     */
    explicit PseudoCamera(const std::string& _data_path);
    // override 关键字表示将会要在当前类中重新覆盖实现该函数
    /** @brief 析构函数,虽然是重写,但是还是默认的 */
    ~PseudoCamera() override = default;

    /**
     * @brief  获取RGB-D
     * @return InputFrame RGB-D帧对
     */
    InputFrame grab_frame() const override;
    /**
     * @brief  获取相机内参结构体
     * @return CameraParameters 相机内参结构体
     */
    CameraParameters get_parameters() const override;

private:
    std::string data_path;              // 数据文件的存放路径
                                        // data_path/seq_cparam.txt         配置文件
                                        // data_path/seq_depth/<index>.png  深度图像
                                        // data_path/seq_color/<index>.png  彩色图像
    CameraParameters cam_params;        // 相机内参结构体
    mutable size_t current_index;       // 当前已经读取的照片的id,构造函数中将会被初始化为0
};

/**
 * @brief Provides depth frames acquired by a Asus Xtion PRO LIVE camera.
 */
class XtionCamera : public DepthCamera {
public:
    XtionCamera();
    ~XtionCamera() override = default;

    InputFrame grab_frame() const override;

    CameraParameters get_parameters() const override;

private:
    openni::Device device;
    
    // mutable 用于在 const 修饰的类成员函数中突破不能修改的限制
    // ref: https://blog.csdn.net/starlee/article/details/1430387

    mutable openni::VideoStream depthStream;
    mutable openni::VideoStream colorStream;
    mutable openni::VideoFrameRef depthFrame;
    mutable openni::VideoFrameRef colorFrame;

    CameraParameters cam_params;
};

/*
 * @brief Provides depth frames acquired by an Intel Realsense camera.
 */
class RealSenseCamera : public DepthCamera {
public:
    RealSenseCamera();
    RealSenseCamera(const std::string& filename);

    ~RealSenseCamera() override = default;

    InputFrame grab_frame() const override;

    CameraParameters get_parameters() const override;

private:
    rs2::pipeline pipeline;
    CameraParameters cam_params;

    float depth_scale;
};


/*
 * Provides depth frames acquired by a Microsoft Kinect camera.
 */
/*
class KinectCamera : public DepthCamera {
public:
    KinectCamera();

    ~KinectCamera();

    InputFrame grab_frame() const override;

    CameraParameters get_parameters() const override;
};
 */

#endif //KINECTFUSION_CAMERA_H
