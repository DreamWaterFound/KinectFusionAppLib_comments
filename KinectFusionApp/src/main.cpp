
/**
 * @file main.cpp
 * @author guoqing (1337841346@qq.com)
 * @brief KinectFusionApp 运行的主文件
 * @version 0.1
 * @date 2019-07-01
 * 
 * @copyright Copyright (c) 2019
 * 
 */

// KinectFusionApp 自身的依赖
#include <kinectfusion.h>
#include <depth_camera.h>
#include <util.h>

// C++ 标准库
#include <iostream>
#include <fstream>
#include <iomanip>


// 为了整洁，忽略在导入 opencv 的时候产生的警告
#pragma GCC diagnostic push                 // 忽略警告的代码段 -- 开始
#pragma GCC diagnostic ignored "-Wall"      // 设置忽略的警告的类型
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Weffc++"
#include <opencv2/highgui.hpp>
#pragma GCC diagnostic pop                  // 忽略警告的代码段 -- 结束

// C++ 命令行参数解析工具
#include <cxxopts.hpp>
// C++ 解析 toml 格式配置文件的工具
#include <cpptoml.h>

// 数据集的存放目录
std::string data_path {};
// 录制的topic名称
std::string recording_name {};

/**
 * @brief 根据配置文件来生成KinectFusionLib所需要的配置信息
 * @param[in] toml_config 配置文件的root表
 * @return auto KinectFusionLib的配置对象
 */
auto make_configuration(const std::shared_ptr<cpptoml::table>& toml_config)
{
    kinectfusion::GlobalConfiguration configuration;

    // NOTICE cpptoml only supports int64_t, so we need to explicitly cast to int to suppress the warning
    // 三个维度上的体素尺寸大小
    auto volume_size_values = *toml_config->get_qualified_array_of<int64_t>("kinectfusion.volume_size");
    configuration.volume_size = make_int3(static_cast<int>(volume_size_values[0]),
                                          static_cast<int>(volume_size_values[1]),
                                          static_cast<int>(volume_size_values[2]));
    // 一个体素表示的真实环境中的mm数
    configuration.voxel_scale = static_cast<float>(*toml_config->get_qualified_as<double>("kinectfusion.voxel_scale"));

    // 双边滤波器的核尺寸. 下面的这些参数和 cv::cuda::bilateralFilter() 函数的参数是一一对应的
    configuration.bfilter_kernel_size = *toml_config->get_qualified_as<int>("kinectfusion.bfilter_kernel_size");
    // 颜色空间中的方差
    // ? 为什么会有这一项? 为什么需要对彩色图像也进行双边滤波?
    // 这个应该是"值域核"滤波使用的方差吧
    configuration.bfilter_color_sigma  = static_cast<float>(*toml_config->get_qualified_as<double>("kinectfusion.bfilter_color_sigma"));
    // 深度图对应的空间中使用的方差 -- 空间核滤波使用的方差
    configuration.bfilter_spatial_sigma  = static_cast<float>(*toml_config->get_qualified_as<double>("kinectfusion.bfilter_spatial_sigma"));

    // 指的是第一帧的时候相机光心距离体素块中心点的距离
    configuration.init_depth  = static_cast<float>(*toml_config->get_qualified_as<double>("kinectfusion.init_depth"));
    // 是否每一帧都要从显存中获取当前模型
    configuration.use_output_frame = *toml_config->get_qualified_as<bool>("kinectfusion.use_output_frame");
    // TSDF中会发生截断的距离
    configuration.truncation_distance  = static_cast<float>(*toml_config->get_qualified_as<double>("kinectfusion.truncation_distance"));
    // 认为超过了一定距离后的物体是背景 // ? 但是现在我也不是非常确定
    configuration.depth_cutoff_distance  = static_cast<float>(*toml_config->get_qualified_as<double>("kinectfusion.depth_cutoff_distance"));
    // 处理每一层的时候使用的金字塔层数
    configuration.num_levels  = *toml_config->get_qualified_as<int>("kinectfusion.num_levels");
    // ? 不是很明白,貌似随着时间的进行,某个东西会增多,这个就是进行上线阈值的设定
    configuration.triangles_buffer_size  = *toml_config->get_qualified_as<int>("kinectfusion.triangles_buffer_size");
    // ? 同上. 但是感觉KF的过程并不需要进行点云的存储啊, 而且怎么控制啊
    configuration.pointcloud_buffer_size  = *toml_config->get_qualified_as<int>("kinectfusion.pointcloud_buffer_size");

    // 如果关联到的点的深度阈值差超过了这个阈值,那么它们就会被认为是outlier,不会参与ICP过程 (mm) (参考原始论文)
    configuration.distance_threshold  = static_cast<float>(*toml_config->get_qualified_as<double>("kinectfusion.distance_threshold"));
    // 和上面相似,如果关联到的点的法向量的角度差超过了这个阈值, 就认为是outlier,不会参与ICP过程 (deg) (参考原始论文)
    configuration.angle_threshold  = static_cast<float>(*toml_config->get_qualified_as<double>("kinectfusion.angle_threshold"));
    // 金字塔每一层进行ICP迭代的次数
    auto icp_iterations_values = *toml_config->get_qualified_array_of<int64_t>("kinectfusion.icp_iterations");
    configuration.icp_iterations = {icp_iterations_values.begin(), icp_iterations_values.end()};

    return configuration;
}

/**
 * @brief 根据配置文件中的信息来生成 camera 对象
 * @param[in] toml_config toml文件中的一个配置表，这里给的参数应该是root
 * @return auto 返回得到的 DepthCmaera 对象
 */
auto make_camera(const std::shared_ptr<cpptoml::table>& toml_config)
{
    // 基类
    std::unique_ptr<DepthCamera> camera;

    // 获取要使用的深度相机类型（这个看 toml 配置文件）
    const auto camera_type = *toml_config->get_qualified_as<std::string>("camera.type");
    // 对于不同的相机构造不同的类型
    if (camera_type == "Pseudo") {                                                  // 虚拟相机（说白了就是跑数据集）
        std::stringstream source_path {};                                           // 生成数据集的路径
        source_path << data_path << "source/" << recording_name << "/"; 
        camera = std::make_unique<PseudoCamera>(source_path.str());                 // 构造
    } else if (camera_type == "Xtion") {                                            // 华硕的 Xtion
        camera = std::make_unique<XtionCamera>();                   
    } else if (camera_type == "RealSense") {                                        // 牙膏厂的 RealSense
        if(*toml_config->get_qualified_as<bool>("camera.realsense.live")) {         // 如果是需要实时图像,就构造吧
            camera = std::make_unique<RealSenseCamera>();
        } else {
            std::stringstream source_file {};                                       // 如果不是需要实时图像那么就读取bag文件
            source_file << data_path << "source/" << recording_name << ".bag";
            camera = std::make_unique<RealSenseCamera>(source_file.str());          // 然后调用这个构造函数
        }
    } else {                                                                        // 如果都不是? 那么就报个错
        throw std::logic_error("There is no implementation for the camera type you specified.");
    }

    // 返回构造或者是生成的相机对象
    return camera;
}

/**
 * @brief 主循环
 * @param[in] camera            相机模型
 * @param[in] configuration     配置结构体
 */
void main_loop(const std::unique_ptr<DepthCamera> camera, const kinectfusion::GlobalConfiguration& configuration)
{
    // step 1 生成 pipeline
    kinectfusion::Pipeline pipeline { camera->get_parameters(), configuration };

    // step 2 生成用于显示的窗口
    // 这tm竟然是个OpenCV的窗口...
    cv::namedWindow("Pipeline Output");

    // step 3 main loop
    // 通过一个标志位 end 控制
    for (bool end = false; !end;) {
        // step 3.1 Get frame (depth + color)
        InputFrame frame = camera->grab_frame();

        // step 3.2 Process frame
        bool success = pipeline.process_frame(frame.depth_map, frame.color_map);
        if (!success)
            std::cout << "Frame could not be processed" << std::endl;

        // step 3.3 Display the output
        // 这里调用函数得到的是一个固定视角下的渲染出来的图像
        cv::imshow("Pipeline Output", pipeline.get_last_model_frame());

        // step 3.4 处理键盘事件
        switch (cv::waitKey(1)) {
            // 按下 a 键, 表示保存相机的位姿和重建的mesh模型,然后退出
            case 'a': { // Save all available data
                std::cout << "Saving all ..." << std::endl;
                // 保存相机的位姿
                std::cout << "Saving poses ..." << std::endl;
                // 一个 T 矩阵序列
                auto poses = pipeline.get_poses();

                // 卧槽, 每一个相机位姿都单独的存储在一个 .txt 文件中
                for (size_t i = 0; i < poses.size(); ++i) {
                    std::stringstream file_name {};
                    file_name << data_path << "poses/" << recording_name << "/seq_pose" << std::setfill('0')
                              << std::setw(5) << i << ".txt";
                    std::ofstream { file_name.str() } << poses[i] << std::endl;
                }

                // 提取重建的 mesh 模型, 首先从GPU导出,然后保存为 .ply 文件
                std::cout << "Extracting mesh ..." << std::endl;
                auto mesh = pipeline.extract_mesh();
                std::cout << "Saving mesh ..." << std::endl;
                std::stringstream file_name {};
                file_name << data_path << "meshes/" << recording_name << ".ply";
                kinectfusion::export_ply(file_name.str(), mesh);
                end = true;
                break;
            }
            // 如果 p 被按下,就只保存相机的位姿,然后退出
            case 'p': { // Save poses only
                std::cout << "Saving poses ..." << std::endl;
                auto poses = pipeline.get_poses();

                for (size_t i = 0; i < poses.size(); ++i) {
                   std::stringstream file_name {};
                    file_name << data_path << "poses/" << recording_name << "/seq_pose" << std::setfill('0')
                              << std::setw(5) << i << ".txt";
                    std::ofstream { file_name.str() } << poses[i] << std::endl;
                }
                end = true;
                break;
            }
            // 如果 m 被按下,就只保存重建的结果,然后退出
            case 'm': { // Save mesh only
                std::cout << "Extracting mesh ..." << std::endl;
                auto mesh = pipeline.extract_mesh();
                std::cout << "Saving mesh ..." << std::endl;
                std::stringstream file_name {};
                file_name << data_path << "meshes/" << recording_name << ".ply";
                kinectfusion::export_ply(file_name.str(), mesh);
                end = true;
                break;
            }
            // 如果 空格 被按下,就什么都不保存,直接退出
            case ' ': // Save nothing
                end = true;
                break;
            default:
                break;
        }// 处理键盘事件
    }// 主循环

    // HERE
    // 接下来可以开始看的入口:
    // 1. KinectFusion pipeline 的构造函数
    // 2. KinectFusion的处理函数
    // 3. 获取相机位姿的接口函数
    // 4. 获取重建的mesh模型的接口
    // 5. 将mesh模型转换成为 ply文件的接口
}

/** @brief 设置和选择CUDA设备 */
void setup_cuda_device()
{
    // NOTICE 注意这里使用的竟然是OpenCV提供的CUDA接口
    // 获取设备计数
    auto n_devices = cv::cuda::getCudaEnabledDeviceCount();
    std::cout << "Found " << n_devices << " CUDA devices" << std::endl;
    // 遍历每一个CUDA设备，并且获取其内存大小
    for (int device_idx = 0; device_idx < n_devices; ++device_idx) {
        cv::cuda::DeviceInfo info { device_idx };
        std::cout << "Device #" << device_idx << ": " << info.name()
                  << " with " << info.totalMemory() / 1048576 << "MB total memory" << std::endl;
    }

    // Hardcoded to first device; change if necessary
    // 这里是写死在程序中了，默认就使用第一个CUDA设备
    std::cout << "Using device #0" << std::endl;
    cv::cuda::setDevice(0);
}

/**
 * @brief 程序入口
 * @param[in] argc argc
 * @param[in] argv argv
 * @return int     运行状态
 */
int main(int argc, char* argv[])
{
    // Parse command line options
    // step 1 解析命令行参数
    // 构造解析对象并且生成对当前程序的描述
    cxxopts::Options options { "KinectFusionApp",
                               "Sample application for KinectFusionLib, a modern implementation of the KinectFusion approach"};
    // 只添加这一个参数
    options.add_options()("c,config", "Configuration filename", cxxopts::value<std::string>());
    // 解析参数
    auto program_arguments = options.parse(argc, argv);
    // 如果没给定就抛出一个异常
    if (program_arguments.count("config") == 0)
        throw std::invalid_argument("You have to specify a path to the configuration file");

    // Parse TOML configuration file
    // step 2 解析 TOML 配置文件
    // 指定要解析的配置文件名
    auto toml_config = cpptoml::parse_file(program_arguments["config"].as<std::string>());
    // 数据集所在的目录
    data_path = *toml_config->get_as<std::string>("data_path");
    // 疑似录制的topic名称
    recording_name = *toml_config->get_as<std::string>("recording_name");

    // Print info about available CUDA devices and specify device to use
    // step 3 设置CUDA设备
    setup_cuda_device();

    // Start the program's main loop
    // step 4 进入主循环
    main_loop(
            make_camera(toml_config),               // 生成相机模型
            make_configuration(toml_config)         // 读取配置
    );

    return EXIT_SUCCESS;
}
