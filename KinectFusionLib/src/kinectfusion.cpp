// This is the KinectFusion Pipeline Implementation
// Author: Christian Diller, git@christian-diller.de

#include <kinectfusion.h>

#include <fstream>

using cv::cuda::GpuMat;

namespace kinectfusion {

    Pipeline::Pipeline(const CameraParameters _camera_parameters,
                       const GlobalConfiguration _configuration) :
            //  生成参数对象
            camera_parameters(_camera_parameters), configuration(_configuration),
            // 设置volume
            volume(_configuration.volume_size, _configuration.voxel_scale),
            // 设置模型数据
            model_data(_configuration.num_levels, _camera_parameters),
            // 初始化数据: 清空位姿,轨迹等
            current_pose{}, poses{}, frame_id{0}, last_model_frame{}
    {
        // The pose starts in the middle of the cube, offset along z by the initial depth
        // 第一帧的相机位姿设置在 Volume 的中心, 然后在z轴上拉远一点
        current_pose.setIdentity();
        current_pose(0, 3) = _configuration.volume_size.x / 2 * _configuration.voxel_scale;
        current_pose(1, 3) = _configuration.volume_size.y / 2 * _configuration.voxel_scale;
        current_pose(2, 3) = _configuration.volume_size.z / 2 * _configuration.voxel_scale - _configuration.init_depth;
    }

    // 每一帧的数据处理都要调用这个函数
    // HERE
    bool Pipeline::process_frame(const cv::Mat_<float>& depth_map, const cv::Mat_<cv::Vec3b>& color_map)
    {
        // STEP 1: Surface measurement
        // 主要工作：
        // 1、对输入的深度图像计算金字塔
        // 2、对金字塔的每一层图像进行双边滤波
        // 3、对金字塔的每一层深度图像计算顶点图
        // 4、对金字塔的每一层顶点图计算法向图
        internal::FrameData frame_data = internal::surface_measurement(
            depth_map,                                          // 输入的深度图像
            camera_parameters,                                  // 相机的内参数
            configuration.num_levels,                           // 金字塔层数
            configuration.depth_cutoff_distance,                // 视为背景, 不进行重建的点的距离
            configuration.bfilter_kernel_size,                  // 双边滤波器使用的核大小
            configuration.bfilter_color_sigma,                  // 值域滤波的方差
            configuration.bfilter_spatial_sigma);               // 空间域滤波的方差
        // 将原始彩色图上传到显存中
        frame_data.color_pyramid[0].upload(color_map);

        // STEP 2: Pose estimation
        // 表示icp过程是否成功的变量
        bool icp_success { true };
        if (frame_id > 0) { // Do not perform ICP for the very first frame
            // 不在第一帧进行位姿估计
            icp_success = internal::pose_estimation(
                current_pose,                                   // 输入: 上一帧的相机位姿; 输出: 当前帧得到的相机位姿
                frame_data,                                     // 当前帧的彩色图/深度图/顶点图/法向图数据
                model_data,                                     // 上一帧图像输入后, 推理出的平面模型，使用顶点图、法向图来表示
                camera_parameters,                              // 相机内参
                configuration.num_levels,                       // 金字塔层数
                configuration.distance_threshold,               // icp 匹配过程中视为 outlier 的距离差
                configuration.angle_threshold,                  // icp 匹配过程中视为 outlier 的角度差 (deg)
                configuration.icp_iterations);                  // icp 过程的迭代次数
        }
        // 如果 icp 过程不成功, 那么就说明当前失败了
        if (!icp_success)
            // icp失败之后本次处理退出,但是上一帧推理的得到的平面将会一直保持, 每次新来一帧都会重新icp后一直都在尝试重新icp, 尝试重定位回去
            return false;
        // 记录当前帧的位姿
        poses.push_back(current_pose);

        // STEP 3: Surface reconstruction
        // 进行表面重建的工作, 其实是是将当前帧的观测信息融合到Global Volume
        internal::cuda::surface_reconstruction(
            frame_data.depth_pyramid[0],                        // 金字塔底层的深度图像
            frame_data.color_pyramid[0],                        // 金字塔底层的彩色图像
            volume,                                             // Global Volume
            camera_parameters,                                  // 相机内参
            configuration.truncation_distance,                  // 阶段距离u
            current_pose.inverse());                            // 相机外参 -- 其实这里可以加速的, 直接对Eigen::Matrix4f求逆有点耗时间

        // Step 4: Surface prediction
        // 在当前帧的位姿上得到对表面的推理结果
        // 从下到上依次遍历图像金字塔
        for (int level = 0; level < configuration.num_levels; ++level)
            // 对每层图像的数据都进行表面的推理
            internal::cuda::surface_prediction(
                volume,                                         // Global Volume
                model_data.vertex_pyramid[level],               // 推理得到的平面的顶点图
                model_data.normal_pyramid[level],               // 推理得到的平面的法向图 
                model_data.color_pyramid[level],                // 推理得到的彩色图
                camera_parameters.level(level),                 // 当前图层的相机内参
                configuration.truncation_distance,              // 截断距离
                current_pose);                                  // 当前时刻的相机位姿(注意没有取逆)

        // Step 5 如果需要显示模型, 则从GPU中获取
        if (configuration.use_output_frame) // Not using the output will speed up the processing
            // 注意这里是直接将这个模型填充到了彩色帧的底层
            model_data.color_pyramid[0].download(last_model_frame);

        // 帧id++
        ++frame_id;

        // 这一帧正确处理了
        return true;
    }

    cv::Mat Pipeline::get_last_model_frame() const
    {
        if (configuration.use_output_frame)
            return last_model_frame;

        return cv::Mat(1, 1, CV_8UC1);
    }

    std::vector<Eigen::Matrix4f> Pipeline::get_poses() const
    {
        for (auto pose : poses)
            pose.block(0, 0, 3, 3) = pose.block(0, 0, 3, 3).inverse();
        return poses;
    }

    PointCloud Pipeline::extract_pointcloud() const
    {
        PointCloud cloud_data = internal::cuda::extract_points(volume, configuration.pointcloud_buffer_size);
        return cloud_data;
    }

    SurfaceMesh Pipeline::extract_mesh() const
    {
        SurfaceMesh surface_mesh = internal::cuda::marching_cubes(volume, configuration.triangles_buffer_size);
        return surface_mesh;
    }

    void export_ply(const std::string& filename, const PointCloud& point_cloud)
    {
        std::ofstream file_out { filename };
        if (!file_out.is_open())
            return;

        file_out << "ply" << std::endl;
        file_out << "format ascii 1.0" << std::endl;
        file_out << "element vertex " << point_cloud.num_points << std::endl;
        file_out << "property float x" << std::endl;
        file_out << "property float y" << std::endl;
        file_out << "property float z" << std::endl;
        file_out << "property float nx" << std::endl;
        file_out << "property float ny" << std::endl;
        file_out << "property float nz" << std::endl;
        file_out << "property uchar red" << std::endl;
        file_out << "property uchar green" << std::endl;
        file_out << "property uchar blue" << std::endl;
        file_out << "end_header" << std::endl;

        for (int i = 0; i < point_cloud.num_points; ++i) {
            float3 vertex = point_cloud.vertices.ptr<float3>(0)[i];
            float3 normal = point_cloud.normals.ptr<float3>(0)[i];
            uchar3 color = point_cloud.color.ptr<uchar3>(0)[i];
            file_out << vertex.x << " " << vertex.y << " " << vertex.z << " " << normal.x << " " << normal.y << " "
                     << normal.z << " ";
            file_out << static_cast<int>(color.x) << " " << static_cast<int>(color.y) << " "
                     << static_cast<int>(color.z) << std::endl;
        }
    }

    void export_ply(const std::string& filename, const SurfaceMesh& surface_mesh)
    {
        std::ofstream file_out { filename };
        if (!file_out.is_open())
            return;

        file_out << "ply" << std::endl;
        file_out << "format ascii 1.0" << std::endl;
        file_out << "element vertex " << surface_mesh.num_vertices << std::endl;
        file_out << "property float x" << std::endl;
        file_out << "property float y" << std::endl;
        file_out << "property float z" << std::endl;
        file_out << "property uchar red" << std::endl;
        file_out << "property uchar green" << std::endl;
        file_out << "property uchar blue" << std::endl;
        file_out << "element face " << surface_mesh.num_triangles << std::endl;
        file_out << "property list uchar int vertex_index" << std::endl;
        file_out << "end_header" << std::endl;

        for (int v_idx = 0; v_idx < surface_mesh.num_vertices; ++v_idx) {
            float3 vertex = surface_mesh.triangles.ptr<float3>(0)[v_idx];
            uchar3 color = surface_mesh.colors.ptr<uchar3>(0)[v_idx];
            file_out << vertex.x << " " << vertex.y << " " << vertex.z << " ";
            file_out << (int) color.z << " " << (int) color.y << " " << (int) color.x << std::endl;
        }

        for (int t_idx = 0; t_idx < surface_mesh.num_vertices; t_idx += 3) {
            file_out << 3 << " " << t_idx + 1 << " " << t_idx << " " << t_idx + 2 << std::endl;
        }
    }
}