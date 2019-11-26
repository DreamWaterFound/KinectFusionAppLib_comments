// This is the CPU part of the ICP implementation
// Author: Christian Diller, git@christian-diller.de

#include <kinectfusion.h>

using Matf31da   = Eigen::Matrix<float, 3, 1, Eigen::DontAlign>;
// 后缀 rm = Row Major
using Matrix3frm = Eigen::Matrix<float, 3, 3, Eigen::RowMajor>;

namespace kinectfusion {
    namespace internal {

        namespace cuda { // Forward declare CUDA functions
            void estimate_step(const Eigen::Matrix3f& rotation_current, const Matf31da& translation_current,
                               const cv::cuda::GpuMat& vertex_map_current, const cv::cuda::GpuMat& normal_map_current,
                               const Eigen::Matrix3f& rotation_previous_inv, const Matf31da& translation_previous,
                               const CameraParameters& cam_params,
                               const cv::cuda::GpuMat& vertex_map_previous, const cv::cuda::GpuMat& normal_map_previous,
                               float distance_threshold, float angle_threshold,
                               Eigen::Matrix<double, 6, 6, Eigen::RowMajor>& A, Eigen::Matrix<double, 6, 1>& b);
        }

        // 估计相机的位姿，这里是CPU的部分
        bool pose_estimation(
            Eigen::Matrix4f& pose,                  // 输入: 上一帧的相机位姿; 输出: 当前帧得到的相机位姿
            const FrameData& frame_data,            // 当前帧中的数据(顶点图+法向图)
            const ModelData& model_data,            // 上一帧对Global TSDF Model 进行表面推理得到的表面模型数据(Vertex Map + Normal Map)
            const CameraParameters& cam_params,     // 相机的内参
            const int pyramid_height,               // 金字塔的图层数目
            const float distance_threshold,         // ICP 过程中视为外点的距离阈值
            const float angle_threshold,            // ICP 过程中视为外点的角度阈值
            const std::vector<int>& iterations)     // 每一个图层上的 ICP 迭代次数
        {
            // step 0 数据准备
            // Get initial rotation and translation
            // 其实就是得到的上一帧的相机旋转和平移, 如果是放在迭代过程中看的话, 其实就是在进行第一次迭代之前, 相机的位姿
            Eigen::Matrix3f current_global_rotation    = pose.block(0, 0, 3, 3);        // Rwc
            Eigen::Vector3f current_global_translation = pose.block(0, 3, 3, 1);        // twc

            // !这里的求逆其实可以直接求转置的.... 速度更快
            // 上一帧相机的旋转, 外参表示, 可以将世界坐标系下的点转换到相机坐标系下
            Eigen::Matrix3f previous_global_rotation_inverse(current_global_rotation.inverse());    // Rcw
            // 上一帧相机的平移, 相机光心在世界坐标系下的位置
            Eigen::Vector3f previous_global_translation = pose.block(0, 3, 3, 1);                   // twc

            // step 1 ICP loop, from coarse to sparse
            // 对于每个图层, 这里是从顶层开始的
            for (int level = pyramid_height - 1; level >= 0; --level) {
                // 进行每一次迭代
                for (int iteration = 0; iteration < iterations[level]; ++iteration) {
                    // step 1.0 数据准备
                    // 矩阵A大小: 6x6, 并且是斜对称矩阵
                    Eigen::Matrix<double, 6, 6, Eigen::RowMajor> A {};
                    // 向量b的大小为 6x1
                    Eigen::Matrix<double, 6, 1> b {};

                    // step 1.1 Estimate one step on the CPU 
                    // 说是CPU, 但是计算矩阵 A 和向量 b 的过程是使用GPU计算的
                    cuda::estimate_step(
                        current_global_rotation,                    // 上次迭代得到的旋转, Rwc
                        current_global_translation,                 // 上次迭代得到的平移, twc
                        frame_data.vertex_pyramid[level],           // 当前帧对应图层的的顶点图
                        frame_data.normal_pyramid[level],           // 当前帧对应图层的的法向图
                        previous_global_rotation_inverse,           // 上一帧相机外参中的旋转的逆, Rcw
                        previous_global_translation,                // 上一帧相机的平移, twc 
                        cam_params.level(level),                    // 当前图层的相机内参
                        model_data.vertex_pyramid[level],           // 对应图层的推理得到的平面顶点图
                        model_data.normal_pyramid[level],           // 对应图层的推理得到的平面法向图
                        distance_threshold,                         // ICP迭代过程中视为外点的距离阈值
                        sinf(angle_threshold * 3.14159254f / 180.f),// ICP迭代过程中视为外点的角度阈值(角度变正弦值)
                        A, b);                                      // 计算得到的矩阵 A 和向量 b

                    // step 1.2 Solve equation to get alpha, beta and gamma
                    // 进行合法性验证,要求矩阵A的行列式不能够小于某个阈值
                    double det = A.determinant();
                    // ? 为什么这里给出的阈值这么大呢
                    if (fabs(det) < 100000 /*1e-15*/ || std::isnan(det))
                        return false;

                    // step 1.3 计算增量, 这里的增量应该说是相对于上次迭代计算的相机位姿来说的
                    // fullPivLu: 进行LU分解
                    // solve:     求解当非齐次项为 b 的时候, 方程 Ax = b的解
                    Eigen::Matrix<float, 6, 1> result { A.fullPivLu().solve(b).cast<float>() };
                    
                    // step 1.3.1 计算旋转的增量
                    // 首先需要从方程 Ax=b 中的b中,恢复出旋转矩阵的三个参数alpha, beta和gamma 
                    float alpha = result(0);
                    float beta = result(1);
                    float gamma = result(2);
                    // Update rotation -- 恢复出原始的旋转增量
                    auto camera_rotation_incremental(
                            Eigen::AngleAxisf(gamma, Eigen::Vector3f::UnitZ()) *
                            Eigen::AngleAxisf(beta, Eigen::Vector3f::UnitY()) *
                            Eigen::AngleAxisf(alpha, Eigen::Vector3f::UnitX()));

                    // step 1.3.1 计算平移的增量
                    auto camera_translation_incremental = result.tail<3>();

                    // step 1.4 计算在世界坐标系下当前帧相机的平移和旋转
                    // Update translation
                    // 符号约定:
                    // current_global_translation,  current_global_rotation => Tw1 = | Rw1 tw1 |
                    //                                                               |  0   1  |
                    //
                    // camera_translation_incremental, camera_rotation_incremental => T12 = | R12 t12 |
                    //                                                                      |  0   1  |
                    // 最终的结果为 Tw2 = | Rw2 tw2 |
                    //                  |  0   1  |
                    // 下面可以按照这样理解:
                    //  Tw2 = T12*Tw1 = | R12 t12 | | Rw1 tw1 | = | R12*Rw1 R12*tw1 + t12 |
                    //                  |  0   1  | |  0   1  | = |   0           1     |
                    // 所以 tw2 = R12*tw1 + t12
                    //     Rw2 = R12*Rw1
                    // ? 但是不明白, 为什么不是 Tw2 = Tw1*T12 ?

                    current_global_translation =
                            camera_rotation_incremental * current_global_translation + camera_translation_incremental;
                    current_global_rotation = camera_rotation_incremental * current_global_rotation;
                }
            }

            // Step 2 Return the new pose
            pose.block(0, 0, 3, 3) = current_global_rotation;
            pose.block(0, 3, 3, 1) = current_global_translation;

            return true;
        }
    }
}