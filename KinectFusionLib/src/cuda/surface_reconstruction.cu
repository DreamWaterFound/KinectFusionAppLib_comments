// Performs surface reconstruction, i.e. updates the internal volume with data from the current frame
// This is CUDA code; compile with nvcc
// Author: Christian Diller, git@christian-diller.de

#include "include/common.h"

// 一个二维的Eigen向量, 不强制使用字节对齐
using Vec2ida = Eigen::Matrix<int, 2, 1, Eigen::DontAlign>;

namespace kinectfusion {
    namespace internal {
        namespace cuda {

            // 更新 TSDF 模型的核函数
            __global__
            void update_tsdf_kernel(
                const PtrStepSz<float> depth_image,                         // 原始大小深度图
                const PtrStepSz<uchar3> color_image,                        // 原始大小彩色图
                PtrStepSz<short2> tsdf_volume, 
                PtrStepSz<uchar3> color_volume,
                int3 volume_size, 
                float voxel_scale,
                CameraParameters cam_params,                                // 原始图层上的相机内参
                const float truncation_distance,
                Eigen::Matrix<float, 3, 3, Eigen::DontAlign> rotation,      // 旋转矩阵 -- 这里要求Eigen编译的时候使能cuda
                Vec3fda translation)                                        // 平移向量
            {
                // step 1 获取当前线程的id, 并检查是否落在 volume 中.
                // 这里实际上是每个线程对应(x,y,*),每一个线程负责处理z轴上的所有数据
                const int x = blockIdx.x * blockDim.x + threadIdx.x;
                const int y = blockIdx.y * blockDim.y + threadIdx.y;
                // 合法性检查
                if (x >= volume_size.x || y >= volume_size.y)
                    return;

                // step 2 处理z轴上的每一个体素的数据
                for (int z = 0; z < volume_size.z; ++z) {
                    // step 2.1 计算该体素中心点在当前帧相机坐标系下的坐标, 然后投影到图像中得到投影点坐标, 其中进行合法性检查
                    // 获取当前要处理的体素中心点在空间中的实际位置. 其中的0.5表示的是计算得到体素的中心, * voxel_scale 对应为实际空间尺度下体素的中心
                    const Vec3fda position((static_cast<float>(x) + 0.5f) * voxel_scale,
                                           (static_cast<float>(y) + 0.5f) * voxel_scale,
                                           (static_cast<float>(z) + 0.5f) * voxel_scale);
                    // 将上面的在世界坐标系下的表示变换到在当前相机坐标系下的坐标
                    const Vec3fda camera_pos = rotation * position + translation;
                    // 合法性检查1: 如果这个体素相机看不到那么我们就不管了
                    if (camera_pos.z() <= 0)
                        continue;

                    // int __float2int_rn(float) : 求最近的偶数 // ?  为什么要求偶数? -- 我怀疑作者写错了, 这里应该是求整数吧
                    // 计算空间点在图像上的投影点, 并且认为这个投影点就是对这个空间体素的观测
                    const Vec2ida uv(
                            __float2int_rn(camera_pos.x() / camera_pos.z() * cam_params.focal_x + cam_params.principal_x),
                            __float2int_rn(camera_pos.y() / camera_pos.z() * cam_params.focal_y + cam_params.principal_y));
                    // 合法性检查2: 查看投影点是否正确地投影在了图像范围内
                    if (uv.x() < 0 || uv.x() >= depth_image.cols || uv.y() < 0 || uv.y() >= depth_image.rows)
                        continue;
                    // 获取该体素中心点的深度的观测值(相对于当前图像来说)
                    const float depth = depth_image.ptr(uv.y())[uv.x()];
                    // 合法性检查3: 深度的观测值应该非负
                    if (depth <= 0)
                        continue;

                    // step 2.2 计算论文中公式7的 lambda
                    // Tips:             | 1/fx   0    -cx/fx | 
                    //          K^(-1) = |  0   1/fy   -cy/fy |
                    //                   |  0     0       1   |
                    const Vec3fda xylambda(
                            (uv.x() - cam_params.principal_x) / cam_params.focal_x,             // (x/z)
                            (uv.y() - cam_params.principal_y) / cam_params.focal_y,             // (y/z)
                            1.f);
                    // 计算得到公式7中的 lambda
                    const float lambda = xylambda.norm();

                    // step 2.3 计算 SDF, 参考论文公式6中括号的部分
                    // 这里的 camera_pos 已经是该体素中心点在当前世界坐标系下的坐标了, 论文中的公式的意思是计算相机光心到该点的距离, 就相当于这个坐标直接取欧式二范数
                    // 前面乘的负号是因为, 咱们定义 SDF 表示中平面前的部分为正, 平面后的部分为负
                    // SDF 其实也就是表示了空间体素点的(预测值 - 观测值)
                    const float sdf = (-1.f) * ((1.f / lambda) * camera_pos.norm() - depth);

                    // step 2.4 计算 TSDF, 参考论文公式9
                    // 如果根据我们得到的 SDF 告诉我们, 这个距离我们能够观测到 (即相当于观测的距离是在 -u 之前的)
                    //0---TSDF------------------------             |
                    //          |                    |\            |
                    //          |                    | \           |
                    //          |                    |  \          |
                    //          |                    |   \         |
                    //          |                    |    \        |
                    //          |                    |     \       |
                    //0---camera|-------------------(u)----(0)----(-u)------------distance
                    //          |                    |       \     |
                    //          |                    |        \    |
                    //          |                    |         \   |
                    //          |                    |          \  |
                    //          |                    |           \ |
                    //          |                    |            \|
                    //          |                    |             |-x-x-x-x-x-x-x-x-x-x-
                    //          |      截断区         |  TSDF表示区  |    不可观测区      
                    //
                    if (sdf >= -truncation_distance) {
                        // 说明当前的 SDF 表示获得的观测的深度值, 在我们构建的TSDF模型中, 是可观测的
                        // step 2.4.1 计算当前次观测得到的 TSDF 值
                        // 注意这里的 TSDF 值一直都是小于1的. 后面会利用这个特点来将浮点型的 TSDF 值保存为 uint16_t 类型
                        const float new_tsdf = fmin(1.f, sdf / truncation_distance);

                        // step 2.4.2 获取当前的global模型中已经存储的 TSDF 值和权重
                        // 有点z对应行, y对应列的感觉
                        // volume 的下标组织: 二维 GpuMat
                        //0             ---x0  x1  x2  x3  x4 ... x511 ----------->(x)
                        //   (  z0,  y0)|
                        //   (  z0,  y1)|
                        //   (  z0,  y2)|
                        //   (  z0,  y3)|
                        //   (  z0, ...)|
                        //   (  z0,y511)|
                        //   (  z1,  y0)|
                        //   (  z1,  y1)|
                        //   (  z1,  y2)|
                        //   (  z1,  y3)|
                        //   (  z1, ...)|
                        //   (  z1,y511)|
                        //   ( ..., ...)|
                        //   (z511,  y0)|
                        //   (z511,  y1)|
                        //   (z511,  y2)|
                        //   (z511,  y3)|
                        //   (z511, ...)|
                        //   (z511,y511)|
                        //              ^
                        // 获取对应的 TSDF 体素中已经存储的 TSDF 值和权重 (注意获取的数据是个向量)
                        short2 voxel_tuple = tsdf_volume.ptr(z * volume_size.y + y)[x];

                        // 这里的 current 表示已经存储在全局模型中的数据, 对应在论文公式11中为下标(k-1)的符号 
                        // 由于TSDF值现在是按照 uint16_t 的格式来存储的,所以为了变换成为float型需要进行变换, 乘DIVSHORTMAX
                        // 使用乘法代替除法, 运算更快. 这个部分可以参考下面 浮点型 TSDF值是怎么存储为 uint16_t 格式的
                        const float current_tsdf = static_cast<float>(voxel_tuple.x) * DIVSHORTMAX;
                        const int current_weight = voxel_tuple.y;

                        // step 2.4.3 更新 TSDF 值和权重值
                        // 见下
                        const int add_weight = 1;
                        
                        // 参考论文公式11, 计算得到该体素中更新后的 TSDF 值, 符号对应关系如下:
                        // current_weight => W_{k-1}(p)
                        // current_tsdf   => F_{k-1}(p)
                        // add_weight     => W_{R_k}(p)
                        // new_tsdf       => F_{R_k}(p)
                        const float updated_tsdf = (current_weight * current_tsdf + add_weight * new_tsdf) /
                                                   (current_weight + add_weight);
                        // 论文公式 13 对权重 进行更新
                        const int new_weight = min(current_weight + add_weight, MAX_WEIGHT);
                        // 将 浮点的 TSDF 值经过 int32_t 类型 保存为 uint16_t 类型. 限幅是因为理想情况下 无论是当前帧计算的还是融合之后的TSDF值都应该是小于1的
                        // (所以对应的值属于 -SHORTMAX ~ SHORTMAX)
                        //  类型中转是因为不这样做, updated_tsdf 一旦越界会出现截断, 导致 min max 函数都无法有效工作
                        const int new_value  = max(-SHORTMAX, min(SHORTMAX, static_cast<int>(updated_tsdf * SHORTMAX)));

                        // step 2.4.4 保存计算结果
                        tsdf_volume.ptr(z * volume_size.y + y)[x] = make_short2(static_cast<short>(new_value),
                                                                                static_cast<short>(new_weight));

                        // step 2.4.5 对 彩色图进行更新
                        // 前提是当前的这个体素的中心观测值在 TSDF 的1/2未截断区域内. 注意这里的约束其实更加严格, 这里是截断距离除了2
                        if (sdf <= truncation_distance / 2 && sdf >= -truncation_distance / 2) {
                            // step 2.4.5.1 获取当前体素对应的投影点的颜色的观测值和之前的储存值
                            // 储存值
                            uchar3& model_color = color_volume.ptr(z * volume_size.y + y)[x];
                            // 观测值
                            const uchar3 image_color = color_image.ptr(uv.y())[uv.x()];

                            // step 2.4.5.2 颜色均匀化之后再写入, 仿照 TSDF 值的加权更新方式
                            model_color.x = static_cast<uchar>(
                                    (current_weight * model_color.x + add_weight * image_color.x) /
                                    (current_weight + add_weight));
                            model_color.y = static_cast<uchar>(
                                    (current_weight * model_color.y + add_weight * image_color.y) /
                                    (current_weight + add_weight));
                            model_color.z = static_cast<uchar>(
                                    (current_weight * model_color.z + add_weight * image_color.z) /
                                    (current_weight + add_weight));
                        }// 对彩色图进行更新
                    }// 如果根据我们得到的 SDF 告诉我们, 这个距离我们能够观测到, 那么更新 TSDF
                }// 处理z轴上的每一个体素的数据
            }// 核函数

            // 实现表面的重建, 即将当前帧的相机位姿已知的时候, 根据当前帧的surface mearsurment,融合到Global TSDF Model 中
            // 主机端函数
            void surface_reconstruction(const cv::cuda::GpuMat& depth_image, const cv::cuda::GpuMat& color_image,
                                        VolumeData& volume,
                                        const CameraParameters& cam_params, const float truncation_distance,
                                        const Eigen::Matrix4f& model_view)
            {
                // step 1 根据TSDF Volume的大小, 计算核函数的大小
                const dim3 threads(32, 32);
                const dim3 blocks((volume.volume_size.x + threads.x - 1) / threads.x,
                                  (volume.volume_size.y + threads.y - 1) / threads.y);

                // step 2 启动核函数
                update_tsdf_kernel<<<blocks, threads>>>(
                    depth_image,                            // 原始大小的深度图像
                    color_image,                            // 原始大小的彩色图像
                    volume.tsdf_volume,                     // TSDF Volume, GpuMat
                    volume.color_volume,                    // color Volume, GpuMat
                    volume.volume_size,                     // Volume 的大小, int3
                    volume.voxel_scale,                     // 尺度缩放, float
                    cam_params,                             // 在当前图层上的相机内参
                    truncation_distance,                    // 截断距离u
                    model_view.block(0, 0, 3, 3),           // 提取旋转矩阵
                                                            // (Index startRow, Index startCol, Index blockRows, Index blockCols)
                    model_view.block(0, 3, 3, 1));          // 提取平移向量

                // step 3 等待所有的并行线程结束
                cudaThreadSynchronize();
            }
        }
    }
}
