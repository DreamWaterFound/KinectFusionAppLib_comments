// Predicts the surface, i.e. performs raycasting
// This is CUDA code; compile with nvcc
// Author: Christian Diller, git@christian-diller.de

#include "include/common.h"

using Vec3ida = Eigen::Matrix<int, 3, 1, Eigen::DontAlign>;

namespace kinectfusion {
    namespace internal {
        namespace cuda {

            __device__ __forceinline__
            // 三线型插值
            float interpolate_trilinearly(
                const Vec3fda& point, 
                const PtrStepSz<short2>& volume,
                const int3& volume_size, 
                const float voxel_scale)
            {
                // 这个点在 Volume 下的坐标, 转换成为整数下标标的表示
                Vec3ida point_in_grid = point.cast<int>();

                // 恢复成体素中心点的坐标
                const float vx = (static_cast<float>(point_in_grid.x()) + 0.5f);
                const float vy = (static_cast<float>(point_in_grid.y()) + 0.5f);
                const float vz = (static_cast<float>(point_in_grid.z()) + 0.5f);

                // 查看原始的点的坐标是否更偏向于某个坐标轴上数值更小的一侧, 如果是的话就坐标-1; 否则不变
                // ? 为什么要减1? 为什么在数值更大的一侧就不用减?
                point_in_grid.x() = (point.x() < vx) ? (point_in_grid.x() - 1) : point_in_grid.x();
                point_in_grid.y() = (point.y() < vy) ? (point_in_grid.y() - 1) : point_in_grid.y();
                point_in_grid.z() = (point.z() < vz) ? (point_in_grid.z() - 1) : point_in_grid.z();

                // ? 为什么要 +0.5f?
                // 计算精确的(浮点型)的点坐标和整型化之后的点坐标的差
                const float a = (point.x() - (static_cast<float>(point_in_grid.x()) + 0.5f));
                const float b = (point.y() - (static_cast<float>(point_in_grid.y()) + 0.5f));
                const float c = (point.z() - (static_cast<float>(point_in_grid.z()) + 0.5f));

                return 
                    static_cast<float>(volume.ptr((point_in_grid.z()) * volume_size.y + point_in_grid.y())[point_in_grid.x()].x) * DIVSHORTMAX 
                        // volume[ x ][ y ][ z ]
                        * (1 - a) * (1 - b) * (1 - c) +
                    static_cast<float>(volume.ptr((point_in_grid.z() + 1) * volume_size.y + point_in_grid.y())[point_in_grid.x()].x) * DIVSHORTMAX 
                        // volume[ x ][ y ][z+1]
                        * (1 - a) * (1 - b) * c +
                    static_cast<float>(volume.ptr((point_in_grid.z()) * volume_size.y + point_in_grid.y() + 1)[point_in_grid.x()].x) * DIVSHORTMAX 
                        // volume[ x ][y+1][ z ]
                        * (1 - a) * b * (1 - c) +
                    static_cast<float>(volume.ptr((point_in_grid.z() + 1) * volume_size.y + point_in_grid.y() + 1)[point_in_grid.x()].x) * DIVSHORTMAX 
                        // volume[ x ][y+1][z+1]
                        * (1 - a) * b * c +
                    static_cast<float>(volume.ptr((point_in_grid.z()) * volume_size.y + point_in_grid.y())[point_in_grid.x() + 1].x) * DIVSHORTMAX 
                        // volume[x+1][ y ][ z ]
                        * a * (1 - b) * (1 - c) +
                    static_cast<float>(volume.ptr((point_in_grid.z() + 1) * volume_size.y + point_in_grid.y())[point_in_grid.x() + 1].x) * DIVSHORTMAX 
                        // volume[x+1][ y ][z+1]
                        * a * (1 - b) * c +
                    static_cast<float>(volume.ptr((point_in_grid.z()) * volume_size.y + point_in_grid.y() + 1)[point_in_grid.x() + 1].x) * DIVSHORTMAX 
                        // volume[x+1][y+1][ z ]
                        * a * b * (1 - c) +
                    static_cast<float>(volume.ptr((point_in_grid.z() + 1) * volume_size.y + point_in_grid.y() + 1)[point_in_grid.x() + 1].x) * DIVSHORTMAX 
                        // volume[x+1][y+1][z+1]
                        * a * b * c;
            }


            // __forceinline__: 强制为内联函数
            __device__ __forceinline__
            float get_min_time(
                const float3&   volume_max,     // 体素的范围(真实尺度)
                const Vec3fda&  origin,         // 出发点
                const Vec3fda&  direction)      // 方向
            {
                // ? 世界坐标系是如何定义的? 和Volume的坐标系一致? 

                // 只能向一个方向进行 raycasting. 
                //       +y^
                //         | volume 范围
                //         |------- 
                //        /|      |
                //       / |      |         
                //0     /  |--------------> +x
                //     |  /      /
                //     | /      /
                //     |/-------
                //     /
                //    /+z
                // 这里的方向是前面相机图像中的每个像素向空间反投影, 得到的射线的方向
                // 这里就是根据
                //  1. 这个方向在 x 轴上的分量
                //  2. 在 x 轴上当前相机光心到最近的Volume的距离
                // 来估算在 x 轴上迭代到Volume所耗费的"时间"的(和实际需要耗费的时间差一个因子)
                // ? 但是这里对情况的分类我觉得有问题
                float txmin = ((direction.x() > 0 ? 0.f : volume_max.x) - origin.x()) / direction.x();
                // y 轴 z 轴同理
                float tymin = ((direction.y() > 0 ? 0.f : volume_max.y) - origin.y()) / direction.y();
                float tzmin = ((direction.z() > 0 ? 0.f : volume_max.z) - origin.z()) / direction.z();

                // 返回三个数中最大的
                // ? 怎么就成为了最小需要耗费的时间了呢?
                return fmax(fmax(txmin, tymin), tzmin);
            }

            __device__ __forceinline__
            float get_max_time(const float3& volume_max, const Vec3fda& origin, const Vec3fda& direction)
            {
                // ?????? 卧槽, 条件的判断方式完全反了
                float txmax = ((direction.x() > 0 ? volume_max.x : 0.f) - origin.x()) / direction.x();
                float tymax = ((direction.y() > 0 ? volume_max.y : 0.f) - origin.y()) / direction.y();
                float tzmax = ((direction.z() > 0 ? volume_max.z : 0.f) - origin.z()) / direction.z();

                return fmin(fmin(txmax, tymax), tzmax);
            }

            __global__
            void raycast_tsdf_kernel(
                const PtrStepSz<short2>     tsdf_volume,                        // Global TSDF Volume
                const PtrStepSz<uchar3>     color_volume,                       // Global Color Volume
                PtrStepSz<float3>           model_vertex,                       // 推理出来的顶点图
                PtrStepSz<float3>           model_normal,                       // 推理出来的法向图
                PtrStepSz<uchar3>           model_color,                        // 推理出来的颜色图
                const int3                  volume_size,                        // Volume 尺寸
                const float                 voxel_scale,                        // Volume 缩放洗漱
                const CameraParameters      cam_parameters,                     // 当前图层相机内参
                const float                 truncation_distance,                // 截断距离
                const Eigen::Matrix<float, 3, 3, Eigen::DontAlign> rotation,    // 相机位姿的旋转矩阵
                const Vec3fda               translation)                        // 相机位姿的平移向量
            {
                // step 0 获取当前线程要处理的图像像素, 并且进行合法性检查
                const int x = blockIdx.x * blockDim.x + threadIdx.x;
                const int y = blockIdx.y * blockDim.y + threadIdx.y;
                // 合法性检查: 判断是否在当前图层图像范围内
                if (x >= model_vertex.cols || y >= model_vertex.rows)
                    return;

                // step 2 ?
                // 计算 Volume 对应的空间范围
                // ? 但是我觉得, 这个范围其实和当前的线程id没有关系, 我们完全可以离线计算啊, 这里让 512*512*512 的每一个线程都计算一次是不是太浪费计算资源了
                const float3 volume_range = make_float3(volume_size.x * voxel_scale,
                                                        volume_size.y * voxel_scale,
                                                        volume_size.z * voxel_scale);
                // 计算当前的点和相机光心的连线, 使用的是在当前相机坐标系下的坐标; 由于后面只是为了得到方向所以这里没有乘Z
                const Vec3fda pixel_position(
                        (x - cam_parameters.principal_x) / cam_parameters.focal_x,      // X/Z
                        (y - cam_parameters.principal_y) / cam_parameters.focal_y,      // Y/Z
                        1.f);                                                           // Z/Z
                // 得到这个连线的方向(从相机指向空间中的反投影射线)在世界坐标系下的表示, 联想: P_w = R_{wc} * P_c
                Vec3fda ray_direction = (rotation * pixel_position);
                ray_direction.normalize();

                // ? 获取raycast的时候的射线的长度 ?
                // fmax: CUDA 中 float 版的 max() 函数
                // 参数 translation 应该理解为相机光心在世界坐标系下的坐标
                // ? 取最长的作为 Raycast 射线的长度? 
                float ray_length = fmax(get_min_time(volume_range, translation, ray_direction), 0.f);
                // ? 
                if (ray_length >= get_max_time(volume_range, translation, ray_direction))
                    return;

                // ?
                ray_length += voxel_scale;
                Vec3fda grid = (translation + (ray_direction * ray_length)) / voxel_scale;

                // ? 拿到对应体素处的 TSDF 值?
                float tsdf = static_cast<float>(tsdf_volume.ptr(
                        __float2int_rd(grid(2)) * volume_size.y + __float2int_rd(grid(1)))[__float2int_rd(grid(0))].x) *
                             DIVSHORTMAX;

                // ? 计算最大搜索长度
                const float max_search_length = ray_length + volume_range.x * sqrt(2.f);
                // ? 开始迭代搜索了, raycasting 开始
                for (; ray_length < max_search_length; ray_length += truncation_distance * 0.5f) {

                    // ? 计算当前迭代的时候的网格?
                    grid = ((translation + (ray_direction * (ray_length + truncation_distance * 0.5f))) / voxel_scale);

                    // ? 合法性检查
                    if (grid.x() < 1 || grid.x() >= volume_size.x - 1 || grid.y() < 1 ||
                        grid.y() >= volume_size.y - 1 ||
                        grid.z() < 1 || grid.z() >= volume_size.z - 1)
                        continue;

                    // 保存上一次的 TSDF 值, 用于进行下面的判断
                    const float previous_tsdf = tsdf;
                    // ? 计算当前 Grid 处的 TSDF 值
                    tsdf = static_cast<float>(tsdf_volume.ptr(
                            __float2int_rd(grid(2)) * volume_size.y + __float2int_rd(grid(1)))[__float2int_rd(
                            grid(0))].x) *
                           DIVSHORTMAX;

                    // 判断是否穿过了平面
                    if (previous_tsdf < 0.f && tsdf > 0.f) //Zero crossing from behind
                        // 这种情况是从平面的后方穿出了
                        break;
                    if (previous_tsdf > 0.f && tsdf < 0.f) { //Zero crossing
                        // 确实在当前的位置穿过了平面

                        // ? 好像是一个系数
                        const float t_star =
                                ray_length - truncation_distance * 0.5f * previous_tsdf / (tsdf - previous_tsdf);
                        // ? 根据这个系数计算当前这个 Grid ? 点的位置, 其实就是对于当前线程处理的像素(x,y)所观测到的平面的空间位置
                        // vec3f
                        const auto vertex = translation + ray_direction * t_star;

                        // 计算这个点在 volume 中的位置
                        const Vec3fda location_in_grid = (vertex / voxel_scale);
                        // 然后进行合法性检查, 如果确认这个 vertex 不在我们的 Volume 中那么我们就不管它了
                        if (location_in_grid.x() < 1 | location_in_grid.x() >= volume_size.x - 1 ||
                            location_in_grid.y() < 1 || location_in_grid.y() >= volume_size.y - 1 ||
                            location_in_grid.z() < 1 || location_in_grid.z() >= volume_size.z - 1)
                            break;

                        // 计算这个 Grid 点所在处的平面的法向量

                        // normal  - 法向量
                        // shifted - 中间变量, 见下
                        Vec3fda normal, shifted;

                        // ==== 对 x 轴方向

                        // 获取 Grid 点在 Volume 中的位置
                        shifted = location_in_grid;
                        // 哎我滑~
                        shifted.x() += 1;
                        // 如果滑出体素范围了就不管了
                        if (shifted.x() >= volume_size.x - 1)
                            break;
                        // ? 
                        const float Fx1 = interpolate_trilinearly(
                            shifted,            // vertex 点在Volume的坐标滑动之后的点, Vec3fda
                            tsdf_volume,        // TSDF Volume
                            volume_size,        // Volume 的大小
                            voxel_scale);       // 尺度信息

                        // 类似的操作, 不过滑动的时候换了一个方向
                        shifted = location_in_grid;
                        shifted.x() -= 1;
                        if (shifted.x() < 1)
                            break;
                        const float Fx2 = interpolate_trilinearly(shifted, tsdf_volume, volume_size, voxel_scale);

                        // ?
                        normal.x() = (Fx1 - Fx2);

                        // ======  同上

                        shifted = location_in_grid;
                        shifted.y() += 1;
                        if (shifted.y() >= volume_size.y - 1)
                            break;
                        const float Fy1 = interpolate_trilinearly(shifted, tsdf_volume, volume_size, voxel_scale);

                        shifted = location_in_grid;
                        shifted.y() -= 1;
                        if (shifted.y() < 1)
                            break;
                        const float Fy2 = interpolate_trilinearly(shifted, tsdf_volume, volume_size, voxel_scale);

                        normal.y() = (Fy1 - Fy2);

                        // ======== 同上

                        shifted = location_in_grid;
                        shifted.z() += 1;
                        if (shifted.z() >= volume_size.z - 1)
                            break;
                        const float Fz1 = interpolate_trilinearly(shifted, tsdf_volume, volume_size, voxel_scale);

                        shifted = location_in_grid;
                        shifted.z() -= 1;
                        if (shifted.z() < 1)
                            break;
                        const float Fz2 = interpolate_trilinearly(shifted, tsdf_volume, volume_size, voxel_scale);

                        normal.z() = (Fz1 - Fz2);

                        // === 滑动的地方完事了

                        // 检查法向量是否计算成功
                        if (normal.norm() == 0)
                            break;

                        // 如果法向量计算成功, 那么首先归一化
                        normal.normalize();

                        // 然后将计算结果保存到顶点图和法向图中
                        model_vertex.ptr(y)[x] = make_float3(vertex.x(), vertex.y(), vertex.z());
                        model_normal.ptr(y)[x] = make_float3(normal.x(), normal.y(), normal.z());

                        // 将浮点类型的这个顶点在Volume中的位置转换成为以int类型表示的
                        auto location_in_grid_int = location_in_grid.cast<int>();
                        // 然后就可以使用这个整数下标获取 Color Volume 中存储的彩色数据了, 将它保存到彩色图中
                        model_color.ptr(y)[x] = color_volume.ptr(
                                location_in_grid_int.z() * volume_size.y +
                                location_in_grid_int.y())[location_in_grid_int.x()];

                        break;
                    }
                }
            }

            // 执行当前帧的指定图层上的表面推理
            void surface_prediction(
                const VolumeData& volume,                   // Global Volume
                GpuMat& model_vertex,                       // 推理得到的顶点图
                GpuMat& model_normal,                       // 推理得到的法向图
                GpuMat& model_color,                        // 推理得到的颜色
                const CameraParameters& cam_parameters,     // 当前图层的相机内参
                const float truncation_distance,            // 截断距离
                const Eigen::Matrix4f& pose)                // 当前帧的相机位姿
            {
                // step 0 数据准备: 清空顶点图\法向图\彩色图
                model_vertex.setTo(0);
                model_normal.setTo(0);
                model_color.setTo(0);

                // step 1 计算线程数量, 这和当前图层图像的大小有关
                dim3 threads(32, 32);
                dim3 blocks((model_vertex.cols + threads.x - 1) / threads.x,
                            (model_vertex.rows + threads.y - 1) / threads.y);

                // step 2 调用核函数进行并行计算
                raycast_tsdf_kernel<<<blocks, threads>>>(
                        volume.tsdf_volume,                 // Global TSDF Volume
                        volume.color_volume,                // Global Color Volume
                        model_vertex,                       // 推理出来的顶点图
                        model_normal,                       // 推理出来的法向图
                        model_color,                        // 推理出来的颜色图
                        volume.volume_size,                 // Volume 尺寸
                        volume.voxel_scale,                 // Volume 缩放洗漱
                        cam_parameters,                     // 当前图层相机内参
                        truncation_distance,                // 截断距离
                        pose.block(0, 0, 3, 3),             // 从相机位姿中提取旋转矩阵
                        pose.block(0, 3, 3, 1));            // 从相机位姿中提取平移向量
                                                            // ? 为什么这里看上去也像是提取反了呢? 

                // step 3 等待线程同步, 然后结束
                cudaThreadSynchronize();
            }
        }
    }
}