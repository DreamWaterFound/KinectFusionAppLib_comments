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
                const Vec3fda& point,               // 想要得到 TSDF 数值的点的坐标(非整数)
                const PtrStepSz<short2>& volume,    // TSDF Volume 对象
                const int3& volume_size,            // TSDF Volume 对象 大小
                const float voxel_scale)            // TSDF Volume 中的体素坐标和现实世界中长度的度量关系
            {
                // 本函数中考虑的, 都是

                // 这个点在 Volume 下的坐标, 转换成为整数下标标的表示
                Vec3ida point_in_grid = point.cast<int>();

                // 恢复成体素中心点的坐标
                const float vx = (static_cast<float>(point_in_grid.x()) + 0.5f);
                const float vy = (static_cast<float>(point_in_grid.y()) + 0.5f);
                const float vz = (static_cast<float>(point_in_grid.z()) + 0.5f);

                // 查看原始的点的坐标是否更偏向于某个坐标轴上数值更小的一侧, 如果是的话就坐标-1; 否则不变
                // 为什么要减1? 为什么在数值更大的一侧就不用减? 画图:(只看x轴)
                // ^x
                // |
                // |
                // |                                    * (TSDF) ++++++++++++
                // |                                                        +
                // |                                                        +(a)
                // |--------------------------------------------------------+----------------
                // |                                                        +
                // |                                    * point++++++++++++++
                // |    * vx  TSDF++++++++++++(1-a)     * vx ++++++++++++++++
                // |    * point ++++++++++++++          
                // |                         +
                // |----* point_in_grid------+----------* point_in_grid----------------------------------
                // |                         +
                // |                         +
                // |   (*) TSDF              +(a)
                // |                         +
                // |                         +
                // |----* point_in_grid-1 ++++-------------------------------------------------------------
                // |
                // 分成这两种情况是为了方便计算不同组织形式下的插值        
                point_in_grid.x() = (point.x() < vx) ? (point_in_grid.x() - 1) : point_in_grid.x();
                point_in_grid.y() = (point.y() < vy) ? (point_in_grid.y() - 1) : point_in_grid.y();
                point_in_grid.z() = (point.z() < vz) ? (point_in_grid.z() - 1) : point_in_grid.z();

                // +0.5f 的原因是, point_in_grid 处体素存储的TSDF值是体素的中心点的TSDF值
                // 三线型插值, ref: https://en.wikipedia.org/wiki/Trilinear_interpolation
                // 计算精确的(浮点型)的点坐标和整型化之后的点坐标的差
                const float a = (point.x() - (static_cast<float>(point_in_grid.x()) + 0.5f));
                const float b = (point.y() - (static_cast<float>(point_in_grid.y()) + 0.5f));
                const float c = (point.z() - (static_cast<float>(point_in_grid.z()) + 0.5f));

                return 
                    static_cast<float>(volume.ptr((point_in_grid.z()) * volume_size.y + point_in_grid.y())[point_in_grid.x()].x) * DIVSHORTMAX 
                        // volume[ x ][ y ][ z ], C000
                        * (1 - a) * (1 - b) * (1 - c) +
                    static_cast<float>(volume.ptr((point_in_grid.z() + 1) * volume_size.y + point_in_grid.y())[point_in_grid.x()].x) * DIVSHORTMAX 
                        // volume[ x ][ y ][z+1], C001
                        * (1 - a) * (1 - b) * c +
                    static_cast<float>(volume.ptr((point_in_grid.z()) * volume_size.y + point_in_grid.y() + 1)[point_in_grid.x()].x) * DIVSHORTMAX 
                        // volume[ x ][y+1][ z ], C010
                        * (1 - a) * b * (1 - c) +
                    static_cast<float>(volume.ptr((point_in_grid.z() + 1) * volume_size.y + point_in_grid.y() + 1)[point_in_grid.x()].x) * DIVSHORTMAX 
                        // volume[ x ][y+1][z+1], C011
                        * (1 - a) * b * c +
                    static_cast<float>(volume.ptr((point_in_grid.z()) * volume_size.y + point_in_grid.y())[point_in_grid.x() + 1].x) * DIVSHORTMAX 
                        // volume[x+1][ y ][ z ], C100
                        * a * (1 - b) * (1 - c) +
                    static_cast<float>(volume.ptr((point_in_grid.z() + 1) * volume_size.y + point_in_grid.y())[point_in_grid.x() + 1].x) * DIVSHORTMAX 
                        // volume[x+1][ y ][z+1], C101
                        * a * (1 - b) * c +
                    static_cast<float>(volume.ptr((point_in_grid.z()) * volume_size.y + point_in_grid.y() + 1)[point_in_grid.x() + 1].x) * DIVSHORTMAX 
                        // volume[x+1][y+1][ z ], C110
                        * a * b * (1 - c) +
                    static_cast<float>(volume.ptr((point_in_grid.z() + 1) * volume_size.y + point_in_grid.y() + 1)[point_in_grid.x() + 1].x) * DIVSHORTMAX 
                        // volume[x+1][y+1][z+1], C111
                        * a * b * c;
            }

            /*************************************************************
            NOTE 下面的两个函数是为了得到在raycast过程中, 什么时候射线开始射入 volume, 什么时候射线射出 volume

            这里有一个假设: 相机是一直都向Volume的方向观测的
            如果只看x和y轴, 那么坐标系是这样定义的:
                        y^
                         |
                         |
                         |---------
                         |        |
                         | volume |
                         |        |
            -------------|------------->x
                         |
                         |
            首先想求的是在每个轴上进行 raycast 需要最短的时间. 在步长一致的情况下:
            在每个轴上耗费的单位时间数tmin = (当前相机位置在这个轴上到 volume 的距离)/(raycast方向在这个轴上的分量)
            类似地, 若要射线完整地穿过 volume 所需要的最长的时间, 对于每个轴:
            在每个轴上耗费的时间单位数tmax = (当前相机位置在这个轴上到 volume 另一端的距离)/(raycast方向在这个轴上的分量)
            而为了近似得到当前射线方向按照给定步长前进所需的最少时间, 程序这样计算
            final_min_time = max(txmin, tymin, tzmin)
            目的是保证当射线前进了 final_min_time 后, 所有的轴上(几乎)一定接触到了 Volume , 可以进行 raycast 过程了
            类似地为了近似地得到当前射线方向按照给定步长前进,走出Volume所耗费的最少时间, 程序也使用了比较保守的策略:
            final_max_time = min(txmax, tymax, tzmax)
            这样能够确定经过了 final_max_time 之后, 射线在其中一个轴上就脱离 volmue 了, 相当于射线已经出了 volume, raycast就可以停止了
            
            // ! 但是上述的分析在相机处于某些区域的时候可能站不住脚, 比如相机的位姿中 0<ty<volume.size.y的时候并且direct.y > 0. 得到的tmin是个负值
            这个时候就会出现计算错误的情况
            **************************************************************/

            // __forceinline__: 强制为内联函数
            __device__ __forceinline__
            // 求射线为了射入Volume, 在给定步长下所需要的最少的前进次数(也可以理解为前进所需要的时间)
            float get_min_time(
                const float3&   volume_max,     // 体素的范围(真实尺度)
                const Vec3fda&  origin,         // 出发点, 也就是相机当前的位置
                const Vec3fda&  direction)      // 射线方向
            {
                // 分别计算三个轴上的次数, 并且返回其中最大; 当前进了这个最大的次数之后, 三个轴上射线的分量就都已经射入volume了
                float txmin = ((direction.x() > 0 ? 0.f : volume_max.x) - origin.x()) / direction.x();
                float tymin = ((direction.y() > 0 ? 0.f : volume_max.y) - origin.y()) / direction.y();
                float tzmin = ((direction.z() > 0 ? 0.f : volume_max.z) - origin.z()) / direction.z();
                
                return fmax(fmax(txmin, tymin), tzmin);
            }

            __device__ __forceinline__
            // 求射线为了射出Volume, 在给定步长下所需要的最少的前进次数(也可以理解为前进所需要的时间)
            float get_max_time(const float3& volume_max, const Vec3fda& origin, const Vec3fda& direction)
            {
                // 分别计算三个轴上的次数, 并且返回其中最小. 当前进了这个最小的次数后, 三个轴上的射线的分量中就有一个已经射出了volume了
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

                // step 2 计算 raycast 射线, 以及应该在何处开始, 在何处结束
                // 计算 Volume 对应的空间范围
                // ! 但是我觉得, 这个范围其实和当前的线程id没有关系, 我们完全可以离线计算啊, 这里让 512*512*512 的每一个线程都计算一次是不是太浪费计算资源了
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

                // fmax: CUDA 中 float 版的 max() 函数
                // 参数 translation 应该理解为相机光心在世界坐标系下的坐标
                // 获得 raycast 的起始位置
                float ray_length = fmax(get_min_time(volume_range, translation, ray_direction), 0.f);
                // 验证是否合法: 起始位置的射线长度应该小于等于结束位置的射线长度
                if (ray_length >= get_max_time(volume_range, translation, ray_direction))
                    return;

                // 在开始位置继续前进一个体素, 确保该位置已经接触到 volume
                ray_length += voxel_scale;
                Vec3fda grid = (translation + (ray_direction * ray_length)) / voxel_scale;

                // 拿到 Grid 对应体素处的 TSDF 值, 这里充当当前射线的上一次的TSDF计算结果
                // 如果拿到的坐标并不在 volume 中, 那么得到的 tsdf 值无法确定, 甚至可能会触发段错误
                // __float2int_rd: 向下取整
                float tsdf = static_cast<float>(tsdf_volume.ptr(
                        __float2int_rd(grid(2)) * volume_size.y + __float2int_rd(grid(1)))[__float2int_rd(grid(0))].x) *
                             DIVSHORTMAX;

                // 计算最大搜索长度(考虑了光线开始“投射”的时候已经走过的路程 ray_length )  
                // ! 不明白这里为什么是根号2 而不是根号3
                // ! 这里没有乘 SCALE 也应该有问题
                const float max_search_length = ray_length + volume_range.x * sqrt(2.f);
                // step 3 开始迭代搜索了, raycasting 开始. 步长为一半截断距离
                for (; ray_length < max_search_length; ray_length += truncation_distance * 0.5f) {

                    // step 3.1 获取当前射线位置的 TSDF
                    // 计算当前次前进后, 射线到达的体素id
                    grid = ((translation + (ray_direction * (ray_length + truncation_distance * 0.5f))) / voxel_scale);

                    // 合法性检查
                    if (grid.x() < 1 || grid.x() >= volume_size.x - 1 || grid.y() < 1 ||
                        grid.y() >= volume_size.y - 1 ||
                        grid.z() < 1 || grid.z() >= volume_size.z - 1)
                        continue;

                    // 保存上一次的 TSDF 值, 用于进行下面的判断
                    const float previous_tsdf = tsdf;
                    // 计算当前 Grid 处的 TSDF 值
                    tsdf = static_cast<float>(tsdf_volume.ptr(
                            __float2int_rd(grid(2)) * volume_size.y + __float2int_rd(grid(1)))[__float2int_rd(
                            grid(0))].x) *
                           DIVSHORTMAX;

                    // step 3.2 判断是否穿过了平面
                    if (previous_tsdf < 0.f && tsdf > 0.f) //Zero crossing from behind
                        // 这种情况是从平面的后方穿出了
                        break;
                    if (previous_tsdf > 0.f && tsdf < 0.f) { //Zero crossing
                        // step 3.3 确实在当前的位置穿过了平面, 计算当前射线与该平面的交点

                        // 精确确定这个平面所在的位置(反映为射线的长度), 计算公式与论文中式(15)保持一致
                        const float t_star =
                                ray_length - truncation_distance * 0.5f * previous_tsdf / (tsdf - previous_tsdf);
                        // 计算射线和这个平面的交点. 下文简称平面顶点. vec3f 类型
                        const auto vertex = translation + ray_direction * t_star;

                        // 计算平面顶点在 volume 中的位置
                        const Vec3fda location_in_grid = (vertex / voxel_scale);
                        // 然后进行合法性检查, 如果确认这个 vertex 不在我们的 Volume 中那么我们就不管它了
                        if (location_in_grid.x() < 1 || location_in_grid.x() >= volume_size.x - 1 ||
                            location_in_grid.y() < 1 || location_in_grid.y() >= volume_size.y - 1 ||
                            location_in_grid.z() < 1 || location_in_grid.z() >= volume_size.z - 1)
                            break;

                        // step 3.4 分x, y, z三个轴, 计算这个 Grid 点所在处的平面的法向量

                        // normal  - 法向量
                        // shifted - 中间变量, 用于滑动
                        Vec3fda normal, shifted;

                        // step 3.4.1 对 x 轴方向
                        shifted = location_in_grid;
                        // 在平面顶点的体素位置的基础上, 哎我滑~ 如果滑出体素范围就不管了
                        shifted.x() += 1;
                        if (shifted.x() >= volume_size.x - 1)
                            break;
                        // 这里得到的是 TSDF 值. 
                        // 为什么不直接使用 shifted 对应体素的 TSDF 值而是进行三线性插值, 是因为 Volume 中只保存了体素中心点到平面的距离, 
                        // 但是这里的 location_in_grid+1 也就是 shifted 是个浮点数, 为了得到相对准确的TSDF值, 需要进行三线性插值
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

                        // 由于 TSDF 值就反映了该体素中心点处到相机反投影平面的距离, 所以这里可以使用这个数据来进行表示
                        // ! 但是这样基于这个点周围体素中的距离都没有被截断才比较准确, 否则可能出现一个轴上的法向量为0的情况
                        normal.x() = (Fx1 - Fx2);

                        // step 3.4.2 对 y 轴方向
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

                        // step 3.4.3 对 z 轴方向
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

                        // step 3.4.4 检查法向量是否计算成功, 如果成功进行归一化
                        if (normal.norm() == 0)
                            break;

                        // 如果法向量计算成功, 那么首先归一化
                        normal.normalize();

                        // step 3.5 保存平面顶点和平面法向数据
                        // 然后将计算结果保存到顶点图和法向图中
                        model_vertex.ptr(y)[x] = make_float3(vertex.x(), vertex.y(), vertex.z());
                        model_normal.ptr(y)[x] = make_float3(normal.x(), normal.y(), normal.z());

                        // step 3.6 获取该点处的彩色数据
                        // 将浮点类型的这个顶点在Volume中的位置转换成为以int类型表示的
                        auto location_in_grid_int = location_in_grid.cast<int>();
                        // 然后就可以使用这个整数下标获取 Color Volume 中存储的彩色数据了, 将它保存到彩色图中
                        model_color.ptr(y)[x] = color_volume.ptr(
                                location_in_grid_int.z() * volume_size.y +
                                location_in_grid_int.y())[location_in_grid_int.x()];

                        break;
                    }
                } // raycasting
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

                // step 3 等待线程同步, 然后结束
                cudaThreadSynchronize();
            }
        }
    }
}