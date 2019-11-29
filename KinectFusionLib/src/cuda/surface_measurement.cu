// Measures the surface, i.e. computes vertex and normal maps from a depth frame
// This is CUDA code; compile with nvcc
// Author: Christian Diller, git@christian-diller.de

#include "include/common.h"

using cv::cuda::GpuMat;

namespace kinectfusion {
    namespace internal {
        namespace cuda {

            // 核函数，用于计算深度图像中每一个像素的3D点
            __global__
            void kernel_compute_vertex_map(
                const PtrStepSz<float> depth_map,       // 滤波之后的深度图像对象
                PtrStep<float3> vertex_map,             // 保存计算结果的顶点图. 我猜由于上面的参数已经给出了这个图的大小了, 所以这里只是使用了 PtrStep 类型
                const float depth_cutoff,               // 不考虑的过远的点的距离
                const CameraParameters cam_params)      // 相机内参
            {
                // step 1 计算对应的下标, 并且进行区域有效性检查
                const int x = blockIdx.x * blockDim.x + threadIdx.x;
                const int y = blockIdx.y * blockDim.y + threadIdx.y;

                // 如果当前线程对应的图像像素并不真正地在图像中，那么当前就不需要计算了
                if (x >= depth_map.cols || y >= depth_map.rows)
                    return;

                // step 2 获取深度并且进行背景判断
                float depth_value = depth_map.ptr(y)[x];
                if (depth_value > depth_cutoff) depth_value = 0.f; // Depth cutoff

                // step 3 生成三维点, 根据相机内参进行反投影即可, 得到的是三维点在当前世界坐标系下的坐标
                Vec3fda vertex(
                        (x - cam_params.principal_x) * depth_value / cam_params.focal_x,
                        (y - cam_params.principal_y) * depth_value / cam_params.focal_y,
                        depth_value);

                // step 4 保存计算结果
                vertex_map.ptr(y)[x] = make_float3(vertex.x(), vertex.y(), vertex.z());
            }

            // 核函数, 用于根据顶点图计算法向图
            __global__
            void kernel_compute_normal_map(
                const PtrStepSz<float3> vertex_map,                 // 输入, 顶点图
                PtrStep<float3> normal_map)                         // 输出, 法向图
            {
                // step 1 根据当前线程id得到要进行处理的像素, 并且进行区域有效性判断
                const int x = blockIdx.x * blockDim.x + threadIdx.x;
                const int y = blockIdx.y * blockDim.y + threadIdx.y;

                if (x < 1 || x >= vertex_map.cols - 1 || y < 1 || y >= vertex_map.rows - 1)
                    return;

                // step 2 获取以当前顶点为中心, 上下左右四个方向的顶点数据, 都是在当前帧相机坐标系下的坐标表示
                const Vec3fda left(&vertex_map.ptr(y)[x - 1].x);
                const Vec3fda right(&vertex_map.ptr(y)[x + 1].x);
                const Vec3fda upper(&vertex_map.ptr(y - 1)[x].x);
                const Vec3fda lower(&vertex_map.ptr(y + 1)[x].x);

                // step 3 计算当前顶点的法向
                Vec3fda normal;
                // 当前的顶点的上下左右只要有一个顶点无数据, 那么当前的顶点就没有法向数据
                if (left.z() == 0 || right.z() == 0 || upper.z() == 0 || lower.z() == 0)
                    normal = Vec3fda(0.f, 0.f, 0.f);
                else {
                    // 计算一个 right -> left 的向量
                    Vec3fda hor(left.x() - right.x(), left.y() - right.y(), left.z() - right.z());
                    // 计算一个 lower -> upper 的向量
                    Vec3fda ver(upper.x() - lower.x(), upper.y() - lower.y(), upper.z() - lower.z());

                    // 叉乘, 归一化, 确保法向的方向是"朝外的"(相对于相机光心来说)
                    normal = hor.cross(ver);
                    normal.normalize();

                    if (normal.z() > 0)
                        normal *= -1;
                }

                // 保存计算的法向量结果
                normal_map.ptr(y)[x] = make_float3(normal.x(), normal.y(), normal.z());
            }

            // 计算某一层图像的顶点图. 主机端
            void compute_vertex_map(
                const GpuMat& depth_map,                // 输入的一层滤波之后的深度图像
                GpuMat& vertex_map,                     // 计算好的顶点图
                const float depth_cutoff,               // 不考虑的过远的点的距离
                const CameraParameters cam_params)      // 当前图层下的相机内参数
            {
                // step 1 计算需要使用的线程数目和线程块数目
                // 每个线程块中的线程的数目,一般要求是32的倍数来得到最充分的资源利用
                dim3 threads(32, 32);
                // 线程块的数目. 每个线程块的中的线程运行在同一个流处理器上, 可以通过 shaed memory 通信, 但是这点对于咱们当前应用似乎是没有什么帮助
                // 所以要尽量保证每一个线程块内的线程都有完全相同的处理步骤
                // 这里多加了一个threads.x是避免出现计算结果默认向下取整后出现线程数不够用的情况
                // +1则是考虑到了线程的下标是从0开始的
                dim3 blocks((depth_map.cols + threads.x - 1) / threads.x, (depth_map.rows + threads.y - 1) / threads.y);

                // step 2 启动 GPU 核函数
                kernel_compute_vertex_map << < blocks, threads >> > (depth_map, vertex_map, depth_cutoff, cam_params);

                // step 3 等待所有开启的并行线程结束
                cudaThreadSynchronize();
            }

            // 根据某一层图像的顶点图计算其法向图
            void compute_normal_map(
                const GpuMat& vertex_map,               // 输入, 某一图层的顶点图
                GpuMat& normal_map)                     // 输出, 某一图层的法向图
            {
                // step 1 计算核函数尺寸
                dim3 threads(32, 32);
                dim3 blocks((vertex_map.cols + threads.x - 1) / threads.x,
                            (vertex_map.rows + threads.y - 1) / threads.y);

                // step 2 启动核函数
                kernel_compute_normal_map<<<blocks, threads>>>(vertex_map, normal_map);

                // step 3 等待核函数计算完成
                cudaThreadSynchronize();
            }
        }
    }
}