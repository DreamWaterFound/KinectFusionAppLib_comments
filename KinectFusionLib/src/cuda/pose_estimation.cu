// Estimates the current pose using ICP
// This is CUDA code; compile with nvcc
// Author: Christian Diller, git@christian-diller.de

#include "include/common.h"

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32

using Matf31da = Eigen::Matrix<float, 3, 1, Eigen::DontAlign>;

namespace kinectfusion {
    namespace internal {
        namespace cuda {

            // 卧槽, nvcc 支持模板函数啊, 牛逼
            // 设备端的函数, 用于执行归约累加的操作
            template<int SIZE>
            static __device__ __forceinline__
            // volatile 关键字禁止了 nvcc 编译器优化掉这个变量, 确保每次都要读值, 避免了潜在的使用上次用剩下的指针的可能
            void reduce(volatile double* buffer)
            {
                // step 0 获取当前线程id , 每个线程对应其中的一个
                const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
                double value = buffer[thread_id];

                // step 1 归约过程开始, 之所以这样做是为了充分利用 GPU 的并行特性
                if (SIZE >= 1024) {
                    if (thread_id < 512) buffer[thread_id] = value = value + buffer[thread_id + 512];
                    // 一定要同步! 因为如果block规模很大的话, 其中的线程是分批次执行的, 这里就会得到错误的结果
                    __syncthreads();
                }
                if (SIZE >= 512) {
                    if (thread_id < 256) buffer[thread_id] = value = value + buffer[thread_id + 256];
                    __syncthreads();
                }
                if (SIZE >= 256) {
                    if (thread_id < 128) buffer[thread_id] = value = value + buffer[thread_id + 128];
                    __syncthreads();
                }
                if (SIZE >= 128) {
                    if (thread_id < 64) buffer[thread_id] = value = value + buffer[thread_id + 64];
                    __syncthreads();
                }

                // step 2 随着归约过程的进行, 当最后剩下的几个线程都在一个warp中时, 就不用考虑线程间同步的问题了, 这样操作可以更快
                // 因为在 128 折半之后, 有64个数据等待加和, 此时需要使用的线程数目不会超过32个. 
                // 而一个warp,正好是32个线程, 所以如果我们使用这32个线程(或者更少的话)就不会遇到线程间同步的问题了(单指令多数据模式, 这32个线程会共享一套取指令单元, 一定是同时完成工作的)
                // 只激活低32个线程, CUDA 中底层的这32个线程一定是在一个warp上进行的.
                if (thread_id < 32) {
                    if (SIZE >= 64) buffer[thread_id] = value = value + buffer[thread_id + 32];
                    if (SIZE >= 32) buffer[thread_id] = value = value + buffer[thread_id + 16];
                    if (SIZE >= 16) buffer[thread_id] = value = value + buffer[thread_id + 8];
                    if (SIZE >= 8) buffer[thread_id] = value = value + buffer[thread_id + 4];
                    if (SIZE >= 4) buffer[thread_id] = value = value + buffer[thread_id + 2];
                    if (SIZE >= 2) buffer[thread_id] = value = value + buffer[thread_id + 1];
                } // 判断当前需要激活的线程是否少于32个
            }

            __global__
            void estimate_kernel(
                const Eigen::Matrix<float, 3, 3, Eigen::DontAlign> rotation_current,        // 上次迭代得到的旋转 Rwc
                const Matf31da translation_current,                                         // 上次迭代得到的平移 twc
                const PtrStep<float3> vertex_map_current,                                   // 当前帧对应图层的顶点图
                const PtrStep<float3> normal_map_current,                                   // 当前帧对应图层的法向图
                const Eigen::Matrix<float, 3, 3, Eigen::DontAlign> rotation_previous_inv,   // 上一帧相机的旋转, Rcw
                const Matf31da translation_previous,                                        // 上一帧相机的平移, twc
                const CameraParameters cam_params,                                          // 当前图层的相机内参
                const PtrStep<float3> vertex_map_previous,                                  // 上一帧相机位姿推理得到的表面顶点图
                const PtrStep<float3> normal_map_previous,                                  // 上一帧相机位姿推理得到的表面法向图
                const float distance_threshold,                                             // ICP 中关联匹配的最大距离阈值
                const float angle_threshold,                                                // ICP 中关联匹配的最大角度阈值
                const int cols,                                                             // 当前图层的图像列数
                const int rows,                                                             // 当前图层的图像行数
                PtrStep<double> global_buffer)                                              // ? 数据缓冲区?
            {
                // step 0 数据准备
                // 获取当前线程处理的像素坐标
                const int x = blockIdx.x * blockDim.x + threadIdx.x;
                const int y = blockIdx.y * blockDim.y + threadIdx.y;
            
                Matf31da n,         // 目标点的法向, KinectFusion中为上一帧的点云对应的法向量
                         d,         // 目标点,      KinectFusion中为上一帧的点云
                         s;         // 源点,        KinectFusion中为当前帧的点云
                // 匹配点的状态, 表示是否匹配, 初始值为 false
                bool correspondence_found = false;

                // step 1 当处理的像素位置合法时进行 // ? -- 进行投影数据关联
                if (x < cols && y < rows) {
                    // step 1.1 获取当前帧点云法向量的x坐标, 判断其法向是否存在
                    Matf31da normal_current;
                    normal_current.x() = normal_map_current.ptr(y)[x].x;
                    // 如果是个非数, 就认为这个法向是不存在的
                    if (!isnan(normal_current.x())) {
                        // step 1.2 获取的点云法向量确实存在, 
                        // 获取当前帧的顶点
                        Matf31da vertex_current;
                        vertex_current.x() = vertex_map_current.ptr(y)[x].x;
                        vertex_current.y() = vertex_map_current.ptr(y)[x].y;
                        vertex_current.z() = vertex_map_current.ptr(y)[x].z;

                        // 将当前帧的顶点坐标转换到世界坐标系下 Pw = Rwc * Pc + twc
                        Matf31da vertex_current_global = rotation_current * vertex_current + translation_current;

                        // 这个顶点在上一帧相机坐标系下的坐标 Pc(k-1) = Rcw(k-1) * (Pw - twc(k-1)) 
                        // ! 这里就是为什么要对旋转求逆的原因了
                        Matf31da vertex_current_camera =
                                rotation_previous_inv * (vertex_current_global - translation_previous);

                        // 接着将该空间点投影到上一帧的图像中坐标系中
                        Eigen::Vector2i point;
                        // __float2int_rd 向下舍入, +0.5 是为了实现"四舍五入"的效果
                        point.x() = __float2int_rd(
                                vertex_current_camera.x() * cam_params.focal_x / vertex_current_camera.z() +
                                cam_params.principal_x + 0.5f);
                        point.y() = __float2int_rd(
                                vertex_current_camera.y() * cam_params.focal_y / vertex_current_camera.z() +
                                cam_params.principal_y + 0.5f);

                        // 检查投影点是否在图像中
                        if (point.x() >= 0 && point.y() >= 0 && point.x() < cols && point.y() < rows &&
                            vertex_current_camera.z() >= 0) {

                            // 如果在的话, 说明数据关联有戏. 但是还需要检查两个地方
                            // 我们先获取上一帧的疑似关联点的法向
                            Matf31da normal_previous_global;
                            normal_previous_global.x() = normal_map_previous.ptr(point.y())[point.x()].x;
                            // 如果它确认存在
                            if (!isnan(normal_previous_global.x())) {
                                // 获取对应顶点
                                Matf31da vertex_previous_global;
                                vertex_previous_global.x() = vertex_map_previous.ptr(point.y())[point.x()].x;
                                vertex_previous_global.y() = vertex_map_previous.ptr(point.y())[point.x()].y;
                                vertex_previous_global.z() = vertex_map_previous.ptr(point.y())[point.x()].z;
                                // 距离检查, 如果顶点距离相差太多则认为不是正确的点
                                const float distance = (vertex_previous_global - vertex_current_global).norm();
                                if (distance <= distance_threshold) {
                                    // 获取完整的当前帧该顶点的法向, 获取的过程移动到这里的主要目的也是为了避免不必要的计算
                                    normal_current.y() = normal_map_current.ptr(y)[x].y;
                                    normal_current.z() = normal_map_current.ptr(y)[x].z;
                                    // 上面获取的法向是在当前帧相机坐标系下表示的, 这里需要转换到世界坐标系下的表示
                                    Matf31da normal_current_global = rotation_current * normal_current;

                                    // 同样获取完整的, 在上一帧中对应顶点的法向. 注意在平面推理阶段得到的法向就是在世界坐标系下的表示
                                    // TODO 确认一下
                                    normal_previous_global.y() = normal_map_previous.ptr(point.y())[point.x()].y;
                                    normal_previous_global.z() = normal_map_previous.ptr(point.y())[point.x()].z;

                                    // 通过计算叉乘得到两个向量夹角的正弦值. 由于 |axb|=|a||b|sin \alpha, 所以叉乘计算得到的向量的模就是 sin \alpha
                                    const float sine = normal_current_global.cross(normal_previous_global).norm();
                                    // ? 应该是夹角越大, sine 越大啊, 为什么这里是大于等于??? 
                                    if (sine >= angle_threshold) {
                                        // 认为通过检查, 保存关联结果和产生的数据
                                        n = normal_previous_global;
                                        d = vertex_previous_global;
                                        s = vertex_current_global;

                                        correspondence_found = true;
                                    }// 通过关联的角度检查
                                }// 通过关联的距离检查
                            }// 上一帧中的关联点有法向
                        }// 当前帧的顶点对应的空间点的对上一帧的重投影点在图像中
                    }// 当前帧的顶点的法向量存在
                }// 当前线程处理的像素位置在图像范围中

                // 保存计算结果. 根据推导, 对于每个点, 对矩阵A贡献有6个元素, 对向量b贡献有一个元素
                float row[7];

                // 只有对成功匹配的点才会进行的操作. 这个判断也会滤除那些线程坐标不在图像中的线程, 这样做可以减少程序中的分支数目
                if (correspondence_found) {
                    // 前面的强制类型转换符号, 目测是为了转换成为 Eigen 中表示矩阵中浮点数元素的类型, 可以将计算结果直接一次写入到 row[0] row[1] row[2]
                    // 矩阵A中的两个主要元素
                    *(Matf31da*) &row[0] = s.cross(n);
                    *(Matf31da*) &row[3] = n;
                    // 矩阵b中当前点贡献的部分
                    row[6] = n.dot(d - s);
                } else
                    // 如果没有找到匹配的点, 或者说是当前线程的id不在图像区域中, 就全都给0
                    // 这样反映在最后的结果中, 就是图像中的这个区域对最后的误差项没有任何贡献, 相当于不存在一样
                    // 貌似这样计算量是多了,但是相比之下GPU更不适合在计算总矩阵A的时候进行多种分支的处理
                    row[0] = row[1] = row[2] = row[3] = row[4] = row[5] = row[6] = 0.f;

                // 存放在 shared_memory. 每一个 block 中的线程共享一个区域的shared_memory
                // smem = Shared MEMory
                __shared__ double smem[BLOCK_SIZE_X * BLOCK_SIZE_Y];
                // 计算当前线程的一维索引
                const int tid = threadIdx.y * blockDim.x + threadIdx.x;

                int shift = 0;
                for (int i = 0; i < 6; ++i) { // Rows
                    for (int j = i; j < 7; ++j) { // Columns and B
                        // 同步当前线程块中的所有线程执行到这里, 避免出现竞争的情况
                        __syncthreads();
                        // 如果把向量中的每个元素都拆分出来的话, 可以发现本质上是对这27个元素累加, 如果我们拿到了最后这27项的累加和, 我们就可以构造矩阵A和向量b了
                        // 这里就是在计算其中的一项, 当前线程, 或者说当前的这个像素给的贡献
                        smem[tid] = row[i] * row[j];
                        // 再同步一次, 确保所有的线程都完成了写入操作, 避免出现"某个线程还在写数据,但是出现了另外的线程还在读数据"的情况
                        __syncthreads();

                        // Block 内对该元素归约 
                        // 调用这个函数的时候使用当前线程自己的线程id
                        // 因为我们最终的目的是对于这一项, 要将所有线程的贡献累加; 累加的过程分为两个阶段, 一个是每个block 内相加,而是对于所有的Block的和,再进行相加.
                        // 这里进行的是每个 block 中相加的一步
                        reduce<BLOCK_SIZE_X * BLOCK_SIZE_Y>(smem);

                        // 当前 block 中的线程#0 负责将归约之后的结果保存到 global_buffer 中. 
                        // shift 其实就是对应着"当前累加的和是哪一项"这一点; 当前block的结果先放在指定位置, 等全部完事之后再在每个block中的累加和已知的基础上,进行归约求和
                        if (tid == 0)
                            global_buffer.ptr(shift++)[gridDim.x * blockIdx.y + blockIdx.x] = smem[0];
                    }
                }// 归约累加
            }

            // 在每个 Block 已经完成累加的基础上, 进行全局的归约累加
            __global__
            void reduction_kernel(PtrStep<double> global_buffer, const int length, PtrStep<double> output)
            {
                double sum = 0.0;

                // 每个线程对应一个 block 的某项求和的结果, 获取之
                // 但是 blocks 可能很多, 这里是以512为一批进行获取, 加和处理的. 640x480只用到300个blocks.
                for (int t = threadIdx.x; t < length; t += 512)
                    sum += *(global_buffer.ptr(blockIdx.x) + t);

                // ? 检查一下每个block可以使用的shared_memory大小
                __shared__ double smem[512];

                // 注意超过范围的线程也能够执行到这里, 上面的循环不会执行, sum=0, 因此保存到 smem 对后面的归约过程没有影响
                smem[threadIdx.x] = sum;
                // 同时运行512个, 一个warp装不下,保险处理就是进行同步
                __syncthreads();

                // 512个线程都归约计算
                reduce<512>(smem);

                // 第0线程负责将每一项的最终求和结果进行转存
                if (threadIdx.x == 0)
                    output.ptr(blockIdx.x)[0] = smem[0];
            };

            // 使用GPU并行计算矩阵A和向量b
            void estimate_step(
                const Eigen::Matrix3f& rotation_current,            // 上次迭代得到的旋转 Rwc
                const Matf31da& translation_current,                // 上次迭代得到的平移 twc
                const cv::cuda::GpuMat& vertex_map_current,         // 当前帧对应图层的的顶点图
                const cv::cuda::GpuMat& normal_map_current,         // 当前帧对应图层的的法向图
                const Eigen::Matrix3f& rotation_previous_inv,       // 上一帧相机外参中的旋转的逆, Rcw
                const Matf31da& translation_previous,               // 上一帧相机的平移 twc
                const CameraParameters& cam_params,                 // 当前图层的相机内参
                const cv::cuda::GpuMat& vertex_map_previous,        // 对应图层的推理得到的平面顶点图
                const cv::cuda::GpuMat& normal_map_previous,        // 对应图层的推理得到的平面法向图
                float distance_threshold,                           // ICP迭代过程中视为外点的距离阈值
                float angle_threshold,                              // ICP迭代过程中视为外点的角度阈值(角度变正弦值)
                Eigen::Matrix<double, 6, 6, Eigen::RowMajor>& A,    // 计算得到的矩阵 A, 行优先
                Eigen::Matrix<double, 6, 1>& b)                     // 计算得到的向量 b
            {
                // step 0 计算需要的线程规模, 每个线程处理当前图像中的一个像素
                const int cols = vertex_map_current.cols;
                const int rows = vertex_map_current.rows;

                // 32 x 32, 但是这里相当于设置的 threads
                dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
                // 这里还开了多个 Grid -- 但是这里相当于设置的 blocks
                dim3 grid(1, 1);
                grid.x = static_cast<unsigned int>(std::ceil(cols / block.x));
                grid.y = static_cast<unsigned int>(std::ceil(rows / block.y));

                // step 1 创建缓冲区                
                // 首先需要解释一下为什么是27项. 这个部分需要根据 estimate_kernel 函数中的 row[7] 得出.
                // 每一对匹配点对矩阵A的贡献是:
                // Ai = | pi x ni | | (pi x ni)^T,  ni^T | 
                //      |    ni   | 
                // row[0]~row[2] 存放 pi x ni
                // row[3]~row[5] 存放 ni
                // 每一对匹配点对向量b的贡献是:
                // bi = | (pi x ni)*((di-si)*ni)|
                //      |       ni *((di-si)*ni)|
                // 所以 row[6] 存放 ((di-si)*ni
                // 如果我们只看下标, 那么展开的:
                //                                  | 0x0 0x1 0x2 |
                // Ci(0,0) = (pi x ni)(pi x ni)^T = | 0x1 1x1 1x2 |, 其中有 0x0 0x1 0x2 1x1 1x2 2x2 共6项
                //                                  | 0x2 1x2 2x2 |  下标:   0   1   2   7   8  13  (对应在buffer中的页id)
                //
                // Ci(0,1) = Ci(1,0)^T              | 0x3 0x4 0x5 |
                //         = (pi x ni)ni^T        = | 1x3 1x4 1x5 |, 其中有 0x3 0x4 0x5 1x3 1x4 1x5 2x3 2x4 2x5 共9项
                //                                  | 2x3 2x4 2x5 |  下标:   3   4   5   9   10  11 14  15  16
                //
                //                                  | 3x3 3x4 3x5 |
                // Ci(1,1) = ni ni^T              = | 3x4 4x4 4x5 |, 其中有 3x3 3x4 3x5 4x4 4x5 5x5 共6项
                //                                  | 3x5 4x5 5x5 |  下标:   18 19  20  22  23  25
                //
                //                                  | 0x6 |
                // bi(0)   = (pi x ni)*rows[7]    = | 1x6 |,         其中有 0x6 1x6 2x6 共3项
                //                                  | 2x6 |          下标:  6   12  17
                //
                //                                  | 3x6 |
                // bi(1)   = ni*rows[7]           = | 4x6 |,         其中有 3x6 4x6 5x6 共3项
                //                                  | 5x6 |          下标:  21  24  26
                // 因此, 对于每一对点的对于最终的矩阵A和向量b的贡献可以拆分成上述27项. 所以如果我们能够分别对每一项, 求出来所有的匹配点对每一项的贡献,
                // 那么我们就可以组合得到最后的矩阵A和向量b.
                // 求和过程分为两个阶段, 第一阶段每个 blocks 得到的先累加在一起, 保存在 global_buffer 中, 其中的每一项是一页, 每一页的尺寸和 Blocks 的尺寸相同
                cv::cuda::GpuMat global_buffer { cv::cuda::createContinuous(27, grid.x * grid.y, CV_64FC1) };
                // 第二阶段再将所有blocks的和再加在一起, 得到最终的27项的和, 存储在 sum_buffer 中
                // 存储最终的每项加和, 一共27项
                cv::cuda::GpuMat sum_buffer { cv::cuda::createContinuous(27, 1, CV_64FC1) };

                // step 2.1 启动核函数, 对于图像上的每个像素执行: 数据关联, 计算误差贡献, 并且每个Block中累加本block的误差总贡献
                // 其实这里可以看到前面的 block 和 grid 都是相当于之前的 threads 和 blocks
                // 
                estimate_kernel<<<grid, block>>>(
                    rotation_current,                               // 上次迭代得到的旋转 Rwc
                    translation_current,                            // 上次迭代得到的平移 twc
                    vertex_map_current,                             // 当前帧对应图层的顶点图
                    normal_map_current,                             // 当前帧对应图层的法向图
                    rotation_previous_inv,                          // 上一帧相机外参的旋转, Rcw
                    translation_previous,                           // 上一帧相机的平移, twc
                    cam_params,                                     // 对应图层的相机内参
                    vertex_map_previous,                            // 上一帧位姿处推理得到的表面顶点图
                    normal_map_previous,                            // 上一帧位姿处推理得到的表面法向图
                    distance_threshold,                             // ICP 中关联匹配的最大距离阈值
                    angle_threshold,                                // ICP 中关联匹配的最大角度阈值
                    cols,                                           // 当前图层的图像列数
                    rows,                                           // 当前图层的图像行数
                    global_buffer);                                 // 暂存每个Block贡献和的缓冲区

                // step 2.2 在得到了每一个block累加和结果的基础上, 进行全局的归约累加
                reduction_kernel<<<27, 512>>>(                      // 27 = 项数, 512 对应blocks数目, 这里是一次批获取多少个 blocks 先前的和. 
                                                                    //如果实际blocks数目超过这些, 超出的部分就类似归约形式累加, 直到累加后的blocks的数目小于512
                    global_buffer,                                  // 每个Block累加的结果
                    grid.x * grid.y,                                // 27项对应global_buffer中的27页,每一页中的每一个元素记录了一个block的累加结果,这里是每一页的尺寸
                    sum_buffer);                                    // 输出, 结果是所有的匹配点对这27项的贡献

                // step 3 将 GPU 中计算好的矩阵A和向量b下载到CPU中，并且组装数据
                // 下载
                cv::Mat host_data { 27, 1, CV_64FC1 };
                sum_buffer.download(host_data);
                // 组装
                // 按照前面的推导, 矩阵A和向量b的最终形式使用rows[*]的下标表示分别为:
                //      | 0x0 0x1 0x2 0x3 0x4 0x5 |                 | 00 01 02 03 04 05 |
                //      | 0x1 1x1 1x2 1x3 1x4 1x5 |                 | 01 07 08 09 10 11 |
                //      | 0x2 1x2 2x2 2x3 2x4 2x5 |   按buffer下标   | 02 08 13 14 15 16 |  
                // A =  | 0x3 1x3 2x3 3x3 3x4 3x5 | =============== | 03 19 14 18 19 20 | => 斜三角对称矩阵, 只要构造上三角, 下三角对称复制就可以了
                //      | 0x4 1x4 2x4 3x4 4x4 4x5 |                 | 04 10 15 19 22 23 |
                //      | 0x5 1x5 2x5 3x5 4x5 5x5 |                 | 05 11 16 20 23 25 |
                //
                //      | 0x6 |                     | 06 |
                //      | 1x6 |                     | 12 |
                //      | 2x6 |   按buffer下标       | 17 |
                // b =  | 3x6 | ================    | 21 |  => j=6 都是 b
                //      | 4x6 |                     | 24 |
                //      | 5x6 |                     | 26 |
                //

                int shift = 0;
                for (int i = 0; i < 6; ++i) { // Rows
                    for (int j = i; j < 7; ++j) { // Columns and B
                        // 获取值.[0]因为这个 host_data 就一列, []中只能填0
                        double value = host_data.ptr<double>(shift++)[0];
                        // j=6 都是 b
                        if (j == 6)
                            b.data()[i] = value;
                        else
                            A.data()[j * 6 + i] = A.data()[i * 6 + j]   // 对称赋值
                                                = value;
                    }
                }
            }
        }
    }
}
