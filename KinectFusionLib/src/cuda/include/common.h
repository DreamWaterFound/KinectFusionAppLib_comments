// CUDA kernel header with commonly used definitions, functions and data structures
// Author: Christian Diller, git@christian-diller.de

//If you are working with CUDA code in CLion
#ifdef __JETBRAINS_IDE__
#define __CUDACC__ 1
#define __host__
#define __device__
#define __global__
#define __forceinline__
#define __shared__
inline void __syncthreads() {}
using blockDim = struct { int x; int y; };
using threadIdx = struct { int x; int y; int z; };
using blockIdx = struct { int x; int y; int z; };
#endif

#include <data_types.h>

// ? 下面的这些数据是干嘛的?
#define DIVSHORTMAX 0.0000305185f       // (1.f / SHRT_MAX);
#define SHORTMAX    32767               // SHRT_MAX;
#define MAX_WEIGHT  128                 // Global TSDF Volume 更新过程中, 允许的最大权重值

// 可以理解为在设备端的 GpuMat 的类型, 精简了数据结构，意味着参数传递的过程比较快
// 保留了 GpuMat 的cols rows step data
using cv::cuda::PtrStep;
// 保留了 GpuMat 的 step data
// NOTE 由于字节对齐的原因，图像在显存中可能无法连续存储，每一行的末尾可能要补几个空字节，因此实际上一行占用的字节由 GpuMat::step 指出
using cv::cuda::PtrStepSz;
using cv::cuda::GpuMat;

// ? 后缀 da 是什么意思? device array, 设备端数组?
// 三维点的数据类型
using Vec3fda = Eigen::Matrix<float, 3, 1, Eigen::DontAlign>;
