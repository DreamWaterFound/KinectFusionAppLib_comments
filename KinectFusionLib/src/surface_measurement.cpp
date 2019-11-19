// This is the CPU part of the surface measurement
// Author: Christian Diller, git@christian-diller.de

#include <kinectfusion.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Weffc++"
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#pragma GCC diagnostic pop

using cv::cuda::GpuMat;

namespace kinectfusion {
    namespace internal {

        namespace cuda { // Forward declare CUDA functions
            /**
             * @brief 计算某层深度图像的顶点图
             * @param[in]  depth_map        某层滤波后的深度图
             * @param[out] vertex_map       计算得到的顶点图
             * @param[in]  depth_cutoff     不考虑的过远的点的距离
             * @param[in]  cam_params       该层图像下的相机内参
             */
            void compute_vertex_map(const GpuMat& depth_map, GpuMat& vertex_map, const float depth_cutoff,
                                    const CameraParameters cam_params);
            /**
             * @brief 根据某层顶点图计算法向图
             * @param[in]  vertex_map       某层顶点图
             * @param[out] normal_map       计算得到的法向图
             */
            void compute_normal_map(const GpuMat& vertex_map, GpuMat& normal_map);
        }

        // 计算输入深度图的顶点\法向金字塔
        FrameData surface_measurement(
            const cv::Mat_<float>& input_frame,                                         // 输入的深度图
            const CameraParameters& camera_params,                                      // 深度相机内参
            const size_t num_levels, const float depth_cutoff,                          // 图层数和不考虑的过远的点的距离
            const int kernel_size, const float color_sigma, const float spatial_sigma)  // 双边滤波器的参数
        {
            // step 1 Initialize frame data
            FrameData data(num_levels);

            // step 2 Allocate GPU memory
            // 对于金字塔中的每一层
            for (size_t level = 0; level < num_levels; ++level) {
                // step 2.1 获取图像大小
                const int width = camera_params.level(level).image_width;
                const int height = camera_params.level(level).image_height;

                // step 2.2 分配每个金字塔"图像"的存储空间
                data.depth_pyramid[level] = cv::cuda::createContinuous(height, width, CV_32FC1);
                data.smoothed_depth_pyramid[level] = cv::cuda::createContinuous(height, width, CV_32FC1);

                data.color_pyramid[level] = cv::cuda::createContinuous(height, width, CV_8UC3);

                data.vertex_pyramid[level] = cv::cuda::createContinuous(height, width, CV_32FC3);
                data.normal_pyramid[level] = cv::cuda::createContinuous(height, width, CV_32FC3);
            }

            // step 3 Start by uploading original frame to GPU
            // 上传原始的深度图到GPU
            data.depth_pyramid[0].upload(input_frame);

            // step 4 Build pyramids and filter bilaterally on GPU
            // 一个容器,封装了在GPU中进行的异步操作流. 或者可以理解为一个在CPU中控制GPU子程序的运行序列的对象
            // cv::cuda::Stream类并不是线程安全的,所以OpenCV文档建议每个线程在GPU中开的任务最好都使用不同的stream对象
            cv::cuda::Stream stream;
            // step 4.1 生成深度图图像金字塔
            for (size_t level = 1; level < num_levels; ++level)
                cv::cuda::pyrDown(data.depth_pyramid[level - 1], data.depth_pyramid[level], stream);
            // step 4.2 对图像金字塔中的每一张图像都进行双边滤波操作
            for (size_t level = 0; level < num_levels; ++level) {
                cv::cuda::bilateralFilter(data.depth_pyramid[level],            // source
                                          data.smoothed_depth_pyramid[level],   // destination
                                          kernel_size,                          // 这个是双边滤波器滤波的核大小, 不是GPU核函数的那个核
                                          color_sigma,
                                          spatial_sigma,
                                          cv::BORDER_DEFAULT,                   // 默认边缘的补充生成方案 gfedcb|abcdefgh|gfedcba
                                          stream);                              // 加入到指定的流中
            }
            // 等待GPU中的这个计算流完成(就是上面进行的双边滤波的操作完成)
            stream.waitForCompletion();

            // step 5 Compute vertex and normal maps
            // 对于每一层图像, 使用GPU计算顶点图和法向图
            for (size_t level = 0; level < num_levels; ++level) {
                // 顶点图, 说白了就是根据深度图计算3D点
                cuda::compute_vertex_map(data.smoothed_depth_pyramid[level], data.vertex_pyramid[level],
                                         depth_cutoff, camera_params.level(level));
                // 法向图, 需要根据顶点图来计算法向量
                cuda::compute_normal_map(data.vertex_pyramid[level], data.normal_pyramid[level]);
            }

            // step 6 返回这个帧
            return data;
        }
    }
}