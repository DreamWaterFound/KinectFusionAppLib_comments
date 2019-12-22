# KinectFusionLib and KinectFusionApp

这个是自己在源仓库的基础上，加上了自己注释的版本。

源仓库：

- KinectFusionLib [https://github.com/chrdiller/KinectFusionLib]
- KinectFusionApp [https://github.com/chrdiller/KinectFusionApp]

程序由两个部分组成，lib部分和app部分。

注意编译通过的时候需要使用在cuda支持下的Eigen和OpenCV库，CUDA编译的时候使用的是8.0的版本。


其实里面还有一些小问题我也没有搞明白，欢迎有想法的大佬们疯狂提issue讨论～～

另 KinectFusionLib 中将最后的模型输出成为点云文件的功能部分我现在也没有进行注释（和 KinectFusion 本身的功能关系不是很大）（好吧，我懒 = =）

KinectFusion 中的 Point-to-plane ICP 的并行化有一些数学推导，导致KinectFusionLib中代码实现看起来比较“骚”，希望 `/doc/【附件】刘国庆-KinectFusion中Plane-to-point_ICP的计算推导.pdf` 这篇文档能够有所帮助。

另外关于此 KinectFusionLib 的实现我在泡泡机器人的沈阳线下会议做了一个PPT：`/doc/【PPT】刘国庆-KinectFusionLib代码精析-Preview-V5.pdf`， 希望对感兴趣的同学有所帮助。

下面是原版的 ReadMe.md.

----


KinectFusionApp
===============

This is a sample application using the [KinectFusionLib](https://github.com/chrdiller/KinectFusionLib). It implements 
cameras (for data acquisition from recordings as well as from a live depth sensor) as data sources. The resulting fused volume 
can then be exported into a pointcloud or a dense surface mesh.

Dependencies
------------
* **GCC 5** as higher versions don't work with current nvcc (as of 2017).
* **CUDA 8.0** or higher. In order to provide real-time reconstruction, this library relies on graphics hardware.
Running it exclusively on CPU is not possible.
* **OpenCV 3.0** or higher. This library heavily depends on the GPU features of OpenCV that have been refactored in the 3.0 release.
Therefore, OpenCV 2 is not supported.
* **Eigen3** for efficient matrix and vector operations.
* **OpenNI2** for data acquisition with a live depth sensor.

Prerequisites
-------------
* Adjust CUDA architecture: Set the CUDA architecture version to that of your graphics hardware
```cmake
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -gencode arch=compute_52,code=sm_52)
```
Tested with a nVidia GeForce 970, compute capability 5.2, maxwell architecture
* Set custom opencv path (if built from source):
```cmake
SET("OpenCV_DIR" "/opt/opencv/usr/local/share/OpenCV")
```

Usage
-----
Setup the data sources in main.cpp. Then, start the application.

Use the following keys to perform actions:
* 'p': Export all camera poses known so far
* 'm': Export a dense surface mesh
* ' ': Export nothing, just end the application
* 'a': Save all available data

License
-------
This library is licensed under MIT.
