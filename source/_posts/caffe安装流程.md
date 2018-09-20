---
title: caffe安装流程
date: 2016-12-04 21:39:25
tags: caffe
categories: 机器学习
---

<blockquote class="blockquote-center">caffe.proto</blockquote>
<!-- more -->

参考[caffe install](http://caffe.berkeleyvision.org/installation.html)
# 前期准备
需要cuda,BLAS(通过ATLAS,MKL,或者OpenBLAS),Boost,各种依赖(protobuf,glog,gflags,hdf5),OpenCV,lmdb,leveldb(这两个是google搞得数据库，数据引擎，caffe里面的数据一般是这两种格式),cuDNN(用于GPU加速)
## 依赖安装
    
    sudo apt-get install build-essential  # basic requirement  
    sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libboost-all-dev libhdf5-serial-dev libgflags-dev libgoogle-glog-dev liblmdb-dev protobuf-compiler #required by caffe

-dev是开发包，lib是库函数包，运行需要lib,编译需要dev,dev是给开发使用的，包含了这个库的头文件，有时里面还有静态库和其他开发时可能需要的文件，上面把protobuf,leveldb,snappy(leveldb需要这个依赖),opencv,boost,hdf5,gflags,glog,lmdb的编译所需要头文件都包含了。
**注意：protobuf需要安装protobuf-compiler，这是protobuf的编译器**
简单讲讲protobuf,它是google搞出的一种类似于xml的数据结构，用这个protobuf，就可以省很多事，我们只需要写消息包，然后编译，protobuf就会根据这个消息包自动生成两个文件.cc和.h，.h这两个文件中就有很多类，供我们调用。
要擅于使用`apt-cache search XXX`,来查找需要的包。
## 安装cuda
首先确保你有nvdia的显卡.
`lspci | grep -i nvidia`
知道你的linux版本，64还是32
`uname -m && cat /etc/*release`
确保gcc安装了，看下版本
`gcc --version`
OK,现在去官网下载cuda
`https://developer.nvidia.com/cuda-downloads`
[官方文档](http://developer.download.nvidia.com/compute/cuda/7_0/Prod/doc/CUDA_Getting_Started_Linux.pdf)是安装步骤最全面的，网上的好多帖子有问题。
deb包安装简单，不过run包稳定，无所谓。
### 安装依赖：
`sudo apt-get install freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libgl1-mesa-glx libglu1-mesa libglu1-mesa-dev`
注意看错误提示，缺啥补啥，比如编译时遇到错误`cannot find -lXi -lXmu -lglut`，那就是没有安装freeglut、mesa和opengl，找对应的dev去安装。
### 还是按照官网上的流程来。
`sudo dpkg -i cuda-repo-<distro>_<version>_<architecture>.deb
sudo apt-get update
sudo apt-get install cuda`
安装cuda的同时就会把显卡驱动也全部安装好。
**注意，不要更新（不是不要update，是不要upgrade），否则会丢失图形界面**
### 设置环境变量

    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH 
/usr/local/cuda是你的cuda安装的地方，记得命名一定要改成cuda，caffe编译的时候找cuda就只找这个文件夹，（机子32位lib64改成lib)

### 安装示例
`cuda-install-samples-6.5.sh <dir> `dir是你想安装例子的位置
然后进入该文件夹make一下，这时候就有可能出现刚刚说的依赖问题，缺哪补哪。

### 测试一下
`cat /proc/driver/nvidia/version `显卡驱动版本
`nvcc -V i `nvcc编译器版本
cd到例子所在文件夹，进入bin路径，`./ deviceQuery`跑下例子。

## 安装BLAS
利用ATLAS或者MKL都可以，OpenBLAS没用过，应该也可以。
`sudo apt-get install libatlas-base-dev`atlas安装方法。
MKL收费，可以申请非商业版，或者网上找找。安装步骤抄了一个
下面有一个install_GUI.sh文件，执行该文件，会出现图形安装界面，根据说明一步一步执行即可。
注意： 安装完成后需要添加library路径
`sudo gedit /etc/ld.so.conf.d/intel_mkl.conf`
在文件中添加内容
`/opt/intel/lib
/opt/intel/mkl/lib/intel64`
注意把路径替换成自己的安装路径。 编辑完后执行
`sudo ldconfig`
用MKL记得在`Makefile.config`改一下`Set BLAS := MKL`

## 安装cuDNN（可选）
参考[NVIDIA CuDNN 安装说明](http://www.cnblogs.com/platero/p/4118139.html)这个不是必须的，只是为了加速

## 安装OpenCV
不要手动安装，网上有人写好了脚本，[脚本地址](https://github.com/jayrambhia/Install-OpenCV)
改权限让sh都可执行，然后执行ubuntu里面的脚本
`sudo ./opencv2_4_10.sh`

## 编译

    cp Makefile.config.example Makefile.config
    make all
    make test
    make runtest
先是复制了一份设置，用于更改自己的caffe配置，主要需要修改的参数包括
CPU_ONLY 是否只使用CPU模式，没有GPU没安装CUDA的同学可以打开这个选项
BLAS (使用intel mkl还是OpenBLAS)
MATLAB_DIR 如果需要使用MATLAB wrapper的同学需要指定matlab的安装路径,
DEBUG 是否使用debug模式，打开此选项则可以在eclipse或者NSight中debug程序
`make all -j 4`-j4是4线程并行。
`make test`检测你上一步的make没有错误，要全是OK
我编译的时候老是找不到opencv,所以换成了cmake
cmake的方法

    mkdir build
    cd build
    cmake ..
    make 
没有出问题，不要make install会安装一些东西造成位置问题，make runset也没用，不清楚为什么。
然后就可以跑跑example了。

## 安装pycaffe（可选）
官网提供了很好的方式，教你怎么下载依赖`for req in $(cat requirements.txt); do pip install $req; done`，不过可能出错，最好装Anaconda，我不想装，所以老老实实地装了各种依赖

    sudo apt-get install python-numpy python-scipy python-matplotlib python-sklearn python-skimage python-h5py python-protobuf python-leveldb python-networkx python-nose python-pandas python-gflags Cython ipython
    sudo apt-get install protobuf-c-compiler protobuf-compiler
编译一下`make pycaffe`
添加caffe里面的Python路径到PYTHONPATH：
`sudo gedit /etc/profile`
末尾添加： `export PYTHONPATH=/path/to/caffe/python:$PYTHONPATH`
用完整路径，不要用`~`
`source /etc/profile`# 使之生效
参考[Ubuntu14.04 安装Caffe（仅CPU）](http://blog.csdn.net/u011762313/article/details/47262549#%E9%85%8D%E7%BD%AEpycaffe)

## 遇到的问题
cuda开始好好的，过几天莫名其妙找不到了，先确定了`$PATH`环境变量包含了`/usr/local/cuda-7.5(对应的文件夹)/bin,/usr/local/cuda-7.5/include`,`LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH `,然后检查了nvcc版本，driver版本，没有不匹配的问题，然后查看nvidia-smi发现gpu居然找不到，估计是nvidia版本多了找不到是哪个？重启了一下显卡驱动`sudo dpkg-reconfigure nvidia-352`,352是你自己机器nvidia对应版本，可以通过`sudo dpkg --list | grep nvidia`来查看，然后reboot重启一下服务器，搞定。
http://blog.csdn.net/pirage/article/details/17553549


# 资源
http://suanfazu.com/t/caffe/281/5
http://mp.weixin.qq.com/s?__biz=MzAxNTE2MjcxNw==&mid=206508839&idx=1&sn=4dea40d781716da2f56d93fe23c158ab#rd
https://yufeigan.github.io/
http://blog.csdn.net/fengbingchun/article/details/49535873
http://blog.csdn.net/dengbingfeng/article/details/51469051
https://www.zhihu.com/question/27982282
https://github.com/BVLC/caffe/wiki/Development
