---
title: layer分类介绍
date: 2016-08-30 17:29:02
tags: caffe
categories: 机器学习
---

五大Layer派生类型
Caffe十分强调网络的层次性，也就是说卷积操作，非线性变换（ReLU等），Pooling，权值连接等全部都由某一种Layer来表示。具体来说分为5大类Layer
1.NeuronLayer类 定义于neuron_layers.hpp中，其派生类主要是元素级别的运算（比如Dropout运算，激活函数ReLu，Sigmoid等），运算均为同址计算（in-place computation，返回值覆盖原值而占用新的内存）。
2.LossLayer类 定义于loss_layers.hpp中，其派生类会产生loss，只有这些层能够产生loss。
3.数据层 定义于data_layer.hpp中，作为网络的最底层，主要实现数据格式的转换。
4.特征表达层（我自己分的类）定义于vision_layers.hpp（为什么叫vision这个名字，我目前还不清楚），实现特征表达功能，更具体地说包含卷积操作，Pooling操作，他们基本都会产生新的内存占用（Pooling相对较小）。
5.网络连接层和激活函数（我自己分的类）定义于common_layers.hpp，Caffe提供了单个层与多个层的连接，并在这个头文件中声明。这里还包括了常用的全连接层InnerProductLayer类。
Layer的重要成员函数
在Layer内部，数据主要有两种传递方式，正向传导（Forward）和反向传导（Backward）。Forward和Backward有CPU和GPU（部分有）两种实现。Caffe中所有的Layer都要用这两种方法传递数据。
virtual void Forward(const vector<Blob<Dtype>*> &bottom,                     
                     vector<Blob<Dtype>*> *top) = 0;virtual void Backward(const vector<Blob<Dtype>*> &top,
                      const vector<bool> &propagate_down, 
                      vector<Blob<Dtype>*> *bottom) = 0;
Layer类派生出来的层类通过这实现这两个虚函数，产生了各式各样功能的层类。Forward是从根据bottom计算top的过程，Backward则相反（根据top计算bottom）。注意这里为什么用了一个包含Blob的容器（vector），对于大多数Layer来说输入和输出都各连接只有一个Layer，然而对于某些Layer存在一对多的情况，比如LossLayer和某些连接层。在网路结构定义文件（*.proto）中每一层的参数bottom和top数目就决定了vector中元素数目。
layers {
  bottom: "decode1neuron"   // 该层底下连接的第一个Layer
  bottom: "flatdata"        // 该层底下连接的第二个Layer
  top: "l2_error"           // 该层顶上连接的一个Layer
  name: "loss"              // 该层的名字
  type: EUCLIDEAN_LOSS      // 该层的类型
  loss_weight: 0}
2.2.3. Layer的重要成员变量
loss
vector<Dtype> loss_;
每一层又有一个loss_值，只不多大多数Layer都是0，只有LossLayer才可能产生非0的loss_。计算loss是会把所有层的loss_相加。
learnable parameters
vector<shared_ptr<Blob<Dtype> > > blobs_;
前面提到过的，Layer学习到的参数。
2.3. Net：
Net用容器的形式将多个Layer有序地放在一起，其自身实现的功能主要是对逐层Layer进行初始化，以及提供Update( )的接口（更新网络参数），本身不能对参数进行有效地学习过程。
vector<shared_ptr<Layer<Dtype> > > layers_;
同样Net也有它自己的
vector<Blob<Dtype>*>& Forward(const vector<Blob<Dtype>* > & bottom,
                              Dtype* loss = NULL);void Net<Dtype>::Backward();
他们是对整个网络的前向和方向传导，各调用一次就可以计算出网络的loss了
