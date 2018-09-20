---
title: caffe.proto文件介绍
date: 2016-08-30 13:32:15
tags: caffe
categories: 机器学习
---
<blockquote class="blockquote-center">caffe.proto</blockquote>
<!-- more -->

## caffe.proto
repeated相当于数组，可以有多个，optional相当于0或者1

    message BlobShape {
    repeated int64 dim = 1 [packed = true];
    }
    message BlobProto {
      optional BlobShape shape = 7;
      repeated float data = 5 [packed = true];
      repeated float diff = 6 [packed = true];
    // 4D dimensions -- deprecated.  Use "shape" instead.
      optional int32 num = 1 [default = 0];
      optional int32 channels = 2 [default = 0];
      optional int32 height = 3 [default = 0];
      optional int32 width = 4 [default = 0];
    }
    // The BlobProtoVector is simply a way to pass multiple blobproto instances
    // around.
    message BlobProtoVector {
     repeated BlobProto blobs = 1;
    }

通过`BlobProtoVector`引用`BlobProto`,所以`blobs`有`BlobProto`的属性，`shape`,`data`,`diff`,其中`shape`属于`BlobShape`类，包含多个`dim`属性。`data`是数据（矩阵），`diff`是导数。

    message Datum {
     optional int32 channels = 1;
     optional int32 height = 2;
     optional int32 width = 3;
    // the actual image data, in bytes
     optional bytes data = 4;
     optional int32 label = 5;
    // Optionally, the datum could also hold float data.
     repeated float float_data = 6;
    // If true data contains an encoded image that need to be decoded
     optional bool encoded = 7 [default = false];
                }
`Datatum`真实的图片数据，`data`是byte或者float类型。

    message NetParameter {
         optional string name = 1; // consider giving the network a name
        // The input blobs to the network.
        repeated string input = 3;
      // The shape of the input blobs.
      repeated BlobShape input_shape = 8;
      // 4D input dimensions -- deprecated.  Use "shape" instead.
      // If specified, for each input blob there should be four
      // values specifying the num, channels, height and width of the input blob.
      // Thus, there should be a total of (4 * #input) numbers.
      repeated int32 input_dim = 4;
      // Whether the network will force every layer to carry out backward operation.
      // If set False, then whether to carry out backward is determined
      // automatically according to the net structure and learning rates.
      optional bool force_backward = 5 [default = false];
      // The current "state" of the network, including the phase, level, and stage.
      // Some layers may be included/excluded depending on this state and the states
      // specified in the layers' include and exclude fields.
      optional NetState state = 6;
      // Print debugging information about results while running Net::Forward,
      // Net::Backward, and Net::Update.
      optional bool debug_info = 7 [default = false];
      // The layers that make up the net.  Each of their configurations, including
      // connectivity and behavior, is specified as a LayerParameter.
      repeated LayerParameter layer = 100;  // ID 100 so layers are printed last.
      // DEPRECATED: use 'layer' instead.
      repeated V1LayerParameter layers = 2;
    }
 message NetParameter {
 
  optional string name = 1;   // 网络名称
  
  repeated string input = 3;  // 网络输入input blobs
 
  repeated BlobShape input_shape = 8; // The shape of the input blobs
  
  // 输入维度blobs，4维(num, channels, height and width)

  repeated int32 input_dim = 4;

  // 网络是否强制每层进行反馈操作开关

 // 如果设置为False，则会根据网络结构和学习率自动确定是否进行反馈操作

 optional bool force_backward = 5 [default = false];

// 网络的state，部分网络层依赖，部分不依赖，需要看具体网络

 optional NetState state = 6;

// 是否打印debug log

optional bool debug_info = 7 [default = false];

 // 网络层参数，Field Number 为100，所以网络层参数在最后

repeated LayerParameter layer = 100; 

 // 弃用: 用 'layer' 代替

 repeated V1LayerParameter layers = 2;
 }   
     
      message SolverParameter {
      optional string train_net = 1; // 训练网络的proto file
      optional string test_net = 2; // 测试网络的proto file
      optional int32 test_iter = 3 [default = 0]; // 每次测试时的迭代次数
      optional int32 test_interval = 4 [default = 0]; // 两次测试的间隔迭代次数
      optional bool test_compute_loss = 19 [default = false];
      optional float base_lr = 5; // 基本学习率
      optional int32 display = 6; // 两次显示的间隔迭代次数
      optional int32 max_iter = 7; // 最大迭代次数
      optional string lr_policy = 8; // 学习速率衰减方式
      optional float gamma = 9; // 关于梯度下降的一个参数
      optional float power = 10; // 计算学习率的一个参数
      optional float momentum = 11; // 动量
      optional float weight_decay = 12; // 权值衰减
      optional int32 stepsize = 13; // 学习速率的衰减步长
      optional int32 snapshot = 14 [default = 0]; // snapshot的间隔
      optional string snapshot_prefix = 15; // snapshot的前缀
      optional bool snapshot_diff = 16 [default = false]; // 是否对于 diff 进行 snapshot
      enum SolverMode {
        CPU = 0;
        GPU = 1;
      }
      optional SolverMode solver_mode = 17 [default = GPU]; // solver的模式，默认为GPU
      optional int32 device_id = 18 [default = 0]; // GPU的ID
      optional int64 random_seed = 20 [default = -1]; // 随机数种子
    }

`TransformationParameter`是对数据的预处理过程，包括缩放，切割，均值化等。
   

     message TransformationParameter {
          // For data pre-processing, we can do simple scaling and subtracting the
          // data mean, if provided. Note that the mean subtraction is always carried
          // out before scaling.
          optional float scale = 1 [default = 1];
          // Specify if we want to randomly mirror data.
          optional bool mirror = 2 [default = false];
          // Specify if we would like to randomly crop an image.
          optional uint32 crop_size = 3 [default = 0];
          // mean_file and mean_value cannot be specified at the same time
          optional string mean_file = 4;
          // if specified can be repeated once (would substract it from all the channels)
          // or can be repeated the same number of times as channels
          // (would subtract them from the corresponding channel)
          repeated float mean_value = 5;
            // Force the decoded image to have 3 color channels.
          optional bool force_color = 6 [default = false];
          // Force the decoded image to have 1 color channels.
          optional bool force_gray = 7 [default = false];
          optional string depth_noise_mnp = 8;
          optional uint32 depth_noise_seed = 9 [default = 0];
          optional string depth_noise_filelist = 10;
          optional uint32 depth_noise_cachesize = 11 [default = 100];
          //specifies in which channels the depth to be transformed is located
          repeated uint32 depth_noise_channels = 12;
        }



