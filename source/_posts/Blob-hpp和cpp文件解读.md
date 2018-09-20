---
title: Blob.hpp和cpp文件解读
date: 2016-08-30 17:05:01
tags: caffe
categories: 机器学习
---

<blockquote class="blockquote-center">blob.hpp && blob.cpp</blockquote>
<!-- more -->

#ifndef CAFFE_BLOB_HPP_//防止头文件重复引用  
    #define CAFFE_BLOB_HPP_  
      
    #include <algorithm>  
    #include <string>  
    #include <vector>  
      
    /*common.hpp主要用来单例化Caffe类， 
    *并封装了boost和CUDA随机数生成的函数， 
    *提供了统一的接口 
    */  
    #include "common.hpp"  
    /*caffe.pb.h是google protocol buffer根据caffe.proto自动生成的。 
    *使用protocol buffer有这些好处，一方面可以用文本文件定义结构化的数据类型， 
    *另一方面可以生成查询效率更高、占空间更小的二进制文件 
    */  
    #include "caffe.pb.h"  
    //syncedmem主要用于分配内存和释放内存  
    #include "syncedmem.hpp"  
    //math_functions里面封装了很多cblas矩阵运算  
    #include "util/math_functions.hpp"  
      
    const int kMaxBlobAxes = INT_MAX;  
      
    namespace caffe {//命名空间为caffe  
      
        /* 
        *主要数据有两个data和diff，用num、channels、height和width 
        *这四个维度来确定数据的具体位置，做一些数据查询和Blobreshape的操作 
        */  
        template <typename Dtype>  
        class Blob {  
        public:  
            Blob()//blob的构造函数  
                : data_(), diff_(), count_(0), capacity_(0)//数据容量 {}
                //data_(), diff_()是用于存放数据的指针，  
            /*num_, channel_, height_, width_主要用来做定位offset和reshape处理。 
            *对于输入(n, c, h, w)位置的数据位置为((n*channels_+c)*height_+h)*width_+w，*可以依据位置取data_()或diff_()中的数据。 
            */  
            explicit Blob(const int num, const int channels, const int height, const int width);  
            explicit Blob(const vector<int>& shape);  
            
            /*Reshape函数的作用是改变一个blob的大小 
            *1.读入num_，channels_，height_，width_的大小  
            *2.计算count_：count_ = num_ * channels_ * height_ * width_;
            *3.如果count_不为0，则重新为data_和diff_分配一块空间  
            *如果count为0，则都初始化为NULL 
            */  
            void Reshape(const int num, const int channels, const int height,  const int width);  
            void Reshape(const vector<int>& shape);  
            void Reshape(const BlobShape& shape);  
            //ReshapeLike的作用是为data_和diff_ 重新分配一块空间，大小和另一个blob的一样   
            void ReshapeLike(const Blob& other);  
      
            inline string shape_string() const {  
                ostringstream stream;  //strem是一个string流
                for (int i = 0; i < shape_.size(); ++i) {  
                    stream << shape_[i] << " ";  //向流中传递shape_数据
                }  
                stream << "(" << count_ << ")";  //传递数据大小（四维相乘）
                return stream.str();  //返回字符串格式
            }  
            inline const vector<int>& shape() const { return shape_; }//返回shape  
      
            //返回第i个索引的shape,index可以是负数，  
            inline int shape(int index) const {  
                return shape_[CanonicalAxisIndex(index)];  //为了可以是负数
            }  
            inline int num_axes() const { return shape_.size(); }//返回shape的大小  
            inline int count() const { return count_; }//返回参数count  
      
            //计算一个slice的体积  
            ////为了统计Blob的容量（volume），或者是某一片（slice），从某个axis到具体某个axis的shape乘积。
            inline int count(int start_axis, int end_axis) const {  
            CHECK_LE(start_axis, end_axis);  
            CHECK_GE(start_axis, 0);  
            CHECK_GE(end_axis, 0);  
            CHECK_LE(start_axis, num_axes());  
            CHECK_LE(end_axis, num_axes()); 
                int count = 1;  
                for (int i = start_axis; i < end_axis; ++i) {  
                    count *= shape(i);  //num_,channel_,height_,width_都可以直接通过shape(i)访问
                }  
                return count;  
            }  
            //计算从从一个特定的axis到最后一个axis的slice的体积。 
            inline int count(int start_axis) const {  
                return count(start_axis, num_axes());  
            }  
      
            //Blob的Index是可以从负坐标开始读的,标准化索引，主要是对参数索引进行标准化，以满足要求  
      inline int CanonicalAxisIndex(int axis_index) const {  
        CHECK_GE(axis_index, -num_axes())  
            << "axis " << axis_index << " out of range for " << num_axes()  
            << "-D Blob with shape " << shape_string();  
        CHECK_LT(axis_index, num_axes())  
            << "axis " << axis_index << " out of range for " << num_axes()  
            << "-D Blob with shape " << shape_string();  
        if (axis_index < 0) {  
          return axis_index + num_axes();  
        }  
        return axis_index;  
      }  
      //Blob中的4个基本变量num,channel,height,width可以直接通过shape(0),shape(1),shape(2),shape(3)来访问  
      /// @brief Deprecated legacy shape accessor num: use shape(0) instead.  
      inline int num() const { return LegacyShape(0); }  
      /// @brief Deprecated legacy shape accessor channels: use shape(1) instead.  
      inline int channels() const { return LegacyShape(1); }  
      /// @brief Deprecated legacy shape accessor height: use shape(2) instead.  
      inline int height() const { return LegacyShape(2); }  
      /// @brief Deprecated legacy shape accessor width: use shape(3) instead.  
      inline int width() const { return LegacyShape(3); }  
    //data_维数不大于4时才能使用，功能同shape()类似。  
      inline int LegacyShape(int index) const {  
        CHECK_LE(num_axes(), 4)  
            << "Cannot use legacy accessors on Blobs with > 4 axes.";  
        CHECK_LT(index, 4);  //lower than判断是否小于
        CHECK_GE(index, -4);  
        if (index >= num_axes() || index < -num_axes()) {  
          // Axis is out of range, but still in [0, 3] (or [-4, -1] for reverse  
          // indexing) -- this special case simulates the one-padding used to fill  
          // extraneous axes of legacy blobs.  
          return 1;  
        }  
        return shape(index);  
      }  
      
      //计算offset,offset计算的方式也支持两种方式，一种直接指定n,c,h,w或者放到一个vector中进行计算，  
      //偏差是根据对应的n,c,h,w，返回的offset是((n*channels()+c)*height()+h)*width()+w  
      inline int offset(const int n, const int c = 0, const int h = 0,  
          const int w = 0) const {  
        CHECK_GE(n, 0);   //greater or equal，判断是否大于等于
        CHECK_LE(n, num());  //lower or equal ，判断是否小于等于
        CHECK_GE(channels(), 0);  
        CHECK_LE(c, channels());  
        CHECK_GE(height(), 0);  
        CHECK_LE(h, height());  
        CHECK_GE(width(), 0);  
        CHECK_LE(w, width());  
        return ((n * channels() + c) * height() + h) * width() + w;  
      }  
      
      inline int offset(const vector<int>& indices) const {  
        CHECK_LE(indices.size(), num_axes());  
        int offset = 0;  
        for (int i = 0; i < num_axes(); ++i) {  
          offset *= shape(i);  
          if (indices.size() > i) {  
            CHECK_GE(indices[i], 0);  
            CHECK_LT(indices[i], shape(i));  
            offset += indices[i];  
          }  
        }  
        return offset;  
      }  
    /** 
     *从source拷贝数据。copy_diff作为标志来区分是拷贝data还是拷贝diff 
                *1.如果是GPU： 如果是拷贝diff：调用cudaMemcpy函数将source的diff拷贝过来，否则拷贝data  
                *2.如果是CPU： 如果是拷贝diff：调用memcpy函数将source的diff拷贝过来 否则拷贝data 
                */  
        void CopyFrom(const Blob<Dtype>& source, bool copy_diff = false,bool reshape = false);  
        //从cpu访问数据data  
    inline Dtype data_at(const int n, const int c, const int h, const int w) const {  
                return cpu_data()[offset(n, c, h, w)];  
                }  
                //从cpu访问数据diff  
     inline Dtype diff_at(const int n, const int c, const int h, const int w) const {  
                return cpu_diff()[offset(n, c, h, w)];  
                }  
                //从cpu访问数据data  
    inline Dtype data_at(const vector<int>& index) const {  
                return cpu_data()[offset(index)];  
                }  
                //从cpu访问数据diff  
     inline Dtype diff_at(const vector<int>& index) const {  
                    return cpu_diff()[offset(index)];  
                }  
                //从cpu访问数据data  
     inline const shared_ptr<SyncedMemory>& data() const {  
                    return data_;  
                }  
                //从cpu访问数据diff  
    inline const shared_ptr<SyncedMemory>& diff() const {  
                    return diff_;  
                }  
     /**调用SyncedMemory的函数，来返回数据的指针;前两个调用to_cpu(),返回cpu_ptr； 
            *第一个对于data对象，第二个对于diff对象 
            *后两个调用to_gpu(),返回gpu_ptr；第一个对于data对象，第二个对于diff对象 
               */  
            void set_cpu_data(Dtype* data);  
            const Dtype* cpu_data() const;  
            const Dtype* gpu_data() const;  
            const Dtype* cpu_diff() const;  
            const Dtype* gpu_diff() const;  
            Dtype* mutable_cpu_data();  
            Dtype* mutable_gpu_data();  
            Dtype* mutable_cpu_diff();  
            Dtype* mutable_gpu_diff();  
        /**更新data_的数据，就是减去diff_的数据。  
           *1.判断blob的位置 
        *2.调用caffe_axpy：在math_functions.cpp可以找到该函数的实现，其实这函数也是封装了mkl的函数。这里调用是为了实现了两个向量的减法。  
        *3.调用caffe_gpu_axpy：在math_functions.cpp可以找到该函数的实现，其实这函数也是封装了cublas的函数。这里调用是为了实现了两个向量的减法。 
           */  
            void Update();  
            /**功能：从proto读数据进来，其实就是反序列化  
            *1.先把blob的大小改变一下  
            *2.得到cpu中数据的地址  
            *3.用proto中的data覆盖blob中的data  
            *4.用proto中的diff覆盖blob中的diff 
            */  
            void FromProto(const BlobProto& proto, bool reshape = true);  
            //把blob数据保存到proto中  
            void ToProto(BlobProto* proto, bool write_diff = false) const;  
      
            //计算绝对值的data总和（L1范数）。  
            Dtype asum_data() const;  
            //计算绝对值的diff总和（L1范数）。  
            Dtype asum_diff() const;  
            //计算绝对值的data总和（L2范数）。  
            Dtype sumsq_data() const;  
            //计算绝对值的diff总和（L2范数）。  
            Dtype sumsq_diff() const;  
            //通过常量因子缩放blob data  
            void scale_data(Dtype scale_factor);  
            ////通过常量因子缩放blob diff  
            void scale_diff(Dtype scale_factor);  
      
            //从other的blob复制data和diff的值  
            void ShareData(const Blob& other);  
            void ShareDiff(const Blob& other);  
            bool ShapeEquals(const BlobProto& other);  
      
        protected:  
            shared_ptr<SyncedMemory> data_;// 存放数据  
            shared_ptr<SyncedMemory> diff_;//存放梯度  
            vector<int> shape_;//存放形状  
            int count_;//数据个数  4维相乘
            int capacity_;//数据容量  
      
            DISABLE_COPY_AND_ASSIGN(Blob);  
        };  // class Blob  
      
    }  // namespace caffe  
      
    #endif  // CAFFE_BLOB_HPP_  

首先是 data_ 指针，指针类型是shared_ptr，属于boost库的一个智能指针，这一部分主要用来申请内存存储data，data主要是正向传播的时候用的。同理， diff_ 主要用来存储偏差，update data， shape_data 和 shape_ 都是存储Blob的形状，一个是老版本一个是新版本。 count 表示Blob中的元素个数，也就是 个数*通道数*高度*宽度 , capacity 表示当前的元素个数，因为Blob可能会reshape。 
Blob类里面有重载很多个 count()函数，主要还是为了统计Blob的容量（volume），或者是某一片（slice），从某个axis到具体某个axis的shape乘积。

    void FromProto(const BlobProto& proto, bool reshape = true);
    void ToProto(BlobProto* proto, bool write_diff = false) const;

这两个函数主要是将数据序列化，存储到BlobProto，这里说到Proto是谷歌的一个数据序列化的存储格式，可以实现语言、平台无关、可扩展的序列化结构数据格式。Caffe里面数据的存储都采用这一结构，这里就不深入展开，具体可以参照这篇文章，对于proto的序列化和反序列都讲解的非常详细 [参考](http://www.w2bc.com/Article/34963)
  //把blob数据保存到proto中  
            void ToProto(BlobProto* proto, bool write_diff = false) const;

对任何Blob<Dtype> blobs调用的都是这个类，成员变量就包括
shared_ptr<SyncedMemory> data_;// 存放数据  
            shared_ptr<SyncedMemory> diff_;//存放梯度  
            vector<int> shape_;//存放形状  
            int count_;//数据个数  4维相乘
            int capacity_;//数据容量 
            
    
        #include <climits>  
        #include <vector>  
          
        #include "caffe/blob.hpp"  
        #include "caffe/common.hpp"  
        #include "caffe/syncedmem.hpp"  
        #include "caffe/util/math_functions.hpp"  
          
        namespace caffe {  
          
        template <typename Dtype>  
        //该函数将num,channels,height,width传递给vector shape_   
        void Blob<Dtype>::Reshape(const int num, const int channels, const int height,  
            const int width) {  
          vector<int> shape(4);  
          shape[0] = num;  
          shape[1] = channels;  
          shape[2] = height;  
          shape[3] = width;  
          Reshape(shape);  
        }  
          
        template <typename Dtype>  
        void Blob<Dtype>::Reshape(const vector<int>& shape) {  
          CHECK_LE(shape.size(), kMaxBlobAxes);  
          count_ = 1;  
          shape_.resize(shape.size());//重新定义vector shape_ 的size  
          for (int i = 0; i < shape.size(); ++i) {  
            CHECK_GE(shape[i], 0);//确保shape 每个元素为正数  
            CHECK_LE(shape[i], INT_MAX / count_) << "blob size exceeds INT_MAX";  
            count_ *= shape[i];  
            shape_[i] = shape[i];  
          }  
          //由于count_超过了当前capacity_ 因此需要重新分配内存空间  
          if (count_ > capacity_) {  
            capacity_ = count_;  
            data_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));  
            diff_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));  
          }  
        }  
          
        template <typename Dtype>// BlobShape 在caffe.proto 中定义  
        void Blob<Dtype>::Reshape(const BlobShape& shape) {  
          CHECK_LE(shape.dim_size(), kMaxBlobAxes);  
          vector<int> shape_vec(shape.dim_size());  
          for (int i = 0; i < shape.dim_size(); ++i) {  
            shape_vec[i] = shape.dim(i);//dim 包含num，channels，height， width  
          }  
          Reshape(shape_vec);//用protobuf传递来dim 对shape_ 进行reshape  
        }  
        //用已知的Blob的shape来对shape_ 进行reshape  
        template <typename Dtype>  
        void Blob<Dtype>::ReshapeLike(const Blob<Dtype>& other) {  
          Reshape(other.shape());  
        }  
        //用num，channels，height， width 初始化  
        template <typename Dtype>  
        Blob<Dtype>::Blob(const int num, const int channels, const int height,  
            const int width)  
          // capacity_ must be initialized before calling Reshape  
          : capacity_(0) {  
          Reshape(num, channels, height, width);  
        }  
        //用shape 初始化  
        template <typename Dtype>  
        Blob<Dtype>::Blob(const vector<int>& shape)  
          // capacity_ must be initialized before calling Reshape  
          : capacity_(0) {  
          Reshape(shape);  
        }  
        //返回cpu 中的数据  
        template <typename Dtype>  
        const Dtype* Blob<Dtype>::cpu_data() const {  
          CHECK(data_);  
          return (const Dtype*)data_->cpu_data();  
        }  
        // 清空cpu 数据  
        template <typename Dtype>  
        void Blob<Dtype>::set_cpu_data(Dtype* data) {  
          CHECK(data);  
          data_->set_cpu_data(data);  
        }  
        //返回gpu 中的数据  
        template <typename Dtype>  
        const Dtype* Blob<Dtype>::gpu_data() const {  
          CHECK(data_);  
          return (const Dtype*)data_->gpu_data();  
        }  
        //反向传播导数diff_ 操作函数,返回cpu 中的数据  
        template <typename Dtype>  
        const Dtype* Blob<Dtype>::cpu_diff() const {  
          CHECK(diff_);  
          return (const Dtype*)diff_->cpu_data();  
        }  
        //返回gpu 中的数据  
        template <typename Dtype>  
        const Dtype* Blob<Dtype>::gpu_diff() const {  
          CHECK(diff_);  
          return (const Dtype*)diff_->gpu_data();  
        }  
          
        template <typename Dtype>  
        Dtype* Blob<Dtype>::mutable_cpu_data() {  
          CHECK(data_);  
          return static_cast<Dtype*>(data_->mutable_cpu_data());  
        }  
          
        template <typename Dtype>  
        Dtype* Blob<Dtype>::mutable_gpu_data() {  
          CHECK(data_);  
          return static_cast<Dtype*>(data_->mutable_gpu_data());  
        }  
          
        template <typename Dtype>  
        Dtype* Blob<Dtype>::mutable_cpu_diff() {  
          CHECK(diff_);  
          return static_cast<Dtype*>(diff_->mutable_cpu_data());  
        }  
          
        template <typename Dtype>  
        Dtype* Blob<Dtype>::mutable_gpu_diff() {  
          CHECK(diff_);  
          return static_cast<Dtype*>(diff_->mutable_gpu_data());  
        }  
        //当前的blob 的data_ 指向已知blob的数据  
        template <typename Dtype>  
        void Blob<Dtype>::ShareData(const Blob& other) {  
          CHECK_EQ(count_, other.count());  
          data_ = other.data();  
        }  
        //当前的blob 的diff_ 指向已知blob的反向传播导数  
        template <typename Dtype>  
        void Blob<Dtype>::ShareDiff(const Blob& other) {  
          CHECK_EQ(count_, other.count());  
          diff_ = other.diff();  
        }  
          
        // The "update" method is used for parameter blobs in a Net, which are stored  
        // as Blob<float> or Blob<double> -- hence we do not define it for  
        // Blob<int> or Blob<unsigned int>.  
        template <> void Blob<unsigned int>::Update() { NOT_IMPLEMENTED; }  
        template <> void Blob<int>::Update() { NOT_IMPLEMENTED; }  
        //Updata函数用于参数blob的更新（weight，bias 等减去对应的导数）  
        template <typename Dtype>  
        void Blob<Dtype>::Update() {  
          // We will perform update based on where the data is located.  
          switch (data_->head()) {  
          case SyncedMemory::HEAD_AT_CPU://数据在cpu上，则在cpu上进行计算  
            // perform computation on CPU  
            caffe_axpy<Dtype>(count_, Dtype(-1),  
                static_cast<const Dtype*>(diff_->cpu_data()),  
                static_cast<Dtype*>(data_->mutable_cpu_data()));  
            break;  
          case SyncedMemory::HEAD_AT_GPU:  
          case SyncedMemory::SYNCED:  
        #ifndef CPU_ONLY//如果没有定义CPU_ONLY，且数据在gpu上，则在gpu上进行计算  
            // perform computation on GPU  
            caffe_gpu_axpy<Dtype>(count_, Dtype(-1),  
                static_cast<const Dtype*>(diff_->gpu_data()),  
                static_cast<Dtype*>(data_->mutable_gpu_data()));  
        #else  
            NO_GPU;  
        #endif  
            break;  
          default:  
            LOG(FATAL) << "Syncedmem not initialized.";  
          }  
        }  
          
        template <> unsigned int Blob<unsigned int>::asum_data() const {  
          NOT_IMPLEMENTED;  
          return 0;  
        }  
          
        template <> int Blob<int>::asum_data() const {  
          NOT_IMPLEMENTED;  
          return 0;  
        }  
        //返回data_ 中所有 element 的绝对值之和  
        template <typename Dtype>  
        Dtype Blob<Dtype>::asum_data() const {  
          if (!data_) { return 0; }  
          switch (data_->head()) {  
          case SyncedMemory::HEAD_AT_CPU:  
            return caffe_cpu_asum(count_, cpu_data());  
          case SyncedMemory::HEAD_AT_GPU:  
          case SyncedMemory::SYNCED:  
        #ifndef CPU_ONLY  
          {  
            Dtype asum;  
            caffe_gpu_asum(count_, gpu_data(), &asum);  
            return asum;  
          }  
        #else  
            NO_GPU;  
        #endif  
          case SyncedMemory::UNINITIALIZED:  
            return 0;  
          default:  
            LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();  
          }  
          return 0;  
        }  
          
        template <> unsigned int Blob<unsigned int>::asum_diff() const {  
          NOT_IMPLEMENTED;  
          return 0;  
        }  
          
        template <> int Blob<int>::asum_diff() const {  
          NOT_IMPLEMENTED;  
          return 0;  
        }  
        //返回diff_ 中所有 element 的绝对值之和  
        template <typename Dtype>  
        Dtype Blob<Dtype>::asum_diff() const {  
          if (!diff_) { return 0; }  
          switch (diff_->head()) {  
          case SyncedMemory::HEAD_AT_CPU:  
            return caffe_cpu_asum(count_, cpu_diff());  
          case SyncedMemory::HEAD_AT_GPU:  
          case SyncedMemory::SYNCED:  
        #ifndef CPU_ONLY  
          {  
            Dtype asum;  
            caffe_gpu_asum(count_, gpu_diff(), &asum);  
            return asum;  
          }  
        #else  
            NO_GPU;  
        #endif  
          case SyncedMemory::UNINITIALIZED:  
            return 0;  
          default:  
            LOG(FATAL) << "Unknown SyncedMemory head state: " << diff_->head();  
          }  
          return 0;  
        }  
          
        template <> unsigned int Blob<unsigned int>::sumsq_data() const {  
          NOT_IMPLEMENTED;  
          return 0;  
        }  
          
        template <> int Blob<int>::sumsq_data() const {  
          NOT_IMPLEMENTED;  
          return 0;  
        }  
        //返回 data_ 中所有 element 的平方和  
        template <typename Dtype>  
        Dtype Blob<Dtype>::sumsq_data() const {  
          Dtype sumsq;  
          const Dtype* data;  
          if (!data_) { return 0; }  
          switch (data_->head()) {  
          case SyncedMemory::HEAD_AT_CPU:  
            data = cpu_data();  
            sumsq = caffe_cpu_dot(count_, data, data);  
            break;  
          case SyncedMemory::HEAD_AT_GPU:  
          case SyncedMemory::SYNCED:  
        #ifndef CPU_ONLY  
            data = gpu_data();  
            caffe_gpu_dot(count_, data, data, &sumsq);  
        #else  
            NO_GPU;  
        #endif  
            break;  
          case SyncedMemory::UNINITIALIZED:  
            return 0;  
          default:  
            LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();  
          }  
          return sumsq;  
        }  
          
        template <> unsigned int Blob<unsigned int>::sumsq_diff() const {  
          NOT_IMPLEMENTED;  
          return 0;  
        }  
          
        template <> int Blob<int>::sumsq_diff() const {  
          NOT_IMPLEMENTED;  
          return 0;  
        }  
        //返回 diff_ 中所有 element 的平方和  
        template <typename Dtype>  
        Dtype Blob<Dtype>::sumsq_diff() const {  
          Dtype sumsq;  
          const Dtype* diff;  
          if (!diff_) { return 0; }  
          switch (diff_->head()) {  
          case SyncedMemory::HEAD_AT_CPU:  
            diff = cpu_diff();  
            sumsq = caffe_cpu_dot(count_, diff, diff);  
            break;  
          case SyncedMemory::HEAD_AT_GPU:  
          case SyncedMemory::SYNCED:  
        #ifndef CPU_ONLY  
            diff = gpu_diff();  
            caffe_gpu_dot(count_, diff, diff, &sumsq);  
            break;  
        #else  
            NO_GPU;  
        #endif  
          case SyncedMemory::UNINITIALIZED:  
            return 0;  
          default:  
            LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();  
          }  
          return sumsq;  
        }  
          
        template <> void Blob<unsigned int>::scale_data(unsigned int scale_factor) {  
          NOT_IMPLEMENTED;  
        }  
          
        template <> void Blob<int>::scale_data(int scale_factor) {  
          NOT_IMPLEMENTED;  
        }  
        // 给data乘以scale_factor  
        template <typename Dtype>  
        void Blob<Dtype>::scale_data(Dtype scale_factor) {  
          Dtype* data;  
          if (!data_) { return; }  
          switch (data_->head()) {  
          case SyncedMemory::HEAD_AT_CPU:  
            data = mutable_cpu_data();  
            caffe_scal(count_, scale_factor, data);  
            return;  
          case SyncedMemory::HEAD_AT_GPU:  
          case SyncedMemory::SYNCED:  
        #ifndef CPU_ONLY  
            data = mutable_gpu_data();  
            caffe_gpu_scal(count_, scale_factor, data);  
            return;  
        #else  
            NO_GPU;  
        #endif  
          case SyncedMemory::UNINITIALIZED:  
            return;  
          default:  
            LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();  
          }  
        }  
          
        template <> void Blob<unsigned int>::scale_diff(unsigned int scale_factor) {  
          NOT_IMPLEMENTED;  
        }  
          
        template <> void Blob<int>::scale_diff(int scale_factor) {  
          NOT_IMPLEMENTED;  
        }  
        // 给diff乘以scale_factor  
        template <typename Dtype>  
        void Blob<Dtype>::scale_diff(Dtype scale_factor) {  
          Dtype* diff;  
          if (!diff_) { return; }  
          switch (diff_->head()) {  
          case SyncedMemory::HEAD_AT_CPU:  
            diff = mutable_cpu_diff();  
            caffe_scal(count_, scale_factor, diff);  
            return;  
          case SyncedMemory::HEAD_AT_GPU:  
          case SyncedMemory::SYNCED:  
        #ifndef CPU_ONLY  
            diff = mutable_gpu_diff();  
            caffe_gpu_scal(count_, scale_factor, diff);  
            return;  
        #else  
            NO_GPU;  
        #endif  
          case SyncedMemory::UNINITIALIZED:  
            return;  
          default:  
            LOG(FATAL) << "Unknown SyncedMemory head state: " << diff_->head();  
          }  
        }  
        //BlobProto 是定义在caffe.proto 中的一个message，其字段有 data,diff,shape,num,channels,height,width  
        template <typename Dtype>  
        bool Blob<Dtype>::ShapeEquals(const BlobProto& other) {  
          if (other.has_num() || other.has_channels() ||  
              other.has_height() || other.has_width()) {  
            // Using deprecated 4D Blob dimensions --  
            // shape is (num, channels, height, width).  
            // Note: we do not use the normal Blob::num(), Blob::channels(), etc.  
            // methods as these index from the beginning of the blob shape, where legacy  
            // parameter blobs were indexed from the end of the blob shape (e.g., bias  
            // Blob shape (1 x 1 x 1 x N), IP layer weight Blob shape (1 x 1 x M x N)).  
            return shape_.size() <= 4 &&  
                   LegacyShape(-4) == other.num() &&  
                   LegacyShape(-3) == other.channels() &&  
                   LegacyShape(-2) == other.height() &&  
                   LegacyShape(-1) == other.width();  
          }  
          vector<int> other_shape(other.shape().dim_size());  
          for (int i = 0; i < other.shape().dim_size(); ++i) {  
            other_shape[i] = other.shape().dim(i);  
          }  
          return shape_ == other_shape;  
        }//检查当前的blob和已知的 other 的 shape 是否相同，相同返回true  
          
        template <typename Dtype>  
        void Blob<Dtype>::CopyFrom(const Blob& source, bool copy_diff, bool reshape) {  
          if (source.count() != count_ || source.shape() != shape_) {  
            if (reshape) {  
              ReshapeLike(source);  
            } else {  
              LOG(FATAL) << "Trying to copy blobs of different sizes.";  
            }  
          }  
          switch (Caffe::mode()) {  
          case Caffe::GPU:  
            if (copy_diff) {  
              caffe_copy(count_, source.gpu_diff(),  
                  static_cast<Dtype*>(diff_->mutable_gpu_data()));  
            } else {  
              caffe_copy(count_, source.gpu_data(),  
                  static_cast<Dtype*>(data_->mutable_gpu_data()));  
            }  
            break;  
          case Caffe::CPU:  
            if (copy_diff) {  
              caffe_copy(count_, source.cpu_diff(),  
                  static_cast<Dtype*>(diff_->mutable_cpu_data()));  
            } else {  
              caffe_copy(count_, source.cpu_data(),  
                  static_cast<Dtype*>(data_->mutable_cpu_data()));  
            }  
            break;  
          default:  
            LOG(FATAL) << "Unknown caffe mode.";  
          }  
        }//从source 拷贝数据,copy_diff控制是拷贝diff还是data  
          
        template <typename Dtype>  
        void Blob<Dtype>::FromProto(const BlobProto& proto, bool reshape) {  
          if (reshape) {  
            vector<int> shape;  
            if (proto.has_num() || proto.has_channels() ||  
                proto.has_height() || proto.has_width()) {  
              // Using deprecated 4D Blob dimensions --  
              // shape is (num, channels, height, width).  
              shape.resize(4);  
              shape[0] = proto.num();  
              shape[1] = proto.channels();  
              shape[2] = proto.height();  
              shape[3] = proto.width();  
            } else {  
              shape.resize(proto.shape().dim_size());  
              for (int i = 0; i < proto.shape().dim_size(); ++i) {  
                shape[i] = proto.shape().dim(i);  
              }  
            }  
            Reshape(shape);  
          } else {//如果不做reshape要求当前的blob的shape和proto传入的shape相同  
            CHECK(ShapeEquals(proto)) << "shape mismatch (reshape not set)";  
          }  
          // copy data  
          Dtype* data_vec = mutable_cpu_data();  
          for (int i = 0; i < count_; ++i) {  
            data_vec[i] = proto.data(i);  
          }//将proto传入的data拷贝到cpu数据  
          if (proto.diff_size() > 0) {  
            Dtype* diff_vec = mutable_cpu_diff();  
            for (int i = 0; i < count_; ++i) {  
              diff_vec[i] = proto.diff(i);  
            }//将proto传入的diff 拷贝到cpu数据  
          }  
        }  
          
        template <typename Dtype>  
        void Blob<Dtype>::ToProto(BlobProto* proto, bool write_diff) const {  
          proto->clear_shape();  
          for (int i = 0; i < shape_.size(); ++i) {  
            proto->mutable_shape()->add_dim(shape_[i]);  
          }  
          proto->clear_data();  
          proto->clear_diff();  
          const Dtype* data_vec = cpu_data();  
          for (int i = 0; i < count_; ++i) {  
            proto->add_data(data_vec[i]);//将data写入proto  
          }  
          if (write_diff) {  
            const Dtype* diff_vec = cpu_diff();  
            for (int i = 0; i < count_; ++i) {  
              proto->add_diff(diff_vec[i]);//将diff写入proto  
            }  
          }  
        }  
          
        INSTANTIATE_CLASS(Blob);  
        template class Blob<int>;  
        template class Blob<unsigned int>;  
          
        }  // namespace caffe  
