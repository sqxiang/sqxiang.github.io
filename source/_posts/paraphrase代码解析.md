---
title: paraphrase代码解析
date: 2016-04-11 22:25:33
tags: 
 - python
 - 机器学习
 - LSTM
categories: 机器学习
---
<blockquote class="blockquote-center">paraphrase代码解析</blockquote>
<!-- more -->
## 初始化编码
    import sys
    reload(sys)
    sys.setdefaultencoding('utf8')
    
Python 里面的编码和解码也就是 unicode 和 str 这两种形式的相互转化。编码是 unicode -> str，相反的，解码就是 str ->unicode。剩下的问题就是确定何时需要进行编码或者解码了.关于文件开头的"编码指示"，也就是 `# -*- coding: -*-` 这个语句。Python 默认脚本文件都是 UTF-8 编码的，当文件中有非 UTF-8 编码范围内的字符的时候就要使用"编码指示"来修正. 关于 sys.defaultencoding，这个在解码没有明确指明解码方式的时候使用。当有如下代码时:

    #! /usr/bin/env python 
    # -*- coding: utf-8 -*- 
    s = '中文'  # 注意这里的 str 是 str 类型的，而不是 unicode 
    s.encode('gb18030') 

这句代码将 s 重新编码为 gb18030 的格式，即进行 unicode -> str 的转换。因为 s 本身就是 str 类型的，因此 Python 会自动的先将 s 解码为 unicode ，然后再编码成 gb18030。因为解码是python自动进行的，我们没有指明解码方式，python 就会使用 sys.defaultencoding 指明的方式来解码。很多情况下 sys.defaultencoding 是 ANSCII，如果 s 不是这个类型就会出错。拿上面的情况来说，我的 sys.defaultencoding 是 anscii，而 s 的编码方式和文件的编码方式一致，是 utf8 的，所以出错了，要改正如下：
一是明确的指示出 s 的编码方式 

    #! /usr/bin/env python 
    # -*- coding: utf-8 -*- 
    s = '中文' 
    s.decode('utf-8').encode('gb18030') 

二是更改 sys.defaultencoding 为文件的编码方式 

    #! /usr/bin/env python 
    # -*- coding: utf-8 -*- 
    import sys 
    reload(sys) # Python2.5 初始化后会删除 sys.setdefaultencoding这个方法，我们需要重新载入 
    sys.setdefaultencoding('utf-8') 
    str = '中文' 
    str.encode('gb18030')
    
## gensim里面的word2vec方法

其实参照[gensim官网](http://radimrehurek.com/gensim/models/word2vec.html)上的word2vec方法非常简单。让我们从简单的开始。
### 数据准备(input)
gensim的word2vec方法需要一列连续句子作为输入，如下:

    >>> sentences = [['first', 'sentence'], ['second', 'sentence']]
    >>> # train word2vec on the two sentences
    >>> model = gensim.models.Word2Vec(sentences, min_count=1)
得到的model就是用sentences训练好的词向量。
### model的用途
用`model[word]`可以直接得到word的词向量。
`model.save(/tmp/model)`可以保存model至指定文件夹。
`model.similarity(w1,w2)`得到w1和w2的相似度。
`model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)`得到的是`[('queen', 0.50882536)]`即最相似的单词和相似度。
`model.doesnt_match("breakfast cereal dinner lunch".split())`得到的不相似的单词，这里是`cereal`。
`model.train(more sentences)`可以在训练好的模型里加入更多训练数据进行训练。
### 训练过程

    model = gensim.models.Word2Vec(sentences, min_count=1)
调用word2Vec函数会分两步对sentences进行训练，首先第一步是收集单词并计算频率，然后构建内部词典树，第二步是训练神经网络模型得到词向量。分解过程:

    >>> model = gensim.models.Word2Vec() # an empty model, no training
    >>> model.build_vocab(some_sentences)  # can be a non-repeatable, 1-pass generator
    >>> model.train(other_sentences)  # can be a non-repeatable, 1-pass generator
训练的时候函数内加入参数可以加快训练，比如:

    model = Word2Vec(sentences, min_count=10)  # default value is 5
min_count是最小出现次数。一个单词出现次数要是过少，那么它的训练效果会很差，而且这个词也会是一个‘垃圾词’，因此设置最小出现次数可以剔除一些很少出现的词。

    model = Word2Vec(sentences, size=200)  # default value is 100
size不用多说，词向量的维数。训练数据越多可以设置越长的维数，会提升训练效果。

    model = Word2Vec(sentences, workers=4) # default = 1 worker = no parallelization
workers，并行计算，加快速度，当你下载了[Cython](http://cython.org/)才能用。
### 载入模型进行训练

    model = gensim.models.Word2Vec.load('/tmp/mymodel')
load可以载入训练好的模型，或者未训练的模型进行训练。

    model = Word2Vec.load_word2vec_format('/tmp/vectors.txt', binary=False)
    model = Word2Vec.load_word2vec_format('/tmp/vectors.bin.gz', binary=True)
你也可以用原生的C工具载入模型，txt文件或者二进制文件。
### 更多函数
请参考:<http://radimrehurek.com/gensim/models/word2vec.html>

## imdb.py文件解释
总共三个函数，如下:

    prepare_data(seqs, labels, maxlen=None)
其中`seqs=[[w1,w2,w3..],[w1,w2,w3..],[w1,w2,w3..]]`每个句子由单词在字典中的序号组成，类似`[10,25,27,188,34]`等形式，`labels=[0,1,1,0,...]`，函数目的在于将seqs中句子长度超过maxlen的删除(如果存在maxlen)，然后将所有句子长度统一成最长的句子长度，返回的x是交换坐标轴后的新的训练集，类似于

    [[12,15,70,34,...,99],  #maxlen*len(seqs)的矩阵，每一列是一个句子，不足添0
     [14,67,88,66,..,33],
     ...
     [0,0,0,34,6,...,0]]

第二个函数用于得到数据集，无需多说。

    get_dataset_file(dataset, default_dataset, origin)
第三个函数，load数据集

    load_data(path="imdb.pkl", n_words=100000, valid_portion=0.1, maxlen=None,sort_by_len=True)
加载imdb.pkl（可以换成你想用的数据集),n_words一般设置为字典长度，如果一个句子中的单词序号超过了n_words,单词将被代替为1，所以我们字典里面把0,1留出来当做unkown单词。valid_portion是验证集的比例。maxlen若存在则只留下小于naxlen的句子。最后返回的是train,valid,test，不言自明。

## creat文件解释
1.`build_dict`函数:
  利用google-newsvector-negative300训练词向量，并生成MSRPC_train训练集的词向量，返回词典和词向量。
  其中最重要的一行就是:`model=gensim.models.Word2Vec.load_word2vec_format(gensim_file,binary=True)`上面已经详细介绍过gensim的word2vec函数，不再赘述。其中对于训练集中有，词向量缺少的词，添加到miss_words。（貌似没什么用）
  返回的词典按照词频高到低排序，留出0,1两行作为UNK。(wordcount记录词数）
  
2.`grab_train_data(dictionary)`函数：
  利用上个函数得到的词典重新组织句子，每个句子用单词的序号列表表示，如`[15,76,16,34,67,88,99]`返回seqs是句子列表的集合，相邻两个句子是paraphrase比较的句子。返回labels是标签集合。每个句子若单词不在字典中就删除这个单词。提前打乱了句子次序。
  
3.`grab_test_data(dictionary)`函数：
  同上，重新组织test
  
4.`main`函数：
  dump数据成pkl文件。imdb_MSRPC_without1.pkl是训练集和测试集，imdb_MSRPC_without1_dict.pkl是词典，initial_emb_MSRPC.pkl是词典对应词向量。
