---
title: 机器学习算法介绍(八)--k均值算法
date: 2017-03-12 10:37:23
tags:
 - 机器学习
 - k均值
categories: 机器学习
---
<blockquote class="blockquote-center">k-means</blockquote>
<!-- more -->
1、从D中随机取k个元素，作为k个簇的各自的中心。
2、分别计算剩下的元素到k个簇中心的相异度，将这些元素分别划归到相异度最低的簇。
3、根据聚类结果，重新计算k个簇各自的中心，计算方法是取簇中所有元素各自维度的算术平均数。
4、将D中全部元素按照新的中心重新聚类。
5、重复第4步，直到聚类结果不再变化。
6、将结果输出

[算法杂货铺——k均值聚类(K-means)](http://www.cnblogs.com/leoo2sk/archive/2010/09/20/k-means.html)
