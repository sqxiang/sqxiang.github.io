---
title: python的各种函数
date: 2016-05-08 10:01:40
tags:
 - python
 - 函数
 - 文档
categories: python
---
<blockquote class="blockquote-center">python</blockquote>
<!--more-->
## struct

 - upack
 - pack
 - calcsize
{% codeblock lang:python %}
     pack(fmt, v1, v2, ...)     按照给定的格式(fmt)，把数据封装成字符串(实际上是类似于c结构体的字节流)
     unpack(fmt, string)       按照给定的格式(fmt)解析字节流string，返回解析出来的tuple
     calcsize(fmt)                 计算给定的格式(fmt)占用多少字节的内存
{% endcodeblock %}
结构体定义如下:
    
    struct Header
    {
    unsigned short id;
    char[4] tag;
    unsigned int version;
    unsigned int count;
    }

通过socket.recv接收到了一个上面的结构体数据，存在字符串s中，现在需要把它解析出来，可以使用unpack()函数.

import struct
id, tag, version, count = struct.unpack("!H4s2I", s)

上面的格式字符串中，!表示我们要使用网络字节顺序解析，因为我们的数据是从网络中接收到的，在网络上传送的时候它是网络字节顺序的.后面的H表示 一个unsigned short的id,4s表示4字节长的字符串，2I表示有两个unsigned int类型的数据.
就通过一个unpack，现在id, tag, version, count里已经保存好我们的信息了.
同样，也可以很方便的把本地数据再pack成struct格式.
ss = struct.pack("!H4s2I", id, tag, version, count);
pack函数就把id, tag, version, count按照指定的格式转换成了结构体Header，ss现在是一个字符串(实际上是类似于c结构体的字节流)，可以通过 socket.send(ss)把这个字符串发送出去.


