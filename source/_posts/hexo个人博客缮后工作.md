---
title: hexo个人博客缮后工作
date: 2016-04-14 16:46:17
tags: hexo
categories: 技术
---
<blockquote class="blockquote-center">博客搭建之后</blockquote>
<!-- more -->
### 数学公式
今天想在hexo里面试一下latex的公式管不管用，果然是不行的，于是查了很多资料，找到一篇不错的博客:[搭建一个支持latex的hexo博客](http://blog.csdn.net/emptyset110/article/details/50123231)
仿照他的方法，成功实现了公式的输入。
#### 利用mathJax来渲染公式
下载mathJax很简单的两条命令:

    npm install hexo-math --save
    hexo math install
下载后`node_module`里面会多一个`hexo-math`的文件夹，然后就可以输入公式了，测试一下:
$$
\begin{eqnarray}
\nabla\cdot\vec{E} &=& \frac{\rho}{\epsilon_0} \\
\nabla\cdot\vec{B} &=& 0 \\
\nabla\times\vec{E} &=& -\frac{\partial B}{\partial t} \\
\nabla\times\vec{B} &=& \mu_0\left(\vec{J}+\epsilon_0\frac{\partial E}{\partial t} \right)
\end{eqnarray}
$$
出现了些瑕疵，不能换行，转义问题，修改文件`marked.js`，在`place_of_yourblog/node_modules/hexo-renderer-marked/node_modules/marked/lib`里面，修改如下:
Step 1:

    escape: /^\\([\\`*{}\[\]()# +\-.!_>])/,
替换成:

    escape: /^\\([`*\[\]()# +\-.!_>])/,
这一步是在原基础上取消了对\\,\{,\}的转义(escape)
Step 2:

    em: /^\b_((?:[^_]|__)+?)_\b|^\*((?:\*\*|[\s\S])+?)\*(?!\*)/,
替换成:

    em:/^\*((?:\*\*|[\s\S])+?)\*(?!\*)/,
又出了瑕疵，博客头部多了一行空白，还有符号---->，转义问题吧。待会解决。
(2016年4月16日更新)
好吧，没能解决，应该是mathJax的问题，装了安装包后就会在博客头部多一条线。我用的是[TKL](https://github.com/SuperKieran/TKL)主题，貌似换成其他主题就没问题了，又搜了搜，发现[next](https://github.com/iissnan/hexo-theme-next)主题也不错，而且有专门的[文档](http://theme-next.iissnan.com/)简直是新手福音，居然没早点选择。
### 换成next主题
安装next的theme：

    $ cd your-hexo-site
    $ git clone https://github.com/iissnan/hexo-theme-next themes/next
把全局的`_config.yml`里面改成:`themes:next`就完成了安装。记得`git pull`，因为你clone下来的是别人的branch，要合并到自己的branch里面。当然有时候合并不了，可以直接把next里面的git文件都删除`sudo rm -r .git*`
　　然后就可以开心的玩主题了。所有配置[文档](http://theme-next.iissnan.com/)都写的很清楚。妈蛋我也是被折腾的，早点换成这个主题就好了。
　　对了，把mathJax的包可以删除了，删除命令:`npm uninstall hexo-math`主题里面有引用的地方，参看文档开启一下就好。
### 文章链接更改
如果不想文章链接是`year/month/day/title`的形式，还是改全局的`_config.yml`文件。永久链接permalink改成：`permalink: :category:title/`的形式，或者你也可以改成自己想要的格式，注意yml语法。
### 后续
换域名的事再说吧。
### 博客链接
[Alex的博客](sqxiang.github.io)

