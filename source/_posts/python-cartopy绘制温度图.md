---
title: cartopy绘制温度变化图
date: 2021-08-21 13:58
tags:
 -python
categories: python
---
<blockquote class="blockquote-center">python-cartopy绘制温度变化图</blockquote>

<!--more-->

## 数据处理：

```python
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib import animation
import matplotlib

# process data
grid_data = xr.open_dataset('lat20-50_lon235-290_res0.1/od_oper_0.1_20170401_000000_0-240.nc')
data = grid_data['t2m']
lats = data.lat
lons = data.lon
new_lats = lats.values.repeat(len(lons))
new_lons = np.tile(lons, len(lats))
new_data = data.values - 273.15
new_data = new_data.reshape(new_data.shape[0],-1) # 125*166152
### new data:
### array([[20.007996  , 20.002136  , 19.925964  , ..., -6.5584106 ,
###        -7.2595825 , -7.1990356 ],
###       [19.990265  , 20.017609  , 20.015656  , ..., -5.740204  ,
###        -6.2968445 , -8.820282  ],
###       [19.998596  , 20.004456  , 19.977112  , ..., -6.5150757 ,
###        -6.41156   , -7.989685  ],
###       ...,
###       [20.15509   , 20.0672    , 19.944153  , ..., -0.60272217,
###        -0.5968628 , -0.5128784 ],
###       [20.065063  , 19.932251  , 19.945923  , ...,  4.586548  ,
###         4.494751  ,  5.2232666 ],
###       [20.164642  , 20.041595  , 20.016205  , ...,  1.580658  ,
###         1.721283  ,  2.1880798 ]], dtype=float32)
```

#### 数据sample:

  <img src="/img/data.png" alt="image-20210813140203049" style="zoom:100%;" />



## 画图函数：

```python
class animation_scatter(object):
    def __init__(self, data, size, lat, lon, vmin, vmax, cmap, fps):
        self.data = data
        self.size = size
        self.lat = lat
        self.lon = lon
        self.lat = lat
        self.vmin = vmin
        self.vmax = vmax
        self.cmap = cmap
        self.fps  = fps
        
        self.fig = plt.figure(figsize=(15, 6))
        prj = ccrs.PlateCarree()
        self.ax = self.fig.add_subplot(projection=prj)
        self.ax.set_extent([235, 290, 20, 50], prj)
        # ax.set_global()

        self.ax.add_feature(cfeature.COASTLINE, edgecolor='gray')
        self.ax.add_feature(cfeature.BORDERS, edgecolor='gray')

        self.scatter = plt.scatter(self.lon, self.lat, s=self.size, c=self.data[0], vmin=self.vmin, vmax=self.vmax, cmap=self.cmap)
        
    def animate(self):
        anim = animation.FuncAnimation(
        self.fig,
        self.update,
        frames=range(len(self.data)),
        interval=1000 / self.fps)
        return anim
        
    def update(self, i):
        self.scatter.remove()
        self.scatter = plt.scatter(self.lon, self.lat, s=self.size, c=self.data[i], vmin=self.vmin, vmax=self.vmax, cmap=self.cmap)
        return self.scatter
```

#### 绘图：

```python
scatter_d = animation_scatter(new_data, 4, new_lats, new_lons, -20, 40, 'RdBu_r', 10)
anim = scatter_d.animate()
from IPython.display import HTML
HTML(anim.to_jshtml())
```

![animation](/img/animation.gif)
