# ALTianChi
## 赛题：口碑商家流量预测
[赛题详见]<https://tianchi.aliyun.com/competition/information.htm?spm=5176.100067.5678.2.m2fvPv&raceId=231591>
## 程序结构
**data_process/**<br>
数据预处理模块，包括对原始数据进行分割和异常值处理<br>
**features/** <br>
特征工程模块，主要包括倒数三周的商家流量数据，商家属性特征，流量统计特征，滑动窗口，差分特征，对数变换特征及多项式交叉特征等<br>
**fig_plot/** <br>
可视化模块，包括预处理阶段对各商家不同日期的流量可视化，最终预测结果同观测样本对比可视化<br>
**models/** <br>
模型模块，单一模型和融合模型<br>

## 程序运行
ALTianChi项目文件夹下新建一data文件夹，将原始数据放在该文件夹下即可
