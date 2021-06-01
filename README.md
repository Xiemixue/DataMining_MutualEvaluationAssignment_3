# 互评作业3: 分类、预测与聚类

# 作业要求

## 1. 问题描述：

  本次作业中，将从下面的3个问题中任选一个进行。

## 2.可选问题

### 2.1 Hotel booking demand, 酒店预订需求

数据集：Hotel booking demand [(download)](https://www.kaggle.com/jessemostipak/hotel-booking-demand)

该数据集包含城市酒店和度假酒店的预订信息，包括预订时间、停留时间，成人/儿童/婴儿人数以及可用停车位数量等信息。

数据量：32列共12W数据。

基于这个数据集，可进行以下问题的探索：

 ```
    基本情况：城市酒店和假日酒店预订需求和入住率比较；
    用户行为：提前预订时间、入住时长、预订间隔、餐食预订情况；
    一年中最佳预订酒店时间；
    利用Logistic预测酒店预订。
    也可以自行发现其他问题，并进行相应的挖掘。
 ```

### 2.2 Video Game Sales 电子游戏销售分析

数据集：Video Game Sales [(download)](https://www.kaggle.com/gregorut/videogamesales)

该数据集包含游戏名称、类型、发行时间、发布者以及在全球各地的销售额数据。

数据量：11列共1.66W数据。

基于这个数据集，可进行以下问题的探索：

```
    电子游戏市场分析：受欢迎的游戏、类型、发布平台、发行人等；
    预测每年电子游戏销售额。
    可视化应用：如何完整清晰地展示这个销售故事。
    也可以自行发现其他问题，并进行相应的挖掘。
```

### 2.3 US Accidents 美国交通事故分析（2016-2019）

数据集：US Accidents [(download)](https://www.kaggle.com/sobhanmoosavi/us-accidents)

该数据集覆盖全美49州的全国性交通事故数据集，时间跨度：2016.02-2019。12，包括事故严重程度、事故开始和结束时间，事故地点、天气、温度、湿度等数据。

数据量：49列共300W数据。

基于这个数据集，可进行以下问题的探索：

```
    发生事故最多的州，什么时候容易发生事故；
    影响事故严重程度的因素；
    预测事故发生的地点；
    可视化应用：讲述4年间美国发生事故的总体情况。
    也可以自行发现其他问题，并进行相应的挖掘。
```

## 3. 提交内容

```
    对数据集进行处理的代码
    数据挖掘代码
    挖掘过程的报告：展示挖掘的过程、结果和你的分析
    所选择的问题在README中说明，数据文件不要上传到Github中
    乐学平台提交注意事项：

    仓库地址：记得加上
    报告：附件，word，pdf，html格式都可以
```

# 选择的数据集介绍

## Trending YouTube Video Statistics [(download)](https://www.kaggle.com/datasnaek/youtube-new)

* Description: 

    该数据集包含有关YouTube每日热门视频的数月（且在不断增加）的数据。包含美国，GB，DE，CA和FR地区（分别为美国，英国，德国，加拿大和法国）的数据，每天最多列出200个趋势视频。
    
    YouTube会在平台上不断更新热门视频的列表。 为了确定本年度最热门的视频，YouTube使用了多种因素，包括衡量用户的互动情况(观看次数，分享次数，评论和喜欢的次数)。
    
    数据集包括video title, channel title, publish time, tags, views, likes and dislikes, description, and comment count属性。

* Contents:
    ```
    US_category_id.json
    USvideos.csv
    ```

# 使用说明
下载数据集并更新代码里的数据集路径，然后使用jupyter notebook 打开MutualEvaluationHomework_2.ipynb文件，运行指定cell，即可得到相应可视化结果。或者用pycharm运行MutualEvaluationHomework_2.py文件。
