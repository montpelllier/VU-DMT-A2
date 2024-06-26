# DMT Assignment 2
This repository is for the assignment 2 of Data Mining Technique at Vrije Universiteit (VU) Amsterdam.

## Competition
The competition for this assignment is held on Kaggle platform. Click [here](https://www.kaggle.com/competitions/dmt-2024-2nd-assignment) for detail.
Besides, the original competition can be found on Kaggle as well by clicking [this](https://www.kaggle.com/c/expedia-personalized-sort).
### Dataset 
The dataset is available on [here](https://www.kaggle.com/competitions/dmt-2024-2nd-assignment/data).
### Metric
The evaluation metric for this competition is Normalized Discounted Cumulative Gain.
See https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG for more details.

## Exploratory Data Analysis (EDA)
[//]: # (In EDA, )
### 1. Overview of dataset
View the overall situation of the data set by functions like `head()`, `tail()`, `describe()` and `info()`.
Those information consists of mean value, data type, data size, non-null count and so on.

### 2. Missing data and anomalies
Features are classified into date, category, numerical, and text features. 
Check the missing rate, number of categories and outliers of each dimension feature.

### 3. Normalize
group by prop_id
#### 
众数 平均数来填充
### 4. Correlation analysis and feature selection
![image](https://github.com/montpelllier/VU-DMT-A2/assets/145342600/014b2e93-aa79-45e3-b8ae-c5f46e3a1693)

## Learn to rank
浅层模型的代表有LR(逻辑回归)、FM
### RankNet
特征工程不足: 如果特征工程不足，即未能提取出对排序任务有意义的特征，那么简单的线性模型可能更容易理解和适应数据
