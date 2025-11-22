使用Decision Tree/Logistic Regression/SVM/Random Forest四种模型尝试了一下
其中Random Forest模型是最终选用生成submission score的模型

运行main.py文件即可查看四种模型的效果

对原bank_marketing_train.csv数据做了数据清洗，生成三份文件。
tree_data_clean.csv用于决策树和随机森林模型
lrsvm_X_scaled.csv和lrsvm_y.csv用于SVM和logistic regression模型，区别主要在于是否标准化


---------------------
相较于上一版本的优化如下：
---------------------
1.最佳AUC从原来的0.8034，现在是0.8069
2.Logistic Regression和SVM的编码方式从one-hot变成了Target Encoding, 避免了维度灾难问题（279->25）
3.补上了第一版没有的5折交叉验证部分以及ROC图
4.优化了项目结构，FineTuning用于数据预处理，Project用于存放模型代码，Result存放结果，Data存放数据
接下来思路如下:
1. 可以考虑特征工程，比如根据某几项特征生成新的特征
2. 尝试运行集成学习、KNN、K-means等方法


-----------------
Update 11.22
使用KNN，优化AUC到90.17%
在测试函数中填充了测试集缺失值