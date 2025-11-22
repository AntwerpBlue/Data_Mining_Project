# Decision_Tree_model.py

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score

def run():
    """
    决策树模型的完整流程：
    1. 加载数据
    2. train/test 划分
    3. 模型训练
    4. 模型预测
    5. 输出 Train Acc、Test Acc、AUC
    """

    # 1. 加载数据
    data = pd.read_csv("Data/tree_data_clean.csv")
    X = data.drop(columns=["y"])
    y = data["y"]

    # 2. 加载模型
    model = DecisionTreeClassifier(
        max_depth=6,
        criterion="gini",
        random_state=42
    )

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # 3. 执行交叉验证
    # 3.1 评估 AUC
    print("正在计算 5-Fold CV (AUC)...")
    auc_scores = cross_val_score(
        model, 
        X, y, 
        cv=kfold, 
        scoring='roc_auc' 
    )
    
    # 3.2 评估 Accuracy
    print("正在计算 5-Fold CV (Accuracy)...")
    acc_scores = cross_val_score(
        model, 
        X, y, 
        cv=kfold, 
        scoring='accuracy' 
    )

    # 4. 评价指标
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)
    mean_acc = np.mean(acc_scores)
    std_acc = np.std(acc_scores)

    print("========== Logistic Regression Results (5-Fold Cross-Validation) ==========")
    # print(f"所有 5 折的 AUC 分数: \n{np.round(auc_scores, 4)}")
    # print("-------------------------------------------------")
    print(f"平均 Test Accuracy : {mean_acc:.4f} (标准差: +/- {std_acc:.4f})")
    print(f"平均 Test AUC      : {mean_auc:.4f} (标准差: +/- {std_auc:.4f})")