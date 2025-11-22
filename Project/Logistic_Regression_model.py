# Logistic_Regression_model.py

import pandas as pd
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score

def run():
    """
    Logistic Regression 的全流程（已修改为 K-Fold 交叉验证）：
    1. 加载标准化数据
    2. 定义模型和 K-Fold
    3. 使用 cross_val_score 进行 10 折交叉验证
    4. 输出稳定的平均 Acc 和 AUC
    """

    # 1. 加载标准化后的数据
    X = pd.read_csv("Data/lrsvm_X_scaled.csv")
    y = pd.read_csv("Data/lrsvm_y.csv")

    # 转成 Series（避免二维标签）
    if "y" in y.columns:
        y = y["y"]
    else:
        y = y.iloc[:, 0]

    # 2. 定义模型
    model = LogisticRegression(
        penalty='l1',        # <--- 使用 L1 (Lasso)
        solver='liblinear',  # <--- 必须使用 liblinear 或 saga 来支持 L1
        C=0.1,               # <--- 增加正则化强度 (值越小, 强度越大)
        max_iter=1000,
        class_weight='balanced' # <--- 处理不平衡
    )
    
    # 3. 定义 5-Fold 交叉验证策略
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # 4. 执行交叉验证
    # 4.1 评估 AUC
    print("正在计算 5-Fold CV (AUC)...")
    auc_scores = cross_val_score(
        model, 
        X, y, 
        cv=kfold, 
        scoring='roc_auc' 
    )
    
    # 4.2 评估 Accuracy
    print("正在计算 5-Fold CV (Accuracy)...")
    acc_scores = cross_val_score(
        model, 
        X, y, 
        cv=kfold, 
        scoring='accuracy' 
    )

    # 5. 评价指标
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)
    mean_acc = np.mean(acc_scores)
    std_acc = np.std(acc_scores)

    print("\n========== Logistic Regression Results (5-Fold Cross-Validation) ==========")
    # print(f"所有 5 折的 AUC 分数: \n{np.round(auc_scores, 4)}")
    # print("-------------------------------------------------")
    print(f"平均 Test Accuracy : {mean_acc:.4f} (标准差: +/- {std_acc:.4f})")
    print(f"平均 Test AUC      : {mean_auc:.4f} (标准差: +/- {std_auc:.4f})")