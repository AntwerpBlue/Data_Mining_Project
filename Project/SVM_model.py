# SVM_model.py

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate

def run():
    """
    RBF SVM 全流程（已修改为 5-Fold 交叉验证）：
    1. 加载标准化数据
    2. 定义 SVM 模型
    3. 使用 StratifiedKFold 进行 5 折交叉验证
    4. 同时计算 Accuracy 和 AUC
    5. 输出平均指标
    """

    # 1. 加载标准化后的数据
    X = pd.read_csv("Data/lrsvm_X_scaled.csv")
    y = pd.read_csv("Data/lrsvm_y.csv")

    # 将 y 转成 Series
    if "y" in y.columns:
        y = y["y"]
    else:
        y = y.iloc[:, 0]

    # 2. 定义 RBF 核 SVM 模型
    model = SVC(
        kernel="rbf",       
        C=0.1,              
        gamma="scale",      
        probability=True,    # 必须为 True 才能计算 AUC
        class_weight='balanced', 
        random_state=42
    )

    # 3. 定义交叉验证策略
    # 使用 StratifiedKFold 确保每一折的类别比例一致
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 4. 执行交叉验证
    scoring_metrics = {'accuracy': 'accuracy', 'auc': 'roc_auc'}
    
    print("正在进行 5-Fold 交叉验证 (SVM 训练较慢，请耐心等待)...")
    
    # n_jobs=-1 使用所有 CPU 核心加速
    results = cross_validate(
        model, X, y, 
        cv=cv, 
        scoring=scoring_metrics,
        n_jobs=-1, 
        return_train_score=False
    )

    # 5. 提取结果并计算平均值
    mean_acc = np.mean(results['test_accuracy'])
    std_acc = np.std(results['test_accuracy'])
    mean_auc = np.mean(results['test_auc'])
    std_auc = np.std(results['test_auc'])

    print("========== SVM (RBF Kernel) 5-Fold CV Results ==========")
    # print(f"所有 5 折的 Accuracy: {np.round(results['test_accuracy'], 4)}")
    # print(f"所有 5 折的 AUC     : {np.round(results['test_auc'], 4)}")
    # print("--------------------------------------------------------")
    print(f"平均 Test Accuracy : {mean_acc:.4f} (标准差: +/- {std_acc:.4f})")
    print(f"平均 Test AUC      : {mean_auc:.4f} (标准差: +/- {std_auc:.4f})")