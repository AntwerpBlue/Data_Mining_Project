# Random_Forest_model.py

import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, roc_curve 
import matplotlib.pyplot as plt 

def run():
    """
    修改后的流程：
    1. 加载模型和数据
    2. 手动执行 5-Fold CV, 并保存每次的预测
    3. 找到 AUC 最高的一折
    4. 绘制该折的 ROC 曲线
    """

    print("加载已保存的最佳随机森林模型……")
    model = joblib.load("Result/best_rf_model.pkl")
    data = pd.read_csv("Data/tree_data_clean.csv")
    
    X = data.drop(columns=["y"])
    y = data["y"]
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    print("正在执行 5-Fold CV (AUC)...")
    
    auc_scores = []
    fold_results = [] # 用于存储每一折的 (y_test, y_proba)
    
    for fold_num, (train_index, test_index) in enumerate(kfold.split(X, y)):
        
        print(f"--- 正在处理第 {fold_num + 1}/5 折 ---")
        
        # a. 划分数据
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # b. 克隆你的基础模型
        model_fold = clone(model)
        
        # c. 在 4/5 的数据上重新训练
        model_fold.fit(X_train, y_train)
        
        # d. 在 1/5 的数据上预测概率
        y_proba = model_fold.predict_proba(X_test)[:, 1]
        
        # e. 计算并保存 AUC
        auc = roc_auc_score(y_test, y_proba)
        auc_scores.append(auc)
        
        # f. 保存该折的预测结果
        fold_results.append({
            'auc': auc,
            'y_test': y_test,
            'y_proba': y_proba,
            'fold_num': fold_num + 1
        })
    
    # 3. 找到最佳折
    best_fold_index = np.argmax(auc_scores)
    best_result = fold_results[best_fold_index]
    best_auc = best_result['auc']
    best_fold_num = best_result['fold_num']

    # 4. 评价指标
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)

    print("========== Random Forest Results (5-Fold Cross-Validation) ==========")
    print(f"平均 Test AUC      : {mean_auc:.4f} (标准差: +/- {std_auc:.4f})")

    # 5. 绘制最佳折的 ROC 曲线
    print(f"\n正在绘制第 {best_fold_num} 折 (AUC={best_auc:.4f}) 的 ROC 曲线...")

    y_test_best = best_result['y_test']
    y_proba_best = best_result['y_proba']
    
    # a. 计算 ROC 曲线的点 (FPR, TPR)
    fpr, tpr, threholds= roc_curve(y_test_best, y_proba_best)
    
    # b. 绘图
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='blue', lw=2, 
             label=f'ROC curve (AUC = {best_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', 
             label='AUC = 0.50')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR (False Positive Rate)', fontsize=12)
    plt.ylabel('TPR (True Positive Rate)', fontsize=12)
    plt.title(f'Random Forest ROC', fontsize=16)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig("Result/ROC.png")
    plt.show()