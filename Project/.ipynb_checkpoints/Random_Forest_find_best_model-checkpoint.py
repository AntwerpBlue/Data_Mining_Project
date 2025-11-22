# Random_Forest_find_best_model.py

import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, GridSearchCV

def run():
    """
    加载已保存的最佳随机森林模型，执行预测和评估。
    """

    # 重新加载数据用于评估
    data = pd.read_csv("Data/tree_data_clean.csv")
    X = data.drop(columns=["y"])
    y = data["y"]
    rf_base = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [120, 180, 240, 300],           # 树的数量
        'max_depth': [5, 10, 20],          # 树的最大深度
        'min_samples_leaf': [5, 10, 20],         # 叶节点所需的最小样本数 (对抗过拟合)
        "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
        'class_weight': ['balanced_subsample', 'balanced'] # 解决不平衡问题
    }
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=rf_base,          # 我们要优化的模型
        param_grid=param_grid,      # 要搜索的参数网格
        cv=kfold,                   # 交叉验证策略
        scoring='roc_auc',          # 我们的主要目标：最大化 AUC
        verbose=2,                  # 打印详细的搜索日志
        n_jobs=-1                   # 使用所有可用的 CPU 核心
    )
    print("\n--- 正在启动 5-Fold CV 网格搜索 ---")
    
    grid_search.fit(X, y)

    print("\n========== 随机森林 (Grid Search) 结果 ==========")
    print(f"最佳平均 AUC (来自 CV): {grid_search.best_score_:.5f}")
    print(f"\n最佳参数组合 (Best Params):")
    print(grid_search.best_params_)

    best_rf_model = grid_search.best_estimator_
    
    model_save_path = "Result/best_rf_model.pkl"
    with open(model_save_path, 'wb') as f:
        pickle.dump(best_rf_model, f)

    # 7. 显示最佳模型的特征重要性
    print("\n--- 最佳模型的特征重要性 (Top 15) ---")
    importances = best_rf_model.feature_importances_
    feature_names = X.columns
    
    feature_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    print(feature_imp_df.head(15))

run()