import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import joblib
import os
from datetime import datetime

def save_model(model, scaler, selector, pca, model_dir='models'):
    """
    保存模型和相关预处理组件
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存模型
    model_path = os.path.join(model_dir, f'knn_model_{timestamp}.pkl')
    joblib.dump(model, model_path)
    
    # 保存预处理组件
    components = {
        'scaler': scaler,
        'selector': selector,
        'pca': pca,
        'feature_names': selector.get_feature_names_out() if hasattr(selector, 'get_feature_names_out') else None
    }
    
    components_path = os.path.join(model_dir, f'preprocessing_{timestamp}.pkl')
    joblib.dump(components, components_path)
    
    return model_path, components_path

def load_model(model_path, components_path):
    """
    加载保存的模型和预处理组件
    """
    model = joblib.load(model_path)
    components = joblib.load(components_path)
    
    return model, components

def run(save_model_flag=True):
    """
    KNN模型的完整流程：
    1. 加载数据
    2. 数据预处理和标准化
    3. 5折交叉验证计算AUC
    4. 模型训练和调参
    5. 预测测试集并生成提交文件
    """

    data = pd.read_csv("../Data/tree_data_clean.csv")

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    selector = SelectKBest(f_classif, k=min(50, X.shape[1]))
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    X = pd.DataFrame(X_selected, columns=selected_features)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    X_train, X_val, y_train, y_val = train_test_split(
        X_resampled, y_resampled,
        test_size=0.2,
        random_state=42,
        stratify=y_resampled
    )

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)

    param_grid = {
        'n_neighbors': [5, 7, 9, 11, 15, 21],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
        'p': [1, 2, 3]
    }

    # 使用分层交叉验证
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 基础KNN模型
    knn = KNeighborsClassifier()

    # 网格搜索
    grid_search = GridSearchCV(
        knn,
        param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train_pca, y_train)
    best_knn = grid_search.best_estimator_
    best_cv_auc = grid_search.best_score_

    print("========== 优化后的KNN Results ==========")
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"调优后5折交叉验证AUC: {best_cv_auc:.4f}")

    val_scores = best_knn.predict_proba(X_val_pca)[:, 1]
    val_auc = roc_auc_score(y_val, test_scores)
    print(f"验证集AUC: {test_auc:.4f}")
    

def test_model(test_path, train_path, model_path, components_path):
    """
    使用保存的模型预测新数据
    """
    # 加载新数据
    test_data = pd.read_csv(test_path).copy()
    train_data = pd.read_csv(test_path)
    X_train = test_data.iloc[:,:-1]
    for col in test_data.columns[test_data.isnull().any()].tolist():
        mid = X_train[col].median()
        if test_data[col].isnull().any():
            test_data[col].fillna(mid, inplace=True)
    
    # 加载模型和预处理组件
    model, components = load_model(model_path, components_path)
    
    # 应用相同的预处理
    X_selected = components['selector'].transform(test_data)
    X_scaled = components['scaler'].transform(X_selected)
    X_pca = components['pca'].transform(X_scaled)
    
    scores = model.predict_proba(X_pca)[:, 1]

    results_df = pd.DataFrame({
        'score': scores
    })
    
    # 保存到CSV，不包含索引和表头
    results_df.to_csv("submission_scores.csv", index=False, header=False)

if __name__ == '__main__':
    #run()
    test_model('../Data/tree_data_clean_test.csv','../Data/tree_data_clean.csv', 'models/knn_model_20251122_230639.pkl', 'models/preprocessing_20251122_230639.pkl')