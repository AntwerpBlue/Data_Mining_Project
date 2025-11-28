import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFECV, SelectFromModel
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.isotonic import IsotonicRegression
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours
import joblib
import os
from datetime import datetime


def save_model(model, scaler, selector, pca, model_dir='../Result'):
    """
    保存模型和相关预处理组件
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # 保存模型
    model_path = os.path.join(model_dir, f'knn_model.pkl')
    joblib.dump(model, model_path)
    
    # 保存预处理组件
    components = {
        'scaler': scaler,
        'selector': selector,
        'pca': pca,
    }
    
    components_path = os.path.join(model_dir, f'preprocessing.pkl')
    joblib.dump(components, components_path)
    
    return model_path, components_path

def load_model(model_path, components_path):
    """
    加载保存的模型和预处理组件
    """
    model = joblib.load(model_path)
    components = joblib.load(components_path)
    
    return model, components

def exponential_weights(distances):
    """指数衰减权重函数"""
    epsilon = 1e-10
    alpha = 2.0
    weights = np.exp(-alpha * distances / (np.mean(distances) + epsilon))
    if np.sum(weights) < epsilon:
        weights = np.ones_like(weights)
    return weights / (np.sum(weights) + epsilon)

def gaussian_weights(distances):
    """高斯权重函数"""
    sigma = np.std(distances)
    if sigma < 1e-10:  # 防止除零
        sigma = 1.0
    weights = np.exp(-(distances**2) / (2 * sigma**2))
    # 确保至少有一个非零权重
    if np.sum(weights) < 1e-10:
        weights = np.ones_like(weights)
    return weights / (np.sum(weights) + 1e-10)  # 归一化

def inverse_square_weights(distances):
    """反平方权重函数"""
    epsilon = 1e-10
    distances_safe = np.maximum(distances, epsilon)  # 避免除零
    weights = 1.0 / (distances_safe**2 + epsilon)
    # 确保至少有一个非零权重
    if np.sum(weights) < epsilon:
        weights = np.ones_like(weights)
    return weights / (np.sum(weights) + epsilon)  # 归一化

def adaptive_weights(distances):
    """自适应权重函数，结合距离和排名信息"""
    epsilon = 1e-10
    # 距离权重（安全处理）
    distances_safe = np.maximum(distances, epsilon)
    dist_weights = 1.0 / (distances_safe + epsilon)
    
    # 排名权重
    rank_weights = 1.0 / (np.argsort(np.argsort(distances)) + 1)
    
    # 组合权重
    weights = dist_weights * rank_weights
    
    # 确保至少有一个非零权重
    if np.sum(weights) < epsilon:
        weights = np.ones_like(weights)
        
    return weights / (np.sum(weights) + epsilon)  # 归一化

def probability_calibration(y_true, y_pred):
    """概率校准"""
    print("进行概率校准...")
    
    # 使用保序回归校准概率
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(y_pred, y_true)
    calibrated_probs = calibrator.transform(y_pred)
    
    return calibrated_probs

def create_knn_ensemble():
    """创建KNN集成模型"""
    
    knn1 = KNeighborsClassifier(
        n_neighbors=15, 
        weights='distance', 
        metric='euclidean',
        n_jobs=-1
    )
    
    knn2 = KNeighborsClassifier(
        n_neighbors=21, 
        weights=gaussian_weights, 
        metric='manhattan',
        p=2,
        n_jobs=-1,
        algorithm='brute'
    )
    
    knn3 = KNeighborsClassifier(
        n_neighbors=25, 
        weights=gaussian_weights, 
        metric='minkowski', 
        p=2,
        n_jobs=-1,
        algorithm='kd_tree'
    )
    
    ensemble = VotingClassifier(
        estimators=[
            ('knn1', knn1),
            ('knn2', knn2), 
            ('knn3', knn3)
        ],
        voting='soft',
        weights=[1, 1, 1],
        n_jobs=-1
    )
    
    return ensemble

def evaluate_model(model, X_train, y_train, X_val, y_val, calibrate=False):
    """评估模型并返回AUC分数"""
    
    # 训练集预测
    train_scores = model.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, train_scores)
    
    # 验证集预测
    val_scores = model.predict_proba(X_val)[:, 1]
    
    # 概率校准
    if calibrate:
        val_scores = probability_calibration(y_val, val_scores)
    
    val_auc = roc_auc_score(y_val, val_scores)
    
    print(f"训练集AUC: {train_auc:.4f}")
    print(f"验证集AUC: {val_auc:.4f}")
    
    # 预测值分布分析
    print("\n预测值分布分析:")
    print(f"训练集预测值范围: {np.min(train_scores):.4f} - {np.max(train_scores):.4f}")
    print(f"验证集预测值范围: {np.min(val_scores):.4f} - {np.max(val_scores):.4f}")
    print(f"验证集预测值>0.9的比例: {np.mean(val_scores > 0.9):.4f}")
    print(f"验证集预测值<0.1的比例: {np.mean(val_scores < 0.1):.4f}")
    
    return train_auc, val_auc, val_scores

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

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    smote_tomek = SMOTETomek(
        sampling_strategy=0.4,
        random_state=114514,
        smote=SMOTE(sampling_strategy=0.4, k_neighbors=min(5, sum(y_train)-1))
    )
    smote_enn = SMOTEENN(
        sampling_strategy=0.35,
        random_state=114514,
        smote=SMOTE(sampling_strategy=0.4, k_neighbors=min(5, sum(y_train)-1))
    )
    smote = SMOTE(sampling_strategy=0.35, random_state=114514)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print("\n采样后的数据分布:")
    print("原始训练集 - 正样本比例:", np.mean(y_train))
    print("SMOTE后 - 正样本比例:", np.mean(y_train_resampled))
    
    rf = RandomForestClassifier(n_estimators=100, random_state=114514, n_jobs=-1)
    rf.fit(X_train_resampled, y_train_resampled)
    
    selector_rf = SelectFromModel(rf, prefit=True, threshold='median')
    selector_mi = SelectKBest(mutual_info_classif, k=min(25, X_train.shape[1]))
    selector_anova = SelectKBest(f_classif, k=min(30, X.shape[1]))

    selector = selector_mi

    if selector == selector_rf:
        X_train_selected = selector_rf.transform(X_train_resampled)
        X_val_selected = selector_rf.transform(X_val)
    else:
        X_train_selected = selector_mi.fit_transform(X_train_resampled, y_train_resampled)
        X_val_selected = selector_mi.transform(X_val)
    
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_val_scaled = scaler.transform(X_val_selected)

    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)

    param_grid = {
        'n_neighbors': [13, 15, 17, 21, 25, 30],
        'weights': ['distance', gaussian_weights, exponential_weights, inverse_square_weights, adaptive_weights],
        'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
        'p': [1, 2, 3],
        'algorithm': ['auto']
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

    grid_search.fit(X_train_pca, y_train_resampled)
    best_knn = grid_search.best_estimator_
    train_scores = best_knn.predict_proba(X_train_pca)[:, 1]
    val_scores = best_knn.predict_proba(X_val_pca)[:, 1]
    
    train_auc = roc_auc_score(y_train_resampled, train_scores)
    val_auc = roc_auc_score(y_val, val_scores)
    
    print(f"训练集AUC: {train_auc:.4f}")
    print(f"验证集AUC: {val_auc:.4f}")
    print(f"最佳参数: {grid_search.best_params_}")
    
    # 检查预测值分布
    print("\n预测值分布:")
    print("训练集预测值范围:", np.min(train_scores), "-", np.max(train_scores))
    print("验证集预测值范围:", np.min(val_scores), "-", np.max(val_scores))
    print("验证集中预测值>0.9的比例:", np.mean(val_scores > 0.9))

    print(f"\n{'='*50}")
    print("尝试集成模型...")
    
    ensemble_model = create_knn_ensemble()
    ensemble_model.fit(X_train_pca, y_train_resampled)
    
    ensemble_train_auc, ensemble_val_auc, ensemble_scores = evaluate_model(
        ensemble_model,
        X_train_pca, y_train_resampled,
        X_val_pca, y_val
    )
    
    # 选择最佳模型（单个或集成）
    if ensemble_val_auc > val_auc:
        best_knn = ensemble_model
        val_auc = ensemble_val_auc
        print("集成模型表现更好!")

    final_val_scores = best_knn.predict_proba(X_val_pca)[:, 1]
    calibrated_scores = probability_calibration(y_val, final_val_scores)
    calibrated_auc = roc_auc_score(y_val, calibrated_scores)
    
    print(f"校准前AUC: {val_auc:.4f}")
    print(f"校准后AUC: {calibrated_auc:.4f}")
    
    model_path, components_path = save_model(best_knn, scaler, selector, pca)
    print(f"\n模型已保存至: {model_path}")
    print(f"预处理组件已保存至: {components_path}")

    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(final_val_scores, y_val)
    calibrator_path = os.path.join('../Result', f'calibrator.pkl')
    joblib.dump(calibrator, calibrator_path)
    print(f"概率校准器已保存至: {calibrator_path}")
    
    return model_path, components_path

def test_model(test_path, train_path, model_path, components_path, calibrator_path):
    """
    使用保存的模型预测新数据
    """
    # 加载新数据
    test_data = pd.read_csv(test_path).copy()
    train_data = pd.read_csv(train_path)
    X_train = train_data.iloc[:,:-1]
    for col in test_data.columns[test_data.isnull().any()].tolist():
        mid = X_train[col].median()
        if test_data[col].isnull().any():
            test_data[col].fillna(mid, inplace=True)
    
    # 加载模型和预处理组件
    model, components = load_model(model_path, components_path)

    # 加载校准器（如果存在）
    calibrator = None
    if calibrator_path and os.path.exists(calibrator_path):
        calibrator = joblib.load(calibrator_path)
    
    # 应用相同的预处理
    X_selected = components['selector'].transform(test_data)
    X_scaled = components['scaler'].transform(X_selected)
    X_pca = components['pca'].transform(X_scaled)
    
    scores = model.predict_proba(X_pca)[:, 1]
    
    # 概率校准
    if calibrator is not None:
        scores = calibrator.transform(scores)
        print("已应用概率校准")

    # 保存结果
    results_df = pd.DataFrame({'score': scores})
    results_df.to_csv("submission_scores.csv", index=False, header=False)
    
    print(f"预测完成! 结果已保存至 submission_scores.csv")
    print(f"预测值统计: 最小值={scores.min():.4f}, 最大值={scores.max():.4f}, 均值={scores.mean():.4f}")

if __name__ == '__main__':
    run()
    test_model('../Data/tree_data_clean_test.csv','../Data/tree_data_clean.csv', '../Result/knn_model.pkl', '../Result/preprocessing.pkl', '../Result/calibrator.pkl')