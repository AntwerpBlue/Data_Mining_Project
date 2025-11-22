import pandas as pd
import joblib

def generate_submission():
    """
    加载已保存的最佳随机森林模型，执行预测并生成提交所需的评分文件。
    """

    # 加载已保存的最佳随机森林模型
    print("加载已保存的最佳随机森林模型……")
    model = joblib.load("Result/best_rf_model.pkl")  # 确保路径正确

    # 重新加载测试数据
    test_data = pd.read_csv("Data/tree_data_clean_test.csv")  # 使用正确的路径
    X_test = test_data
    
    # 预测概率
    test_proba = model.predict_proba(X_test)[:, 1]  # 获取正类（响应）概率

    # 生成提交文件的DataFrame
    submission = pd.DataFrame({
        'score': test_proba
    })

    # 保存提交文件为CSV
    submission.to_csv("Result/submission_scores.csv", index=False, header=False)
    print("提交评分文件已生成并保存为 'submission_scores.csv'")

# 执行函数
generate_submission()
