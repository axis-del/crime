import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# 定义Score1计算函数（适用于有真实标签的情况）
def calculate_score1(lp, la):
    v = np.abs(np.log(lp + 1) - np.log(la + 1))
    score = np.zeros_like(v)
    score[v <= 0.2] = 1
    score[(0.2 < v) & (v <= 0.4)] = 0.8
    score[(0.4 < v) & (v <= 0.6)] = 0.6
    score[(0.6 < v) & (v <= 0.8)] = 0.4
    score[(0.8 < v) & (v <= 1.0)] = 0.2
    score[v > 1.0] = 0
    return score

# 1. 加载数据
train_df = pd.read_csv('train.csv')  # 训练集（含id, fact, label）
test_df = pd.read_csv('testA.csv')    # 测试集（含id, fact）

# 2. 数据预处理
# 提取文本特征和标签
X_train = train_df['fact']
y_train = train_df['label'].values.astype(float)  # 转换为浮点型

# 3. 文本向量化（TF-IDF）
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)

# 4. 训练随机森林回归模型（比线性回归更适合文本数据）
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train_vec, y_train)

# 5. 对测试集进行预测
X_test = vectorizer.transform(test_df['fact'])
y_pred_test = model.predict(X_test)

# 6. 保存预测结果到CSV
result_df = pd.DataFrame({
    'id': test_df['id'],
    '预测减刑月份': y_pred_test.round().astype(int)  # 四舍五入取整
})
result_df.to_csv('test_prediction_results.csv', index=False)
print("测试集预测结果已保存至 test_prediction_results.csv")

# （可选）训练集性能评估（仅用于调试，测试集无真实标签）
if len(train_df) > 0:
    y_pred_train = model.predict(X_train_vec)
    score1 = calculate_score1(y_pred_train, y_train)
    avg_score1 = np.mean(score1)
    ext_acc = np.sum(np.round(y_pred_train) == y_train) / len(y_train)
    final_score = avg_score1 * 0.7 + ext_acc * 0.3
    print("\n训练集评估指标：")
    print(f"Score1均值: {avg_score1:.4f}")
    print(f"ExtAcc: {ext_acc:.4f}")
    print(f"FinalScore: {final_score:.4f}")