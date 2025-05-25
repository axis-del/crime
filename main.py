import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# 定义Score1计算函数
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
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('testA.csv')
#print(test_df)
# 2. 数据预处理
X_train = train_df['fact']
y_train = train_df['label'].values.astype(float)

# 3. 文本向量化
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(test_df['fact'])

# 4. 划分验证集
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_vec, y_train, test_size=0.2, random_state=42
)

# 5. 构建SVM回归模型
# 使用RBF核函数，C=10，epsilon=0.1
model = make_pipeline(
    StandardScaler(with_mean=False),  # SVM对特征缩放敏感
    SVR(kernel='rbf', C=10, epsilon=0.01, gamma='scale')
)

# 6. 训练模型
print("开始训练SVM模型...")
model.fit(X_train_split, y_train_split)

# 7. 评估模型
y_pred_val = model.predict(X_val_split)
score1 = calculate_score1(y_pred_val, y_val_split)
avg_score1 = np.mean(score1)
ext_acc = np.sum(np.round(y_pred_val) == y_val_split) / len(y_val_split)
final_score = avg_score1 * 0.7 + ext_acc * 0.3

print("\n验证集评估指标：")
print(f"Score1均值: {avg_score1:.4f}")
print(f"ExtAcc: {ext_acc:.4f}")
print(f"FinalScore: {final_score:.4f}")

# 8. 对测试集进行预测
y_pred_test = model.predict(X_test_vec)

# 9. 保存预测结果
result_df = pd.DataFrame({
    'id': test_df['id'],
    '预测减刑月份': np.round(y_pred_test).astype(int)
})
result_df.to_csv('test_prediction_results_svm.csv', index=False)
print("SVM模型测试集预测结果已保存至 test_prediction_results_svm.csv")