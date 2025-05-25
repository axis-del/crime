import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import re


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


# 特征强化函数，提取更多特征
def extract_extra_features(df):
    # 提取刑期相关特征
    df['sentence_length'] = df['fact'].str.extract(r'判处有期徒刑(\d+)年', expand=False).astype(float)
    df['sentence_length'] = df['sentence_length'].fillna(0)

    # 提取奖励次数特征
    df['award_count'] = df['fact'].str.count('表扬|记功|奖励')

    # 计算服刑时间
    time_pattern = r'(\d{4})年[\s\S]*?至(\d{4})年'
    time_matches = df['fact'].str.extract(time_pattern)

    df['prison_time'] = 0
    if not time_matches.empty:
        mask = ~time_matches[0].isna() & ~time_matches[1].isna()
        df.loc[mask, 'prison_time'] = (time_matches[1][mask].astype(int) - time_matches[0][mask].astype(int)).fillna(0)

    # 提取罚金特征
    df['fine'] = df['fact'].str.extract(r'罚金.*?(\d+)元', expand=False).astype(float)
    df['fine'] = df['fine'].fillna(0)

    # 提取犯罪类型特征
    crime_types = ['盗窃', '抢劫', '故意杀人', '毒品', '信用卡诈骗', '挪用资金']
    for crime in crime_types:
        df[f'crime_{crime}'] = df['fact'].str.contains(crime).astype(int)

    return df


# 1. 加载数据
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('testA.csv')

# 特征强化
train_df = extract_extra_features(train_df)
test_df = extract_extra_features(test_df)

# 2. 数据预处理
X_train_text = train_df['fact']
y_train = train_df['label'].values.astype(float)

# 3. 文本向量化
vectorizer = TfidfVectorizer(max_features=15000, stop_words='english', ngram_range=(1, 3))
X_train_text_vec = vectorizer.fit_transform(X_train_text)

# 数值特征
numeric_features = ['sentence_length', 'award_count', 'prison_time', 'fine']
numeric_features += [f'crime_{crime}' for crime in ['盗窃', '抢劫', '故意杀人', '毒品', '信用卡诈骗', '挪用资金']]
X_train_num = train_df[numeric_features].values
X_test_num = test_df[numeric_features].values

# 标准化数值特征
scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train_num)
X_test_num_scaled = scaler.transform(X_test_num)

# 合并文本特征和数值特征
X_train_combined = np.hstack([X_train_text_vec.toarray(), X_train_num_scaled])
X_test_combined = np.hstack([vectorizer.transform(test_df['fact']).toarray(), X_test_num_scaled])

# 4. 划分验证集
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_combined, y_train, test_size=0.2, random_state=42
)

# 5. 构建SVM回归模型
# 使用RBF核函数，C=10，epsilon=0.1
model = make_pipeline(
    StandardScaler(with_mean=False),  # SVM对特征缩放敏感
    SVR(kernel='rbf', C=10, epsilon=0.1, gamma='scale')
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
y_pred_test = model.predict(X_test_combined)

# 9. 保存预测结果
result_df = pd.DataFrame({
    'id': test_df['id'],
    '预测减刑月份': np.round(y_pred_test).astype(int)
})
result_df.to_csv('test_prediction_results_svm.csv', index=False)
print("SVM模型测试集预测结果已保存至 test_prediction_results_svm.csv")
