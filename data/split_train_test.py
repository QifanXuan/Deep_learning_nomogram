#encoding=utf-8
# 分割训练集验证集
import pandas as pd

# 读取原始 CSV 文件
data = pd.read_csv('GA_third_hospital.csv', header=None, encoding="GBK")
# 分别筛选出良性和恶性的样本
benign_samples = data[data[2] == 0]
malignant_samples = data[data[2] == 1]
# 计算原始数据中良性和恶性的样本数量
num_benign = len(benign_samples)
num_malignant = len(malignant_samples)
# 定义测试集比例
test_set_ratio = 0.2727  # 例如，选择 20% 的数据作为测试集

# 计算测试集中的样本数量
num_benign_test = int(num_benign * test_set_ratio)
num_malignant_test = int(num_malignant * test_set_ratio)
# 随机选择测试集样本
test_benign_samples = benign_samples.sample(n=num_benign_test, random_state=42)
test_malignant_samples = malignant_samples.sample(n=num_malignant_test, random_state=42)
# 创建测试集数据框
test_set = pd.concat([test_benign_samples, test_malignant_samples])
# 创建训练集数据框，排除测试集中的样本
train_set = data.drop(test_set.index)
# 保存训练集和测试集为 CSV 文件
train_set.to_csv('GA_third_hospital_train.csv', index=False, encoding='GBK')
test_set.to_csv('GA_third_hospital_test.csv', index=False, encoding="GBK")
