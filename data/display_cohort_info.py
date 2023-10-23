#encoding=UTF-8
# 输出三个队列良恶性比例信息
from collections import Counter
import pandas as pd

first_cohort = pd.read_csv('./GA_first_hospital.csv', header=None, encoding='GBK')
first_cohort_id = first_cohort.iloc[:, 0].tolist()
first_cohort_label = first_cohort.iloc[:, 2].tolist()
first_cohort_B_M = dict(Counter(first_cohort_label))

second_cohort = pd.read_csv("./GA_second_hospital.csv", header=None, encoding='GBK')
second_cohort_id = second_cohort.iloc[:, 0].tolist()
second_cohort_label = second_cohort.iloc[:, 2].tolist()
second_cohort_B_M = dict(Counter(second_cohort_label))

# 训练集
third_cohort = pd.read_csv("./GA_third_hospital_train.csv", header=None, encoding='GBK')
third_cohort_id = third_cohort.iloc[:, 0].tolist()
third_cohort_label = third_cohort.iloc[:, 2].tolist()
third_cohort_B_M = dict(Counter(third_cohort_label))

# 测试集
third_cohort1 = pd.read_csv("./GA_third_hospital_test.csv", header=None, encoding='GBK')
third_cohort_id1 = third_cohort1.iloc[:, 0].tolist()
third_cohort_label1 = third_cohort1.iloc[:, 2].tolist()
third_cohort_B_M1 = dict(Counter(third_cohort_label1))

print(f"一院良恶性比例为{first_cohort_B_M}")
print(f"二院良恶性比例为{second_cohort_B_M}")
print(f"三院内部测试集良恶性比例为{third_cohort_B_M1}")
print(f"三院训练集良恶性比例为{third_cohort_B_M}")