#encoding=utf-8
# 判断图片id号在csv中有无记录
import os
import pandas as pd

img_name_list = os.listdir('../dataset/roi/img_train')
data = pd.read_csv('GA_third_hospital.csv', header=None, encoding='GBK')

img_id_list = data.iloc[:, 0].tolist()

img_name_list = [int(name.replace(".png", "")) for name in img_name_list]
for id in img_id_list:
    if id not in img_name_list:
        print(id)