#encoding=utf-8
import pandas as pd
import os

img_name_list = os.listdir('./img')

img_id_list = [int(name.replace(".png", "")) for name in img_name_list]

data = pd.read_csv("test.csv", encoding="UTF-8")
for first_id in img_id_list[0:19]:
    first_name = data.loc[data['病历号']==first_id]['姓名']
    first_name = first_name.tolist()
    first_name = first_name[0]
    print(first_id, first_name)
print("-----------------")
for second_id in img_id_list[19:30]:
    second_name = data.loc[data['病历号']==second_id]['姓名']
    second_name = second_name.tolist()
    second_name = second_name[0]
    print(second_id, second_name)
print("-----------------")
for third_id in img_id_list[30:]:
    third_name = data.loc[data['病历号']==third_id]['姓名']
    third_name = third_name.tolist()
    third_name = third_name[0]
    print(third_id, third_name)