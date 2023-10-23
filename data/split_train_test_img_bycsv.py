import shutil

import pandas as pd
import os

root_path = '../dataset/roi/img_train'
train_img_name = os.listdir('../dataset/roi/img_train')
test_csv = pd.read_csv('GA_third_hospital_test.csv', header=None, encoding='GBK')

test_id = test_csv.iloc[:, 0].tolist()

for name in train_img_name:
    id = int(name.replace(".png", ""))
    img_path = os.path.join(root_path, name)
    mask_path = img_path.replace("img", "mask")
    direct_path = img_path.replace("train", "test0")
    mask_direct_path = mask_path.replace("train", "test0")
    if id in test_id:
        shutil.copy(img_path, direct_path)
        shutil.copy(mask_path, mask_direct_path)
        os.remove(img_path)
        os.remove(mask_path)