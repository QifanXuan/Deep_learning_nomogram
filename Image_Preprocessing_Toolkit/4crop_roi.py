import glob
import os
import cv2
import numpy as np
import pydicom
img_content_list = glob.glob("img/*")
mask_content_list = glob.glob("mask/*")

for k in range(len(img_content_list)):
    flag = np.zeros((512, 512), dtype=int)
    # ds = pydicom.dcmread(img_content_list[k])
    # img = ds.pixel_array
    img = cv2.imread(img_content_list[k], flags=cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_content_list[k], flags=cv2.IMREAD_GRAYSCALE)
    x = []
    y = []
    for i in range(512):
        for j in range(512):
            if (mask[i, j] != 0) & (flag[i, j] == 0):
                y.append(i)
                x_count = 0
                x_mid = 0
                a = 0
                while mask[i, j+a] == 1:
                    flag[i][j + a] = 0
                    x_count = x_count+1
                    a += 1
                x_mid = x_count/2
                x.append(j+x_mid)

    y_final = int(np.mean(y))
    x_final = int(np.mean(x))
    print(x_final, y_final)
    crop_img = img[y_final-64:y_final+64, x_final-64:x_final+64]
    crop_mask = mask[y_final-64:y_final+64, x_final-64:x_final+64]
    # print(crop_img.shape)
    # # 使用 numpy 数组填充 DICOM 文件数据信息
    # ds.PixelData = crop_img.tobytes()
    # ds.Rows = 64
    # ds.Columns = 64
    # # 新建 DICOM 文件并保存
    # ds.save_as(img_content_list[k].replace("img", "img_roi"))
    cv2.imwrite(img_content_list[k].replace("img", "cropped_img"), crop_img)
    cv2.imwrite(mask_content_list[k].replace("mask", "cropped_mask"), crop_mask)