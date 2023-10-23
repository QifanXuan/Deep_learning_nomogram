#encoding=UTF-8
import os
import cv2
mask_p = "./mask"
mask_name = os.listdir(mask_p)
for name in mask_name:
    mask_path = os.path.join(mask_p, name)
    # 读取输入图片
    mask = cv2.imread(mask_path)
    # 将小于等于0的像素值改为255，将大于0的像素值改为0
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    # 把修改后的图像数据写回原文件
    cv2.imwrite(mask_path, mask)
