import cv2
import os

img_name = os.listdir("./img")
mask_name = os.listdir("./mask")

for name in img_name:
    img_path = os.path.join("./img", name)
    mask_path = os.path.join("./mask", name)
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)
    if img.shape[0] != 512 or img.shape[1] !=512:
        print(img_path)
    if mask.shape[0] != 512 or mask.shape[1] !=512:
        print(mask_path)