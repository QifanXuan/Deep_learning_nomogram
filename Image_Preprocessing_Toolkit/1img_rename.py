import os
img_root_path = "./fat_mask"
img_name_list = os.listdir(img_root_path)

for name in img_name_list:
    img_path = os.path.join(img_root_path, name)
    new_path = os.path.join(img_root_path, name[0:6]+".png")
    os.rename(img_path, new_path)