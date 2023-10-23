import glob
import os

import cv2
import numpy as np

img_test1 = glob.glob("../dataset/roi/img_test1/*.png")
mask_test1 = glob.glob('../dataset/roi/mask_test1/*.png')

img_test2 = glob.glob("../dataset/roi/img_test2/*.png")
mask_test2 = glob.glob('../dataset/roi/mask_test2/*.png')

img_train = glob.glob("../dataset/roi/img_train/*.png")
mask_train = glob.glob('../dataset/roi/mask_train/*.png')

# train
for i in range(len(img_train)):
    img_path = img_train[i]
    mask_path = mask_train[i]
    # test是27，train是28，valid是28
    img_name = img_path[-10:]
    img = cv2.imread(img_path, flags=cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, flags=cv2.IMREAD_GRAYSCALE)
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    kernel = np.ones(shape=[3, 3], dtype=np.uint8)
    expansion = cv2.dilate(mask, kernel, iterations=3)

    mask = (mask/255).astype(np.uint8)
    expansion = (expansion/255).astype(np.uint8)

    peritumor = img*(expansion-mask)
    intratumor = img*mask
    mergeRegion = peritumor+intratumor

    # cv2.imshow("1", peritumor)
    # cv2.imshow("2", intratumor)
    # cv2.imshow("3", mergeRegion)
    # cv2.waitKey(-1)

    peritumor_path = os.path.join("../dataset/peritumor/img_train", img_name)
    intratumor_path = os.path.join("../dataset/intratumor/img_train", img_name)
    mergeRegion_path = os.path.join("../dataset/merge_region/img_train", img_name)

    cv2.imwrite(peritumor_path, peritumor)
    cv2.imwrite(intratumor_path, intratumor)
    cv2.imwrite(mergeRegion_path, mergeRegion)

# test2
for i in range(len(img_test2)):
    img_path = img_test2[i]
    mask_path = mask_test2[i]
    # test是27，train是28，valid是28
    img_name = img_path[25:]
    img = cv2.imread(img_path, flags=cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, flags=cv2.IMREAD_GRAYSCALE)
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    kernel = np.ones(shape=[3, 3], dtype=np.uint8)
    expansion = cv2.dilate(mask, kernel, iterations=3)

    mask = (mask/255).astype(np.uint8)
    expansion = (expansion/255).astype(np.uint8)

    peritumor = img*(expansion-mask)
    intratumor = img*mask
    mergeRegion = peritumor+intratumor

    # cv2.imshow("1", peritumor)
    # cv2.imshow("2", intratumor)
    # cv2.imshow("3", mergeRegion)
    # cv2.waitKey(-1)

    peritumor_path = os.path.join("../dataset/peritumor/img_test2", img_name)
    intratumor_path = os.path.join("../dataset/intratumor/img_test2", img_name)
    mergeRegion_path = os.path.join("../dataset/merge_region/img_test2", img_name)

    cv2.imwrite(peritumor_path, peritumor)
    cv2.imwrite(intratumor_path, intratumor)
    cv2.imwrite(mergeRegion_path, mergeRegion)

# test1
for i in range(len(img_test1)):
    img_path = img_test1[i]
    mask_path = mask_test1[i]
    # test是27，train是28，valid是28
    img_name = img_path[25:]
    img = cv2.imread(img_path, flags=cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, flags=cv2.IMREAD_GRAYSCALE)
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    mask = mask.astype(np.uint8)
    kernel = np.ones(shape=[3, 3], dtype=np.uint8)
    expansion = cv2.dilate(mask, kernel, iterations=3)
    mask = (mask/255).astype(np.uint8)
    expansion = (expansion/255).astype(np.uint8)

    peritumor = img*(expansion-mask)
    intratumor = img*mask
    mergeRegion = peritumor+intratumor

    # cv2.imshow("1", peritumor)
    # cv2.imshow("2", intratumor)
    # cv2.imshow("3", mergeRegion)
    # cv2.waitKey(-1)
    peritumor_path = os.path.join("../dataset/peritumor/img_test1", img_name)
    intratumor_path = os.path.join("../dataset/intratumor/img_test1", img_name)
    mergeRegion_path = os.path.join("../dataset/merge_region/img_test1", img_name)

    cv2.imwrite(peritumor_path, peritumor)
    cv2.imwrite(intratumor_path, intratumor)
    cv2.imwrite(mergeRegion_path, mergeRegion)