import argparse
import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from skimage import measure
from skimage.measure import label
from skimage.morphology import skeletonize
# def check_crack(image_input):
#     return len(measure.regionprops(label(image_input, connectivity=2)))

# def single_crack(point_crack):
#     clustering = DBSCAN(eps=5, min_samples=12).fit(point_crack)
#     return np.unique(clustering.labels_)
# if len(single_crack(point_crack)) >= 2:
#     print('No single crack')
#     return

# if np.size(point_crack) < 1:
#     print('image emtpy')
#     return
# img_path =''
# gt_path = ''
# img_list= glob.glob(r'D:\Tai Xuong0\crack datasets\CRACK500\traindata\crack500_1024\train_A\\'+'*.png')
# list_1 =[]
# print(len(img_list))

# for i in range(len(img_list)):
#     img = cv2.imread(img_list[i])
#     skeleton_lee = skeletonize(img, method='lee')
#     cv2.imshow('',skeleton_lee)
#     cv2.waitKey()
#     cv2.destroyAllWindows()
#     point_crack = np.argwhere(skeleton_lee==255)
#     print(len(single_crack(point_crack)))
#     # if len(single_crack(point_crack)) >= 2:
#     #     print('No single crack')
#     #     continue

#     # if np.size(point_crack) < 1:
#     #     print('image emtpy')
#     #     continue
#     cv2.imshow('',img)
#     cv2.waitKey()
#     cv2.destroyAllWindows()
#     #img = cv2.resize(img,(512,512))
#     print(np.unique(img))
#     if len(np.argwhere(img>0))/np.size(img)*100<10:
#         list_1.append(len(np.argwhere(img>0))/np.size(img)*100)
#     #print(len(np.argwhere(img>0))/np.size(img)*100)
# print(max(list_1))
# print(sum(list_1)/len(list_1))
# plt.hist(list_1,edgecolor='black')
# plt.show()

#########################################################################
path_gt = 'D:/pix2pixHD/train/label'
img_list= glob.glob(path_gt +'/' +'*.png')
path_save = 'D:/pix2pixHD/train/skeleton'
from skimage.morphology import skeletonize
for i in range(len(img_list)):
    img = cv2.imread(img_list[i])
    img = cv2.resize(img,(512,512))
    skeleton_lee = skeletonize(img, method='lee')
    print(np.unique(skeleton_lee))
    print(img_list[i][img_list[i].rfind('\\')+1:img_list[1].rfind('.')])
    name_path = path_save +'/'+ img_list[i][img_list[i].rfind('\\')+1:img_list[1].rfind('.')] + '.png'
    cv2.imwrite(name_path,skeleton_lee)
    # cv2.imshow('',skeleton_lee)
    # cv2.waitKey()
    # cv2.destroyAllWindows()