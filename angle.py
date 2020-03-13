#!/usr/bin/env python
# coding: utf-8

# In[11]:


import cv2
import math
import numpy as np
import xml.etree.ElementTree as ET
import os
import argparse

def rotate_image(src, angle, scale=1):
    w = src.shape[1]
    h = src.shape[0]
    # 角度變弧度
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
    nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]
    dst = cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)
    # 仿射變換
    return dst

# 對應修改xml檔案
def rotate_xml(src, xmin, ymin, xmax, ymax, angle, scale=1.):
    w = src.shape[1]
    h = src.shape[0]
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    # 獲取旋轉後圖像的長和寬
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]                                   # rot_mat是最終的旋轉矩陣
    # point1 = np.dot(rot_mat, np.array([xmin, ymin, 1]))          #這種新畫出的框大一圈
    # point2 = np.dot(rot_mat, np.array([xmax, ymin, 1]))
    # point3 = np.dot(rot_mat, np.array([xmax, ymax, 1]))
    # point4 = np.dot(rot_mat, np.array([xmin, ymax, 1]))
    point1 = np.dot(rot_mat, np.array([(xmin+xmax)/2, ymin, 1]))   # 獲取原始矩形的四個中點，然後將這四個點轉換到旋轉後的座標系下
    point2 = np.dot(rot_mat, np.array([xmax, (ymin+ymax)/2, 1]))
    point3 = np.dot(rot_mat, np.array([(xmin+xmax)/2, ymax, 1]))
    point4 = np.dot(rot_mat, np.array([xmin, (ymin+ymax)/2, 1]))
    concat = np.vstack((point1, point2, point3, point4))            # 合併np.array
    # 改變array型別
    concat = concat.astype(np.int32)
    rx, ry, rw, rh = cv2.boundingRect(concat)                        #rx,ry,為新的外接框左上角座標，rw為框寬度，rh為高度，新的xmax=rx+rw,新的ymax=ry+rh
    return rx, ry, rw, rh

# 使影象旋轉60,90,120,150,210,240,300度
def angel(xmlpath1, imgpath1, rotated_xmlpath1, rotated_imgpath1):
    xmlpath = xmlpath1         #源影象路徑
    imgpath = imgpath1         #源影象所對應的xml檔案路徑
    rotated_imgpath = rotated_imgpath1
    rotated_xmlpath = rotated_xmlpath1
    for angle in (0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345):
        for i in os.listdir(imgpath):
            a, b = os.path.splitext(i)                            #分離出檔名a
            img = cv2.imread(imgpath + a + '.jpg')
            rotated_img = rotate_image(img,angle)
            cv2.imwrite(rotated_imgpath + a + '_'+ str(angle) +'d.jpg',rotated_img)
            print (str(i) + ' has been rotated for '+ str(angle)+'°')
            tree = ET.parse(xmlpath + a + '.xml')
            root = tree.getroot()
            root.iter('filename')
            for box in root.iter('bndbox'):
                xmin = float(box.find('xmin').text)
                ymin = float(box.find('ymin').text)
                xmax = float(box.find('xmax').text)
                ymax = float(box.find('ymax').text)
                x, y, w, h = rotate_xml(img, xmin, ymin, xmax, ymax, angle)
                # cv2.rectangle(rotated_img, (x, y), (x+w, y+h), [0, 0, 255], 2)   #可在該步驟測試新畫的框位置是否正確
                # cv2.imshow('xmlbnd',rotated_img)
                # cv2.waitKey(200)
                box.find('xmin').text = str(x)
                box.find('ymin').text = str(y)
                box.find('xmax').text = str(x+w)
                box.find('ymax').text = str(y+h)
            tree.write(rotated_xmlpath + a + '_'+ str(angle) +'d.xml')
            print (str(a) + '.xml has been rotated for '+ str(angle)+'°')

if __name__ == "__main__":
    # Path of the images
    parser = argparse.ArgumentParser()
    parser.add_argument("xmlpath")
    parser.add_argument("imgpath")
    parser.add_argument("rotated_xmlpath")
    parser.add_argument("rotated_imgpath")
    args = parser.parse_args()
    angel(args.xmlpath, args.imgpath, args.rotated_xmlpath, args.rotated_imgpath)

