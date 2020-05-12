import numpy
import dlib
import cv2
import sys
import os
import random


def get_border(shape,keypoint):  # (68,2)
    res = []
    for idx,pt in enumerate(shape.parts()):
        if idx in keypoint:
            res.append([pt.x,pt.y])
    return res[0][0],res[3][1],res[2][0],res[1][1] # 返回对应的x1,y1,x2,y2
def wear_mask(img,mask,keypoint):
    ret = img.copy()
    # mask_use = mask.copy()
    # 人脸检测
    detector = dlib.get_frontal_face_detector()

    # 人脸关键点标注。
    predictor = dlib.shape_predictor(
        'shape_predictor_68_face_landmarks.dat'
    )
    # 转灰度图。
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    dets = detector(gray,0)# 第二个参数越大，代表讲原图放大多少倍在进行检测，提高小人脸的检测效果。

    for d in dets:
        # d 为每一张人脸检测框
        # 使用predictor进行人脸关键点检测 shape为返回的结果（68,2）
        shape = predictor(gray, d)
        x1,y1,x2,y2 = get_border(shape,keypoint)
        mask_use = cv2.resize(mask, (x2-x1, y2-y1))
        w, h = x2-x1, y2-y1
        for i in range(h):
            for j in range(w):
                ret[y1+i, x1+j] = ret[y1+i, x1+j]*0.2 + (mask_use[i,j]*0.8 if mask_use[i,j,0]>80 else ret[y1+i, x1+j]*0.8)
        #ret[y1:y2, x1:x2] = ret[y1:y2, x1:x2]*0.3 + mask_use[:,:]*0.7

        # for index, pt in enumerate(shape.parts()):
        #     print('Part {}: {}'.format(index, pt))
        #     pt_pos = (pt.x, pt.y)
        #     if index in [2, 8, 14, 28]:
        #         points_key.append(pt_pos)

    return ret



if __name__ == "__main__":

    keypoint = [2, 8, 14, 28]
    save_name = 'already_' + '%04d'%(random.randint(1,1000))+'.jpg'
    #dirname, filename = os.path.split(os.path.abspath(sys.argv[0]))
    dirname, filename = os.path.split(sys.argv[0])
    # print(dirname,filename)
    # path = os.getcwd()
    os.chdir(dirname)
    print(os.getcwd())
    path_img = input('Input image filename:')

    img = cv2.imread(path_img)

    mask = cv2.imread('simple_mask.png')
    # cv2.imshow('1',img)
    # cv2.waitKey(0)
    # cv2.imshow('2',mask)
    # cv2.waitKey(0)
    ans = wear_mask(img,mask,keypoint)
    cv2.imshow('Image_wearing_mask',ans)
    cv2.waitKey(0)
    cv2.imwrite(save_name, ans)














