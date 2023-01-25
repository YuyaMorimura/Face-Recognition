
#!/usr/bin/python
# -*- coding: utf-8 -*-

# キャリブレーション情報から平行化を行う

import cv2
import numpy as np
import os
from os import path

def LoadParams(filepath='stereoCalib.npz'):  # ファイルの読み込み
    with np.load(filepath) as data:
        KKLeft, distCoeffsLeft, KKRight, distCoeffsRight, R, T, E, F = \
            [data[i] for i in ('KKLeft', 'distCoeffsLeft', 'KKRight', 'distCoeffsRight', 'R', 'T', 'E', 'F')]
    return KKLeft, distCoeffsLeft, KKRight, distCoeffsRight, R, T, E, F

def make_directories(dir_path):  # 平行化後の画像保存ディレクトリの作成
    os.makedirs(dir_path, exist_ok=True)  # ディレクトリの作成
    dir_path_left = '{}/{}'.format(dir_path, 'left')
    dir_path_right = '{}/{}'.format(dir_path, 'right')
    os.makedirs(dir_path_left, exist_ok=True)
    os.makedirs(dir_path_right, exist_ok=True)
    return dir_path_left, dir_path_right

def rectification() :   # main関数
    KKLeft, distCoeffsLeft, KKRight, distCoeffsRight, R, T, E, F = LoadParams()
    dir_path_left, dir_path_right = make_directories("stereo_camera/face_after")  # 保存したい場所

    N = 2
    for i in range(N):
        i += 1
        TgtImg_l = cv2.imread("stereo_camera/face_images/left/left" +str(i)+ ".jpg")  # 元画像のパス
        TgtImg_r = cv2.imread("stereo_camera/face_images/right/right" +str(i)+ ".jpg")

        flags = 0
        alpha = 1

        newimageSize=(TgtImg_l.shape[1],TgtImg_l.shape[0])
        R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(KKLeft, distCoeffsLeft,
                                                                          KKRight, distCoeffsRight,
                                                                          newimageSize, R, T, flags, alpha, newimageSize)
        # 平行化変換マップを求める
        m1type = cv2.CV_32FC1
        map1_l, map2_l = cv2.initUndistortRectifyMap(KKLeft, distCoeffsLeft, R1, P1, newimageSize, m1type)
        map1_r, map2_r = cv2.initUndistortRectifyMap(KKRight, distCoeffsRight, R2, P2, newimageSize, m1type)

        # ReMapにより平行化を行う
        interpolation = cv2.INTER_NEAREST
        Re_TgtImg_l = cv2.remap(TgtImg_l, map1_l, map2_l, interpolation)
        Re_TgtImg_r = cv2.remap(TgtImg_r, map1_r, map2_r, interpolation)

        # cv2.imshow('Rectified Left Target Image', Re_TgtImg_l)
        # cv2.imshow('Rectified Right Target Image', Re_TgtImg_r)

        cv2.imwrite('{}/{}{}.jpg'.format(dir_path_left, "left", str(i)), Re_TgtImg_l)
        cv2.imwrite('{}/{}{}.jpg'.format(dir_path_right, "right", str(i)), Re_TgtImg_r)