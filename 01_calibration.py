#!/usr/bin/python
# -*- coding: utf-8 -*-

# ステレオキャリブレーションによる基礎行列の計算

# x = 15  y = 8  (1280, 960)

import cv2 as cv
import numpy as np
import glob
import re
from os import path

# 画像の入力順番をソートする
def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


chessboardSize = (7, 10)
frameSize = (960, 1280)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1, 2)

objp = objp * 64
# print(objp)

objpoints = []
imgpointsL = []
imgpointsR = []

# ランダムに画像入力すると、上手くいかなかった
imagesLeft = sorted(glob.glob('stereo_camera/images/left/*.jpg'), key=natural_keys)
imagesRight = sorted(glob.glob('stereo_camera/images/right/*.jpg'), key=natural_keys)

for imgLeft, imgRight in zip(imagesLeft, imagesRight):

    imgL = cv.imread(imgLeft)
    imgR = cv.imread(imgRight)
    grayL = cv.cvtColor(imgL, cv.COLOR_BGRA2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGRA2GRAY)

    retL, cornersL = cv.findChessboardCorners(grayL, chessboardSize, None)
    retR, cornersR = cv.findChessboardCorners(grayR, chessboardSize, None)

    if retL and retR:

        objpoints.append(objp)

        cornersL = cv.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
        imgpointsL.append(cornersL)

        cornersR = cv.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
        imgpointsR.append(cornersR)

        cv.drawChessboardCorners(imgL, chessboardSize, cornersL, retL)
        cv.imshow('img left', imgL)
        cv.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)
        cv.imshow('img right', imgR)
        cv.waitKey(1000)

    # cv.destroyAllWindows()



retL, cameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(objpoints, imgpointsL, frameSize, None, None)
heightL, widthL, channelsL =imgL.shape
# newCameraMatrixL, roi_L = cv.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))
retR, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(objpoints, imgpointsR, frameSize, None, None)
heightR, widthR, channelsR =imgR.shape
# newCameraMatrixR, roi_R = cv.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))



flags = 0
flags |= cv.CALIB_FIX_INTRINSIC

criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, \
    trans, essentialMatrix, fundmenralMatrix = cv.stereoCalibrate(objpoints, imgpointsL, imgpointsR, cameraMatrixL,
                                                                  distL, cameraMatrixR, distR, frameSize,
                                                                  criteria=criteria_stereo, flags=flags)

print('KKLeft', newCameraMatrixL)
print('distCoffsLeft', distL)
print('KKRight', newCameraMatrixR)
print('distCoffsRight', distR)
print('R', rot)
print('T', trans)
print('E', essentialMatrix)
print('F', fundmenralMatrix)

np.savez_compressed('stereoCalib', KKLeft=newCameraMatrixL, distCoeffsLeft=distL,
                    KKRight=newCameraMatrixR, distCoeffsRight=distR, R=rot, T=trans, E=essentialMatrix, F=fundmenralMatrix)


# キャリブレーションの精度を示す（数値が小さいほど精度は高い）
print('retL', retL)
print('retR', retR)
print('retStereo', retStereo)


# rectifyScale = 1
# rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R = cv.stereoRectify(newCameraMatrixL, distL,
#                                                                            newCameraMatrixR, distR,
#                                                                            grayL.shape[::-1], rot, trans, rectifyScale, (0, 0))

# stereoMapL = cv.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, grayL.shape[::-1], cv.CV_16SC2)
# stereoMapR = cv.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, grayR.shape[::-1], cv.CV_16SC2)

# print("Saving parameters!")
# cv_file = cv.FileStorage('stereoMap.xml', cv.FILE_STORAGE_WRITE)

# cv_file.write('stereoMapL_x', stereoMapL[0])
# cv_file.write('stereoMapL_y', stereoMapL[1])
# cv_file.write('stereoMapR_x', stereoMapR[0])
# cv_file.write('stereoMapR_y', stereoMapR[1])

# cv_file.release()

