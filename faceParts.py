import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import open3d as o3d



def mouse(mmg1, mousePoint1) :
    h, w, c = mmg1.shape
    oneHeight = int(h / 100)
    oneWidth = int(w / 100)

    # mmg1[1:oneHeight*69, :] = [0, 0, 0]  # 口の領域
    # 口の領域
    for yy in range(h) :
        for xx in range(w) :
            if mousePoint1[0] < xx and xx < mousePoint1[0]+mousePoint1[2] and mousePoint1[1] < yy and yy < mousePoint1[1]+mousePoint1[3] :  # 前に検出したものを流用
                pass
            else :
                mmg1[yy, xx] = [0, 0, 0]

    # plt.imshow(mmg1)
    # plt.show()
    return mmg1


def nose(mmg1, mousePoint1) :
    h, w, c = mmg1.shape
    oneHeight = int(h / 100)
    oneWidth = int(w / 100)

    # 鼻の領域
    # mmg1[:oneHeight*50, :] = [0, 0, 0]  # 論文の位置関係より　目の領域を除去
    for yy in range(h):
        for xx in range(w):
            if mousePoint1[0] < xx and xx < mousePoint1[0]+mousePoint1[2] and 0 < yy and yy < mousePoint1[1]:  # 口領域より上の領域
                pass
            else:
                mmg1[yy, xx] = [0, 0, 0]

    # plt.imshow(mmg1)
    # plt.show()
    return mmg1


def eyes(mmg1, faceArea1, mousePoint) :
    h, w, c = mmg1.shape
    overNeck = int(mousePoint[3] / 0.16 * 0.15)  # 口から顔領域の高さ
    oneHeight = int((mousePoint[1]+mousePoint[3]+overNeck) / 100)
    '''
    count = 1
    endH = 0
    for n in range(h):
        for m in range(w) :
            if faceArea1[n][m][0] == 255 and faceArea1[n][m][1] == 255 and faceArea1[n][m][2] == 255 and count == 1 :
                startH = n
                count = 2
            if faceArea1[n][m][0] == 255 and faceArea1[n][m][1] == 255 and faceArea1[n][m][2] == 255 and endH < n :
                endH = n

    AreaHeight = endH - startH  # 顔領域の高さ
    '''
    cv2.imwrite('stereo_camera/mmg1/1.png', mmg1)
    # mmg1[mousePoint[1]:, :] = [0, 0, 0]  # 口の領域以下の領域を除去 → 口より上の領域に目の領域があるとする
    mmg1[mousePoint[1]-oneHeight*19 :, :] = [0, 0, 0]

    # plt.imshow(mmg1)
    # plt.show()

    # 肌色検出で眼球を除去
    mmg_edit = mmg1  # 編集画像
    hhh, www, c = mmg_edit.shape
    # 肌色検出
    for m in range(hhh):
        for n in range(www):
            if mmg_edit[m][n][0] < mmg_edit[m][n][1] and mmg_edit[m][n][1] < mmg_edit[m][n][2]:
                mmg_edit[m][n] = [0, 0, 0]
            else:
                pass

    im_gray = cv2.cvtColor(mmg_edit, cv2.COLOR_BGR2GRAY)
    th, im_gray = cv2.threshold(im_gray, 0, 255, cv2.THRESH_OTSU)

    # モルフォロジー変換によるノイズ除去
    kernel = np.ones((5, 5), np.uint8)
    im_gray = cv2.morphologyEx(im_gray, cv2.MORPH_CLOSE, kernel)  # クロージング
    # im_gray = cv2.erode(im_gray, kernel, iterations=2)  # 収縮

    # 顔領域の外側の輪郭検出
    faceArea111 = cv2.cvtColor(faceArea1, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(faceArea111, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img = cv2.cvtColor(faceArea111, cv2.COLOR_GRAY2BGR)
    img = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

    for m in range(hhh):
        for n in range(www):
            if img[m][n][0] == 0 and img[m][n][1] == 255 and img[m][n][0] == 0 :
                im_gray[m][n] = 255
    # plt.imshow(im_gray)
    # plt.show()

    # 全ての輪郭を取得  (https://pystyle.info/opencv-find-contours/)
    contours, hierarchy = cv2.findContours(im_gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # 画像表示用に入力画像をカラーデータに変換する
    img_disp = cv2.cvtColor(im_gray, cv2.COLOR_GRAY2BGR)

    eyeEndH = mousePoint[1] - 19*oneHeight  # 目の領域の一番下の領域
    Conlist = []  # 周囲長・輪郭の座標情報を格納するリスト
    for contour in contours:  # 輪郭の点の描画
        length = cv2.arcLength(contour, True)  # 周囲長
        xNate, yNate, width, height = cv2.boundingRect(contour)  # 輪郭の矩形領域
        if int(eyeEndH) >= yNate and int(eyeEndH) >= (yNate + height) :
        # if mousePoint[1] >= yNate and mousePoint[1] >= (yNate + height) :  # 口領域より上の領域に目がある
            # pass
            length = width
        else:
            length = 0
        oneRec = [length, xNate, yNate, width, height, contour]
        Conlist.append(oneRec)
    Conlist.sort(reverse=True)  # 周囲長が大きい順にソートする

    # 1つ目の眼球を抽出
    cv2.rectangle(img_disp, (Conlist[0][1], Conlist[0][2]),
                  (Conlist[0][1] + Conlist[0][3], Conlist[0][2] + Conlist[0][4]), (255, 0, 0),
                  thickness=5)  # 周囲長が一番長いものを採用 → 口の領域となる
    # 1つ目の眼球を抽出
    mmg1 = cv2.drawContours(mmg1, [Conlist[0][5]], 0, [255, 0, 0], -1)  # 輪郭内を塗りつぶす
    eyeCount = 1

    HHyy = abs(Conlist[0][2] - Conlist[1][2])  # 周囲長が最も長い場所の同じぐらいの高さに2番目に周囲長が長いものがあれば、長目空いているとみなし、抽出
    if 0 <= HHyy and HHyy <= 20 :  # 閾値は実験を通して実験
        # 2つ目の眼球はあれば抽出
        cv2.rectangle(img_disp, (Conlist[1][1], Conlist[1][2]),
                      (Conlist[1][1] + Conlist[1][3], Conlist[1][2] + Conlist[1][4]), (255, 0, 0),
                      thickness=5)
        # 2つ目の眼球はあれば抽出
        mmg1 = cv2.drawContours(mmg1, [Conlist[1][5]], 0, [255, 0, 0], -1)  # 輪郭内を塗りつぶす
        eyeCount = 2

    # plt.imshow(mmg1)
    # plt.show()

    # 目の領域を推定
    if eyeCount == 1:
        startEye = Conlist[0][2]
        endEye = Conlist[0][4] + Conlist[0][2]
    elif eyeCount == 2 :
        if Conlist[0][2] < Conlist[1][4] :
            startEye = Conlist[0][2]
        else :
            startEye = Conlist[1][2]
        if Conlist[0][4] + Conlist[1][2] < Conlist[1][4] + Conlist[1][2] :
            endEye = Conlist[1][4] + Conlist[1][2]
        else :
            endEye = Conlist[0][4] + Conlist[0][2]

    # 目の領域の抽出
    mmg1 = cv2.imread("stereo_camera/mmg1/1.png")  # 眼球の色を変更するときはコメントアウト
    centerEye = (endEye-startEye) / 2 + startEye  # 眼球の中心の高さ
    # centerMouth = (mousePoint[3]) / 2 + mousePoint[1]  # 口領域の中心の高さ
    oneHeight = (mousePoint[1]+mousePoint[3]-startEye) / (25 + 19 + 16)  # 眼球と口領域の高さを線分で結んで、それを基準にして顔の位置関係から目の領域を推定

    mmg1[int(centerEye + oneHeight*25/2):, :] = [0, 0, 0]  # 目の領域より下の領域を除去
    if int(centerEye - oneHeight * 25 / 2) <= 0 :  # 目の領域の上の座標がマイナスではなかったら除去（マイナスなら何もしない）
        pass
    else :
        mmg1[:int(centerEye - oneHeight * 25 / 2), :] = [0, 0, 0]  # 目の領域より上の領域を除去
    # mmg1[:startEye, :] = [0, 0, 0]  # 目の領域より下の領域を除去
    # mmg1[endEye:, :] = [0, 0, 0]  # 目の領域より下の領域を除去

    # mmg1[int(endEye):, :] = [0, 0, 0]  # 目の領域より下の領域を除去
    # mmg1[:int(startEye), :] = [0, 0, 0]  # 目の領域より上の領域を除去

    # plt.imshow(mmg1)
    # plt.show()
    return mmg1
