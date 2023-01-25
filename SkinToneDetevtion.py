import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def skintone(lmg, adress) :
    lmg_edit = lmg  # 編集画像
    h, w , c = lmg.shape

    # 肌色検出
    for m in range(h) :
        for n in range(w) :
            if lmg[m][n][0] < lmg[m][n][1] and lmg[m][n][1] < lmg[m][n][2]:
                pass
            else :
                lmg_edit[m][n] = [0, 0, 0]

    #  グレースケール・2値化
    im_gray = cv.cvtColor(lmg_edit, cv.COLOR_BGR2GRAY)
    th, im_gray = cv.threshold(im_gray, 0, 255, cv.THRESH_OTSU)
    # im_gray = 255 - im_gray  # 白黒反転

    # モルフォロジー変換による微調整
    kernel = np.ones((5, 5), np.uint8)
    im_gray = cv.morphologyEx(im_gray, cv.MORPH_CLOSE, kernel)  # クロージング

    # 両端を黒く塗りつぶすことで肌色検出時に首がはみ出しても大丈夫なようにする
    im_gray[h-70:h] = 0

    # plt.imshow(im_gray)
    # plt.show()

    cv.imwrite(adress, im_gray)



# チェインコードを求め、顔の輪郭のみを検出（大学window 3年秋学期/画像処理/No10/19K1138第10回課題   MATLAB）
def chain_code(sample, adress) :
    h, w = sample.shape

    chain_sam = np.zeros((h, w))
    chain_code_start = np.zeros((600, 2))  # 開始点の座標
    chain_code_point = np.ones((600, 6000)) * 1000  #  方向コード(コードではないのは、要素が1000とする

    max_count = 0  #  チェインコードのmax要素数（1行ごと）
    count_gyo = 0  # 一番チェインコードが多い行数
    a = 0

    for y in range(100, h-100):
        for x in range(100, w-100):
            if sample[y, x] == 0 and sample[y, x + 1] == 255 and chain_sam[y, x + 1] == 0 :  # 開始点の検索
                chain_code_start[a, 0] = x + 1
                chain_code_start[a, 1] = y
                yy = y
                xx = x + 1
                d = [5, 6, 7, 0, 1, 2, 3, 4]  # 探索する順番

                b = 0
                tt = "True"
                while tt == "True" and yy < h-1 and xx < w-1 :
                    tt = "False"
                    for dd in d :
                        if sample[yy, xx + 1] == 255 and dd == 0 :
                            chain_code_point[a, b] = 0 # 方向コードの追加
                            b = b + 1
                            yy = yy
                            xx = xx + 1
                            d = [5, 6, 7, 0, 1, 2, 3, 4]  # 探索する順番
                            tt = "True"
                            chain_sam[yy, xx] = 255
                            break
                        elif sample[yy + 1, xx + 1] == 255 and dd == 1 :
                            chain_code_point[a, b] = 1 # 方向コードの追加
                            b = b + 1
                            yy = yy + 1
                            xx = xx + 1
                            d = [6, 7, 0, 1, 2, 3, 4, 5]   # 探索する順番
                            tt = "True"
                            chain_sam[yy, xx] = 255
                            break
                        elif sample[yy + 1, xx] == 255 and dd == 2 :
                            chain_code_point[a, b] = 2  # 方向コードの追加
                            b = b + 1
                            yy = yy + 1
                            xx = xx
                            d = [7, 0, 1, 2, 3, 4, 5, 6]   # 探索する順番
                            tt = "True"
                            chain_sam[yy, xx] = 255
                            break
                        elif sample[yy + 1, xx - 1] == 255 and dd == 3 :
                            chain_code_point[a, b] = 3  # 方向コードの追加
                            b = b + 1
                            yy = yy + 1
                            xx = xx - 1
                            d = [0, 1, 2, 3, 4, 5, 6, 7]  # 探索する順番
                            tt = "True"
                            chain_sam[yy, xx] = 255
                            break
                        elif sample[yy, xx - 1] == 255 and dd == 4 :
                            chain_code_point[a, b] = 4 # 方向コードの追加
                            b = b + 1
                            yy = yy
                            xx = xx - 1
                            d = [1, 2, 3, 4, 5, 6, 7, 0]  # 探索する順番
                            tt = "True"
                            chain_sam[yy, xx] = 255
                            break
                        elif sample[yy - 1, xx - 1] == 255 and dd == 5 :
                            chain_code_point[a, b] = 5  # 方向コードの追加
                            b = b + 1
                            yy = yy - 1
                            xx = xx - 1
                            d = [2, 3, 4, 5, 6, 7, 0, 1]  #  探索する順番
                            tt = "True"
                            chain_sam[yy, xx] = 255
                            break
                        elif sample[yy - 1, xx] == 255 and dd == 6  :
                            chain_code_point[a, b] = 6  # 方向コードの追加
                            b = b + 1
                            yy = yy - 1
                            xx = xx
                            d = [3, 4, 5, 6, 7, 0, 1, 2] # 探索する順番
                            tt = "True"
                            chain_sam[yy, xx] = 255
                            break
                        elif sample[yy - 1, xx + 1] == 255 and dd == 7 :
                            chain_code_point[a, b] = 7  # 方向コードの追加
                            b = b + 1
                            yy = yy - 1
                            xx = xx + 1
                            d = [4, 5, 6, 7, 0, 1, 2, 3]  # 探索する順番
                            tt = "True"
                            chain_sam[yy, xx] = 255
                            break
                    if xx == chain_code_start[a, 0] and yy == chain_code_start[a, 1] :
                        tt = "False"

                a = a + 1
                if max_count < b :
                    max_count = b
                    count_gyo = a - 1  # 顔のチェインコードがある行数
    print(max_count)
    print(count_gyo)


    # 顔の輪郭を描画する
    chain_sam = np.zeros((h, w))
    start_x = int(chain_code_start[count_gyo, 0])  # チェインコードの開始座標
    start_y = int(chain_code_start[count_gyo, 1])
    chain_code = chain_code_point[count_gyo, :]  # チェインコード

    chain_sam[start_y, start_x] = 255
    for dd in chain_code :
        if dd==0 :
            start_y = start_y;  start_x = start_x + 1
            chain_sam[start_y, start_x] = 255
        elif dd==1 :
            start_y = start_y + 1;  start_x = start_x + 1
            chain_sam[start_y, start_x] = 255
        elif dd==2 :
            start_y = start_y + 1;  start_x = start_x
            chain_sam[start_y, start_x] = 255
        elif dd==3 :
            start_y = start_y+ 1;  start_x = start_x - 1
            chain_sam[start_y, start_x] = 255
        elif dd==4 :
            start_y = start_y;  start_x = start_x - 1
            chain_sam[start_y, start_x] = 255
        elif dd==5 :
            start_y = start_y - 1;  start_x = start_x - 1
            chain_sam[start_y, start_x] = 255
        elif dd==6 :
            start_y = start_y - 1;  start_x = start_x
            chain_sam[start_y, start_x] = 255
        elif dd==7:
            start_y = start_y - 1;  start_x = start_x + 1
            chain_sam[start_y, start_x] = 255
        else :
            break

    # plt.imshow(chain_sam, cmap='gray')
    # plt.show()
    cv.imwrite(adress, chain_sam)

    # if adress == l_adress :
    #     cv.imwrite("stereo_camera/outline/left1.jpg", chain_sam)  # 顔領域の輪郭の画像を保存


# 顔の輪郭内を塗りつぶす
def face_chose(im, lmg, adress, erosion) :
    h, w = im.shape
    lmg22 = cv.imread(lmg)
    contours, hierarchy = cv.findContours(im,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    img = cv.cvtColor(im, cv.COLOR_GRAY2BGR)
    img = cv.drawContours(img, contours, -1, (0,255,0), 3)
    # plt.imshow(img)
    # plt.show()

    img2 = cv.cvtColor(im, cv.COLOR_GRAY2BGR)

    color_list = []
    for cc in range(len(contours)):
        color_list.append((255, 255, 255))

    for i in range(len(contours)):
        cnt = contours[i]
        img2 = cv.drawContours(img2, [cnt], 0, color_list[i], -1)

    # ノイズ除去
    kernel = np.ones((10, 10), np.uint8)
    img2 = cv.erode(img2, kernel, iterations=1)  #モルフォロジー変換（収縮）する
    # plt.imshow(img2)
    # plt.show()


    lmg22 = cv.imread(lmg)
    # plt.imshow(lmg22)
    # plt.show()

    back_color = [0, 0, 0]
    # RGBに対応させる
    for m in range(h) :
        for n in range(w) :
            # チェインコードによる処理
            if img2[m][n][0] == 255 and img2[m][n][1] == 255 and img2[m][n][2] == 255 :
                pass
            else:
                lmg22[m][n] = back_color

    lmg22 = cv.cvtColor(lmg22, cv.COLOR_BGR2RGB)  # GBRからRGBに変換
    # plt.imshow(lmg22)
    # plt.show()
    cv.imwrite(adress, lmg22)

    # if adress == l_adress :
    cv.imwrite(erosion, img2)  # 顔領域の座標情報の画像を保存


def makeFace(lmg, l_adress, erosion, lmg22, rmg22) :

    skintone(lmg, l_adress)
    # skintone(rmg, r_adress)

    l_sample = cv.imread(l_adress, cv.COLOR_BGR2GRAY)
    # r_sample = cv.imread(r_adress, cv.COLOR_BGR2GRAY)

    chain_code(l_sample, l_adress)
    # chain_code(r_sample, r_adress)

    lim = cv.imread(l_adress, flags=cv.IMREAD_GRAYSCALE)
    # rim = cv.imread(r_adress, flags=cv.IMREAD_GRAYSCALE)

    face_chose(lim, lmg22, l_adress, erosion)
    # face_chose(rim, rmg22, r_adress, erosion)




# main


'''
#画像データの読み込み
lmg1 = cv.imread("stereo_camera/face_after/left/left1.jpg")
lmg2 = cv.imread("stereo_camera/face_after/left/left2.jpg")
# rmg = cv.imread("stereo_camera/face_after/right/right1.jpg")
l_adress1 = 'stereo_camera/face_finish/left/left1.jpg'
l_adress2 = 'stereo_camera/face_finish/left/left2.jpg'
# r_adress = 'stereo_camera/face_finish/right/right1.jpg'
erosion1 = "stereo_camera/erosion/left1.jpg"  # 顔領域を示した画像の保存パス
erosion2 = "stereo_camera/erosion/left2.jpg"  # 顔領域を示した画像の保存パス

lmg11 = "stereo_camera/face_after/left/left1.jpg"
rmg11 = "stereo_camera/face_after/right/right1.jpg"
lmg22 = "stereo_camera/face_after/left/left2.jpg"
rmg22 = "stereo_camera/face_after/right/right2.jpg"

h, w, ch = lmg1.shape
back_color = [0, 0, 0]  # 背景色の決定 [155, 162, 155]

makeFace(lmg1, l_adress1, erosion1, lmg11, rmg11)  # mainの実行関数
makeFace(lmg2, l_adress2, erosion2, lmg22, rmg22)  # mainの実行関数


print("end")

'''