import cv2
import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d
from scipy import cluster
from scipy import ndimage
from scipy.spatial.distance import euclidean
from statistics import stdev
# import py3d
from PIL import Image

def sgbm(rimg1, rimg2, max_search):  # disparityの作成
    # run SGM stereo matching with weighted least squares filtering
    #print('Running SGBM stereo matcher...')
    if len(rimg1.shape) > 2:
        # Grayscale Images
        rimg1 = cv2.cvtColor(rimg1, cv2.COLOR_BGR2GRAY)
        rimg2 = cv2.cvtColor(rimg2, cv2.COLOR_BGR2GRAY)

    # Creating an object of StereoBM algorithm
    maxd = max_search #220 #DMAX_SEARCH
    #print('MAXD = ', maxd)
    window_size = 5
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=-maxd,  # 可能な最小視差値。通常はゼロ
        numDisparities=maxd * 2,  # 最大視差から最小視差を引いたもの. 16で割り切れる必要がある
        blockSize=5,  # 一致したブロックサイズ。奇数>=1でないとならない
        P1=8 * 3 * window_size ** 2,  # 視差の滑らかさを制御する最初のパラメータ
        P2=32 * 3 * window_size ** 2,  # 視差の滑らかさを制御する2番目のパラメータ
        disp12MaxDiff=1,  # 左右の視差チェックで許容される最大差（整数ピクセル単位）
        uniquenessRatio=15,  # 計算された最良の（最小の）コスト関数値が、見つかった一致が正しいと見なすために2番目に良い値に「勝つ」必要があるパーセンテージのマージン
        speckleWindowSize=0,  # ノイズスペックルを考慮して無効化するための滑らかな視差領域の最大サイズ. スペックルフィルタリングを無効にするには、0に設定
        speckleRange=2,   # 接続された各コンポーネント内の最大視差変動
        preFilterCap=63,  # 事前フィルタにおいて，画像ピクセルを切り捨てる閾値
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # Compute Disparity Map
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    lmbda = 8000
    sigma = 1.5

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)

    wls_filter.setSigmaColor(sigma)
    displ = left_matcher.compute(rimg1, rimg2)
    dispr = right_matcher.compute(rimg2, rimg1)
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    disparity = wls_filter.filter(displ, rimg1, None, dispr) / 16.0

    return disparity



# Depth画像の作成
def cereateDeaph(lmg, rmg, max_search, para):
    # max_search = 270  # 任意の値を入力（観測対象が近いほど、値を大きくする）240 160
    disparity = sgbm(lmg, rmg, max_search)  # disparityを求める（depthではない）
    # plt.imshow(disparity, cmap='gray')
    # plt.show()
    # i = 1
    # cv2.imwrite('{}/{}{}.png'.format("stereo_camera/depth", "depth", str(i)), disparity)  # disparityの保存（意味ない）

    h, w, ch = lmg.shape
    # キャリブレーションで求めた値を入力する
    ff = 818.54596155
    bb = 0.08  # baseline length（ステレオカメラのカメラ間の長さ）
    tu = 678.35189098
    tv = 528.60570639

    # 50:900, 250:1000
    disp = disparity[50:900, 250:1000]  # disparityを求めたときにカットされる領域を考慮・0MensekiSeach()関数でも使用しているので注意
    # disp = disparity
    # plt.imshow(disp, cmap='gray')
    # plt.title('disp')
    # plt.show()

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    # dp = np.zeros((h,w,3)).asarray(float32)
    dp = np.zeros((h, w, 3))
    dp[:, :, 0] = bb * (x - tu) / disparity
    dp[:, :, 1] = bb * (y - tv) / disparity
    dp[:, :, 2] = bb * ff / disparity
    ddp = dp[50:900, 250:1000, :]   #3D image
    # ddp = dp
    # plt.imshow(ddp[:,:,2], cmap='gray')
    # plt.title('ddp')
    # plt.show()

    mmg = lmg[50:900, 250:1000, :]
    # mmg = lmg
    # plt.imshow(mmg)
    # plt.show()

    # plt.figure()
    # plt.imshow(mmg)
    # plt.figure()
    # plt.imshow(ddp[:,:,2])
    # plt.show()
    print('mmg:', mmg.shape)
    print('ddp:', ddp[:,:,2].shape, ddp.dtype)
    ddpz = ddp[:,:,2]
    rmax = np.max(ddpz)
    rmin = np.min(ddpz)
    print('max=', rmax, 'min=', rmin)

    ddpp = (ddpz - 3.0) * para  # open3Dにするときに、uint8にするので、それに対応する操作
    rmax = np.max(ddpp)
    rmin = np.min(ddpp)
    print('max=', rmax, 'min=', rmin)
    #ddpp = ddp[:,:,2]*500
    # plt.imshow(ddpp)
    # plt.show()

    return mmg, ddpp, disparity



# 首領域の除去
def HeadDelete(depth, rgb, erosion, para) :
    h, w = depth.shape
    kernel_x = np.array([[0, -1, 0],
                         [0, 0, 0],
                         [0, 1, 0]])
    gray_x = cv2.filter2D(depth, cv2.CV_64F, kernel_x)  # 1次微分による、水平方向の輪郭検出
    gray_x = gray_x * para
    # plt.imshow(gray_x)
    # plt.show()

    line = np.zeros((h, w))  # 画像
    pointx = []  # 輪郭のx座標を格納
    faceArea = cv2.imread(erosion)  # 顔領域の座標情報の画像を読み込み(3次元配列・255が顔領域　それ以外は0)
    faceArea = faceArea[50:900, 250:1000, :]
    kernel = np.ones((50, 50), np.uint8)
    faceArea = cv2.erode(faceArea, kernel, iterations=1)  # 収縮(モルフォロジー変換)することが一回り小さくする

    for y in range(h - 1):
        for x in range(w - 1):
            solar = gray_x[y, x] - gray_x[y + 1, x]
            # 自分で決めたしきい値で輪郭判定
            if solar > 600 and faceArea[y, x][0] == 255 and faceArea[y, x][1] == 255 and faceArea[y, x][2] == 255:
                line[y, x] = 255
            if faceArea[y, x][0] == 255 and faceArea[y, x][1] == 255 and faceArea[y, x][2] == 255:
                pointx.append(x)
    kernel = np.ones((5, 5), np.uint8)
    line = cv2.dilate(line, kernel, iterations=1)
    # plt.gray()
    # plt.imshow(line)
    # plt.show()

    cou = 0
    centerx = int((max(pointx) - min(pointx))/2 + min(pointx))  # 顔領域において、x方向の中心を求める
    print("centerx:", centerx)
    for y in reversed(range(h - 1)):  # reversed()は逆の順番で数値を定義する
        for cccX in range(centerx-3, centerx+3) :
            if line[y, cccX] == 255 and line[y-1, cccX] == 255 and line[y-2, cccX] == 255 and cou == 0:
                neck = [y, cccX]  # 顎の座標
                cou = 1
                # break
    print("neck:", neck)

    faceArea = cv2.imread(erosion)  # 顔領域の座標情報の画像を読み込み(3次元配列・255が顔領域　それ以外は0)
    faceArea = faceArea[50:900, 250:1000, :]
    kernel = np.ones((10, 10), np.uint8)  # 顔領域と背景の境界付近の外れ値の除去
    faceArea = cv2.erode(faceArea, kernel, iterations=1)

    facex = []
    facey = []
    for y in range(h):
        for x in range(w):
            # 顎の座標より低い点はすべて背景をみなす
            if neck[0] < y :
                depth[y, x] = 0
                rgb[y, x] = 0
            # 顔領域の一回り小さい領域を顔領域だと再定義（境界線部分で外れ値が発生している）
            if faceArea[y, x][0] == 0 and faceArea[y, x][1] == 0 and faceArea[y, x][2] == 0:
                depth[y, x] = 0
                rgb[y, x] = 0
            # 顔領域の座標を記録
            if depth[y, x] != 0 :
                facex.append(x)
                facey.append(y)
    # 背景をなるべくカットする
    depth = depth[min(facey):max(facey), min(facex):max(facex)]
    rgb = rgb[min(facey):max(facey), min(facex):max(facex)]
    faceArea = faceArea[min(facey):max(facey), min(facex):max(facex)]
    neck[0] = neck[0] - min(facey)
    neck[1] = neck[1] - (min(facex))

    scaleList = [min(facey), max(facey), min(facex), max(facex)]

    print("neckAfter:", neck)
    # plt.imshow(rgb)
    # plt.show()
    return rgb, depth, faceArea, neck, scaleList


def earDelete(depth, adress, scaleList, neck, faceArea, rgb):   # 耳の除去
    h, w = depth.shape

    # 肌色検出
    lmg = cv2.imread(adress)
    lmg = lmg[50:900, 250:1000, :]
    lmg = lmg[scaleList[0]:scaleList[1], scaleList[2]:scaleList[3]]
    lmg_edit = lmg  # 編集画像
    for m in range(h):
        for n in range(w):
            if lmg[m][n][0] < lmg[m][n][1] and lmg[m][n][1] < lmg[m][n][2]:
                pass
            else:
                lmg_edit[m][n] = [0, 0, 0]

    #  グレースケール・2値化
    im_gray = cv2.cvtColor(lmg_edit, cv2.COLOR_BGR2GRAY)
    th, im_gray = cv2.threshold(im_gray, 0, 255, cv2.THRESH_OTSU)

    for m in range(h):
        for n in range(w):
            if m > (h - 5):
                im_gray[m][n] = 255

    kernel = np.ones((5, 5), np.uint8)
    # im_gray = cv2.morphologyEx(im_gray, cv2.MORPH_CLOSE, kernel)  # クロージングによって肌色検出時のノイズを除去
    im_gray = cv2.erode(im_gray, kernel, iterations=1)  # 収縮より、口の領域を1つに繋げる
    im_gray = cv2.dilate(im_gray, kernel, iterations=1)  # 膨張より、境界線の凸凹をなくす

    # plt.imshow(im_gray)
    # plt.gray()
    # plt.show()

    # 全ての輪郭を取得  (https://pystyle.info/opencv-find-contours/)
    contours, hierarchy = cv2.findContours(im_gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # 画像表示用に入力画像をカラーデータに変換する
    img_disp = cv2.cvtColor(im_gray, cv2.COLOR_GRAY2BGR)

    Conlist = []  # 周囲長・輪郭の座標情報を格納するリスト
    for contour in contours:  # 輪郭の点の描画
        length = cv2.arcLength(contour, True)  # 周囲長
        xNate, yNate, width, height = cv2.boundingRect(contour)  # 輪郭の矩形領域
        if int(h/2) < yNate and yNate < neck[0]:  # 顔の位置関係より、y座標が顔領域（画像）の半分より大きく（下で）、顎より小さい（上）のやつのみを採用
            # pass
            length = width   # 横幅がもっともながいものを口領域だとみなす
        else:
            length = 0
        oneRec = [length, xNate, yNate, width, height]
        Conlist.append(oneRec)

    Conlist.sort(reverse=True)  # 周囲長が大きい順にソートする

    '''
    Conlist[0][1] = Conlist[0][1] + 5
    Conlist[0][2] = Conlist[0][2] + 5
    Conlist[0][3] = Conlist[0][3] - 10
    Conlist[0][4] = Conlist[0][4] - 10
    '''

    cv2.rectangle(img_disp, (Conlist[0][1], Conlist[0][2]),
                  (Conlist[0][1] + Conlist[0][3], Conlist[0][2] + Conlist[0][4]), (255, 0, 0),
                  thickness=5)   # 周囲長が一番長いものを採用 → 口の領域となる

    # mouseArea = depth[Conlist[0][2] : Conlist[0][2] + Conlist[0][4], Conlist[0][1] : Conlist[0][1] + Conlist[0][3]]
    # plt.imshow(img_disp)
    # plt.show()

    # 顔の位置関係より耳を除去（特講で使用した論文より）
    xPointLeft = Conlist[0][1] - int(Conlist[0][3] / 2)
    xPointRight = Conlist[0][1] + int(Conlist[0][3] / 2) + Conlist[0][3]

    if xPointLeft > 0 :  # 顔のラインが顔領域以内かチェック
        im_gray = im_gray[:, xPointLeft:]
        depth = depth[:, xPointLeft:]
        faceArea = faceArea[:, xPointLeft:]
        rgb = rgb[:, xPointLeft:]
        neck[1] = neck[1] - xPointLeft
        xPointRight = xPointRight - xPointLeft
        Conlist[0][1] = Conlist[0][1] - xPointLeft

    hh, ww = depth.shape
    if xPointRight < ww :  # 顔のラインが顔領域以内かチェック
        im_gray = im_gray[:, :xPointRight]
        depth = depth[:, :xPointRight]
        faceArea = faceArea[:, :xPointRight]
        rgb = rgb[:, :xPointRight]

    # plt.imshow(img_disp)
    # plt.show()
    # plt.imshow(rgb)
    # plt.show()

    '''
    # 口の領域の高さを正しく検出できていない可能性があるから、検出した高さの中心部分を求め、顔の位置関係から必要は高さを検出
    centerMouth = Conlist[0][4]/2 + Conlist[0][2]  # 検出した口領域の高さを検出
    Conlist[0][2] =
    '''


    mousePoint = [Conlist[0][1], Conlist[0][2], Conlist[0][3], Conlist[0][4]]
    return rgb, depth, faceArea, neck, mousePoint


def HeadAbolition(depth, rgb, faceArea, neck, mousePoint) :   # 首の完全除去
    h, w = depth.shape


    # faceAreaの輪郭を抽出
    im = cv2.cvtColor(faceArea, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    img = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    # plt.imshow(img)
    # plt.show()

    # 顔領域を縦に2分割して、それぞれの領域においてfaceAreaと口領域から鰓（えら）を求める
    count = 0
    y = mousePoint[1]  # 上唇のx座標
    for x in range(0, int(w / 2)):
        if img[y][x][0] == 0 and img[y][x][1] == 255 and img[y][x][2] == 0 and count == 0:
            count = count + 1
            leftNeck = [y, x]  # 鰓の座標
    print("Start leftNeck's point : ", leftNeck)

    count = 0
    y = mousePoint[1]  # 上唇のx座標
    for x in reversed(range(int(w / 2), w)):
        if img[y][x][0] == 0 and img[y][x][1] == 255 and img[y][x][2] == 0 and count == 0:
            count = count + 1
            rightNeck = [y, x]  # 鰓の座標
    print("Start rightNeck's point : ", rightNeck)

    # それぞれの鰓の座標と、顎の座標を結ぶ（線の色は(255, 0, 0)）
    cv2.line(img, (leftNeck[1], leftNeck[0]), (neck[1], neck[0]), (255, 0, 0), 5)
    cv2.line(img, (rightNeck[1], rightNeck[0]), (neck[1], neck[0]), (255, 0, 0), 5)

    img[leftNeck[0]-5:leftNeck[0], :leftNeck[1]] = [255, 0, 0]
    img[rightNeck[0]-5:leftNeck[0], rightNeck[1]:] = [255, 0, 0]

    # plt.imshow(img)
    # plt.show()

    for x in range(w):
        for y in range(h):
            if img[y][x][0] == 255 and img[y][x][1] == 0 and img[y][x][2] == 0 :  # 首のラインの検索
                for m in range(y, h):
                    rgb[m, x] = 0
                    faceArea[m, x] = 0
                    

    # 首のあまりを除去（特講論文の顔の位置関係より）
    overNeck = int(mousePoint[3]/0.16 * 0.15)
    rgb[mousePoint[1]+mousePoint[3]+overNeck : h, :] = 0
    faceArea[mousePoint[1]+mousePoint[3]+overNeck : h, :] = 0


    # plt.imshow(rgb)
    # plt.show()

    return rgb, faceArea

def deleteOutlierNose(mmg, faceArea, ddpp, mousePoint):  # 鼻の外れ値（照明が反射しているため）を削除する　（対象の顔画像のみ適用）
    noseHeight = mousePoint[1]
    noseWidth = mousePoint[2]

    listArea = []
    for y in range(0, noseHeight):
        for x in range(mousePoint[0], mousePoint[0] + noseWidth) :
                point = [ddpp[y, x], y, x]
                listArea.append(point)
    listArea.sort(reverse=False)  # depth値が小さい順にソートする

    # 口より上の領域のうち、下位2％のDepth値が外れ値だと仮定して、除去する
    count = int(noseHeight * noseWidth * 0.01)
    for i in range(0, count) :
        mmg[listArea[i][1], listArea[i][2]] = [0, 0, 0]

    # plt.imshow(mmg)
    # plt.show()
    return mmg


# 背景である座標をリスト化
def BackSeach(faceArea):
    img = faceArea
    h3, w3, ch3 = img.shape  # 3D化するデータの寸法の取得
    # img = img[50:900, 250:1000, :]

    # kernel = np.ones((20, 20), np.uint8)
    # img = cv2.erode(img, kernel, iterations=1)  # モルフォロジー変換（収縮）する

    mlist = []
    for m in range(h3):
        for n in range(w3):
            if img[m][n][0] == 0 and img[m][n][1] == 0 and img[m][n][2] == 0 :
                number = w3 * m + n  # 3D化すると行列から配列になるため、3D化の仕様に合わせる
                mlist.append(number)
    # plt.imshow(img)
    # plt.show()
    return mlist


# 必要な分の顔領域の取得
def startpointCenter(faceArea, neck, quantity, rgb):
    h, w, ch =faceArea.shape

    # plt.imshow(faceArea)
    # plt.show()
    backcolor = [0, 0, 0]

    countList = []
    center = [int(h/2), int(neck[1])]  # 顔領域の中心座標
    faceArea[center[0]][center[1]] = [255, 0, 0]
    countList.append(1)

    # quantity = int((w * h) * rate)  # 点の個数

    # 左上、右上、右下、左下の4点を中心座標から求めて、それに沿って点を取得していく
    upleft = [center[0] - 1, center[1] - 1]
    upright = [center[0] - 1, center[1] + 1]
    downright = [center[0] + 1, center[1] + 1]
    downleft = [center[0] + 1, center[1] - 1]
    while len(countList) < quantity:
        # 上の横線座標を登録
        for n in range(upleft[1], upright[1]):
            x = n
            y = upleft[0]
            if y >= h or x >= w:
                pass
            else:
                if rgb[y][x][0] == backcolor[0] and rgb[y][x][1] == backcolor[1] and rgb[y][x][2] == backcolor[2]:
                    pass
                else:
                    faceArea[y][x] = [255, 0, 0]
                    countList.append(1)
        # 右の縦線座標の登録
        for n in range(upright[0], downright[0]):
            x = upright[1]
            y = n
            if y >= h or x >= w:
                pass
            else:
                if rgb[y][x][0] == backcolor[0] and rgb[y][x][1] == backcolor[1] and rgb[y][x][2] == backcolor[2]:
                    pass
                else:
                    faceArea[y][x] = [255, 0, 0]
                    countList.append(1)
        # 下の横線座標の登録
        for n in range(downleft[1] + 1, downright[1] + 1):
            x = n
            y = downleft[0]
            if y >= h or x >= w:
                pass
            else:
                if rgb[y][x][0] == backcolor[0] and rgb[y][x][1] == backcolor[1] and rgb[y][x][2] == backcolor[2]:
                    pass
                else:
                    faceArea[y][x] = [255, 0, 0]
                    countList.append(1)
        # 左の縦線座標の登録
        for n in range(upleft[0] + 1, downleft[0] + 1):
            x = upleft[1]
            y = n
            if y >= h or x >= w:
                pass
            else:
                if rgb[y][x][0] == backcolor[0] and rgb[y][x][1] == backcolor[1] and rgb[y][x][2] == backcolor[2]:
                    pass
                else:
                    faceArea[y][x] = [255, 0, 0]
                    countList.append(1)
        # 左上、右上、右下、左下の4点を更新
        upleft = [upleft[0] - 1, upleft[1] - 1]
        upright = [upright[0] - 1, upright[1] + 1]
        downright = [downright[0] + 1, downright[1] + 1]
        downleft = [downleft[0] + 1, downleft[1] - 1]

    for y in range(h):
        for x in range(w):
            if faceArea[y][x][0] == 255 and faceArea[y][x][1] == 0 and faceArea[y][x][2] == 0:
                faceArea[y][x] = [255, 255, 255]
            else :
                faceArea[y][x] = [0, 0, 0]
                rgb[y][x] = [0, 0, 0]

    # plt.imshow(faceArea)
    # plt.show()

    return rgb, faceArea


# モルフォロジー変換によって、顔領域を一回り小さくする
def morphology(faceArea, mmg, para):
    h, w, ch = faceArea.shape
    img = cv2.cvtColor(faceArea, cv2.COLOR_BGR2GRAY)  # グレースケール
    th, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)  # 2値化

    # 両端を黒く塗りつぶすことで肌色検出時に首がはみ出しても大丈夫なようにする
    img[0, :] = 0
    img[h-1, :] = 0
    img[:, 0] = 0
    img[:, w-1] = 0

    '''
    # 顔領域の周りを黒くすることで、モロフォルジー処理をちゃんとできるようにする
    for y in range(h):
        for x in range(w):
            if x < 1 or x > w-1 or y < 1 or y > h-1:
                faceArea[y][x] = [0, 0, 0]
                mmg[y][x] = [0, 0, 0]
    '''

    # plt.imshow(mmg)
    # plt.show()

    kernel = np.ones((para, para), np.uint8)
    erosion = cv2.erode(img, kernel, iterations=1)

    for y in range(h):
        for x in range(w):
            if erosion[y][x] == 0 :
                mmg[y][x] = [0, 0, 0]
                faceArea[y][x] = [0, 0, 0]
            # if y >= (h-para) :   # 首を完全に除去することで、首のお化けの部分をなくす
            #     mmg[y][x] = [0, 0, 0]

    # plt.imshow(mmg)
    # plt.show()
    return faceArea, mmg


def connectPoints(mmg1, faceArea1, neck1, mmg2, faceArea2):  # 点群1の点数を少なくして、点群2より少ない点数を採用（手法：顔の中心から点を採用していく。ランダムではない）
    h1, w1, ch1 = mmg1.shape
    h2, w2, ch2 = mmg2.shape

    countPoint1 = 0
    countPoint2 = 0

    for y in range(h1) :
        for x in range(w1) :
            if faceArea1[y][x][0] == 255 and faceArea1[y][x][1] == 0 and faceArea1[y][x][2] == 0:
                pass
            else :
                countPoint1 = countPoint1 + 1
    for y in range(h2) :
        for x in range(w2) :
            if faceArea2[y][x][0] == 255 and faceArea2[y][x][1] == 0 and faceArea2[y][x][2] == 0:
                pass
            else :
                countPoint2 = countPoint2 + 1

    if countPoint2 >= countPoint1 :
        # rate = 1.0
        # quantity = int((w1 * h1) * rate)  # 点の個数
        # mmg1, faceArea1 = startpointCenter(faceArea1, neck1, quantity, mmg1)
        pass
    else :
        quantity = countPoint2 * 0.9   # 点の個数
        mmg1, faceArea1 = startpointCenter(faceArea1, neck1, quantity, mmg1)

    return mmg1, faceArea1



# 3Dモデルの生成
def crateModel(color_raw, depth_raw):
    # depth_scale=1.0, depth_trunc=50.0, convert_rgb_to_intensity=False で3Dモデルを描画したときに色がつく
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw,
        depth_scale=1.0, depth_trunc=50.0, convert_rgb_to_intensity=False)
    print(rgbd_image)

    plt.subplot(1, 2, 1)
    plt.title('Redwood grayscale image')
    plt.imshow(rgbd_image.color)
    plt.subplot(1, 2, 2)
    plt.title('Redwood depth image')
    plt.imshow(rgbd_image.depth)
    plt.show()

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

    return pcd


# RGBDを作成しないで点群を作成する
def view_pcd2(img, rgb, faceArea):  # カラー版 (0 < rgb < 1)
    arrayRGB = []
    h, w = img.shape
    for y in range(h):
        for x in range(w):
            pointColor = rgb[y][x]
            listColor = pointColor.tolist()
            pointColor = [n/255 for n in listColor]  # カラー版 (0 < rgb < 1)
            arrayRGB.append(pointColor)
    arrayRGB = np.array(arrayRGB)

    x = np.arange(w) - w // 2
    y = np.arange(h) - h // 2
    xx, yy = np.meshgrid(x, y)
    pp = np.dstack((xx / 200, yy / 200, img))
    xyz = pp.reshape((-1, 3))

    xyz, arrayRGB = deleteBackNotRGBD(xyz, arrayRGB)  # 背景削除
    # xyz = normalization(xyz)

    # 3Dモデルを作成する
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(arrayRGB)
    # o3d.visualization.draw_geometries([pcd]) # オブジェクトの描画

    return pcd


def deleteBackNotRGBD(xyz, arrayRGB):  # 背景削除（RGBDを使わないver）
    newimg = []
    newrgb = []

    for n in range(len(xyz)):
        if arrayRGB[n][0] == 0 and arrayRGB[n][1] == 0 and arrayRGB[n][2] == 0 :
            pass
        else :
            newimg.append(xyz[n])
            newrgb.append(arrayRGB[n])

    newimg = np.array(newimg)
    newrgb = np.array(newrgb)
    return newimg, newrgb


# 不要な顔領域の削除
def faceDelete(pcd)  :
    point_array = np.asarray(pcd.points)
    count, chan = point_array.shape  # countが点の個数

    colors3 = np.asarray(pcd.colors)  # 色情報を取得  数字の型はfloat64
    points3 = np.asarray(pcd.points)  # 座標情報を取得

    # 背景色の点を探す操作
    for ccc in range(len(colors3)):
        if colors3[ccc][0] == 0 and colors3[ccc][1] == 0 and colors3[ccc][2] == 0:
            colors3[ccc] = np.array([1, 0, 0])  # 色の変更（赤色）[1, 0, 0]
            points3[ccc] = np.array([-100, -100, -100])  # 座標の変更

    # 型変換してから代入 / py3d.Vector3dVector(colors3)の代わり
    pcd.colors = o3d.utility.Vector3dVector(colors3)
    pcd.points = o3d.utility.Vector3dVector(points3)

    # [[0.0001],[0.0001],[0.00001]]から[[50],[20],[35]]の間の点群を残し、それ以外を削除する
    bb = o3d.geometry.AxisAlignedBoundingBox(
        np.array([[-10], [-10], [-10]]),
        np.array([[50], [20], [35]]),
    )
    pcd = pcd.crop(bb)
    return pcd


# 3Dモデルの可視化
def watchModel(pcd):
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # 以下の5行は、o3d.visualization.draw_geometries([pcd], zoom=0.5)の代理
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    o3d.visualization.ViewControl.set_zoom(vis.get_view_control(), 0.8)
    vis.run()


def useNotihgRGBD(lmg, rmg, erosion, AfterAddress):  # RGBD使わないver
    mmg, ddpp, disparity = cereateDeaph(lmg, rmg, 270, 35)  # Depth画像の作成
    mmg, ddpp, faceArea, neck, scaleList = HeadDelete(ddpp, mmg, erosion, 6500)  # 首の削除  5500
    mmg, ddpp, faceArea, neck, mousePoint = earDelete(ddpp, AfterAddress, scaleList, neck, faceArea, mmg)  # 耳の削除
    mmg, faceArea = HeadAbolition(ddpp, mmg, faceArea , neck, mousePoint)  # 首の完全除去
    return mmg, ddpp, faceArea, neck, mousePoint

def adjustmentModels(mmg1, faceArea1, neck1, mmg2, mousePoint1, ddpp1, faceArea2) :  #「1」は対象となる点群（点少なめ） / 「2」はデータベースとなる点群（点多め）
    mmg1 = deleteOutlierNose(mmg1, faceArea1, ddpp1, mousePoint1)  # 鼻の外れ値（照明が反射しているため）を削除する
    faceArea1, mmg1 = morphology(faceArea1, mmg1, 60)  # 一回り小さい顔の取得（データベースの顔には適用しないこと）
    # mmg1, faceArea1 = connectPoints(mmg1, faceArea1, neck1, mmg2, faceArea2)

    return mmg1, ddpp1, faceArea1






# main






'''
# 入力画像
lmg1 = cv2.imread("stereo_camera/face_after/left/left1.jpg")
rmg1 = cv2.imread("stereo_camera/face_after/right/right1.jpg")
lmg1 = cv2.cvtColor(lmg1, cv2.COLOR_BGR2RGB)  # GBRからRGBに変換
rmg1 = cv2.cvtColor(rmg1, cv2.COLOR_BGR2RGB)  # GBRからRGBに変換
erosion1 = "stereo_camera/erosion/left1.jpg"
AfterAddress1 = "stereo_camera/face_after/left/left1.jpg"

lmg2 = cv2.imread("stereo_camera/face_after/left/left2.jpg")
rmg2 = cv2.imread("stereo_camera/face_after/right/right2.jpg")
lmg2 = cv2.cvtColor(lmg2, cv2.COLOR_BGR2RGB)  # GBRからRGBに変換
rmg2 = cv2.cvtColor(rmg2, cv2.COLOR_BGR2RGB)  # GBRからRGBに変換
erosion2 = "stereo_camera/erosion/left2.jpg"
AfterAddress2 = "stereo_camera/face_after/left/left2.jpg"


# pcd = useRGBD(lmg, rmg)  # RGBDを利用した3Dモデルの作成方法
mmg1, ddpp1, faceArea1, neck1, mousePoint1 = useNotihgRGBD(lmg1, rmg1, erosion1, AfterAddress1)  # RGBDを利用しないDepth画像作成・首の除去
mmg2, ddpp2, faceArea2, neck2, mousePoint2 = useNotihgRGBD(lmg2, rmg2, erosion2, AfterAddress2)  # RGBDを利用しないDepth画像作成・首の除去

mmg1, ddpp1, faceArea1 = adjustmentModels(mmg1, faceArea1, neck1, mmg2, mousePoint1, ddpp1, faceArea2)

pcd1 = view_pcd2(ddpp1, mmg1, faceArea1)  # 3Dモデルの作成（背景を削除するのも含まれる）
pcd2 = view_pcd2(ddpp2, mmg2, faceArea2)  # 3Dモデルの作成（背景を削除するのも含まれる）

# watchModel(pcd)  # 3Dモデルの表示
# 「1」は対象となる点群（点少なめ） / 「2」はデータベースとなる点群（点多め）
o3d.io.write_point_cloud("stereo_camera/3Dmodel/model_1.ply", pcd1)  # 3Dモデルの保存
o3d.io.write_point_cloud("stereo_camera/3Dmodel/model_2.ply", pcd2)  # 3Dモデルの保存
'''