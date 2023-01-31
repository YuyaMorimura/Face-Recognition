import cv2
import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d
import time
import treg
import copy
import math
# import py3d
from PIL import Image

def draw_registration_result(source, target, transformation):   # 片方の点群を初期位置に移動させて、可視化する
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # source_temp.paint_uniform_color([1, 0.706, 0])   # 点群の色を一括で変えて、重なり具合を視覚で確認しやすいようにする
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)  # 変換行列によるRGBD画像の変換
    # o3d.visualization.draw_geometries([source_temp, target_temp])  # 読み込んだ3Dモデルの可視化
    return source_temp, target_temp


def startRevolution():
    # RGBD使うとき：np.asarray([[1.0, 0, 0, 0.1],
    #                         [0, 1.0, 0, 0.05],
    #                         [0, 0, 1.0, 0.0],
    #                          [0, 0, 0, 1.0]])
    # 使わないとき:
    # Rt = np.asarray([[1.0, 0, 0, 0.8],
    #                        [0, 1.0, 0, 1.0],
    #                         [0, 0, 1.0, 1.0],
    #                          [0, 0, 0, 1.0]])

    # 回転行列をもとめる
    # ram = [[math.radians(random.randint(0, 30)), math.radians(random.randint(0, 30)),
    #         math.radians(random.randint(0, 30))]]   # 0~30度に回転
    ram = [[math.radians(0), math.radians(0), math.radians(0)]]
    px = ram[0][0]
    py = ram[0][1]
    pz = ram[0][2]
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(px), np.sin(px)],
                   [0, -np.sin(px), np.cos(px)]])
    Ry = np.array([[np.cos(py), 0, -np.sin(py)],
                   [0, 1, 0],
                   [np.sin(py), 0, np.cos(py)]])
    Rz = np.array([[np.cos(pz), np.sin(pz), 0],
                   [-np.sin(pz), np.cos(pz), 0],
                   [0, 0, 1]])
    R = Rz.dot(Ry).dot(Rx)  # 回転行列

    # t = np.random.rand(3, 1) * 10.0  # 並行移動行列
    t = [[0.0], [0.40], [0.10]]  # 従来の値: [[0.0], [0.40], [0.10]]  目の領域での値：[[0.0], [0.15], [0.10]]（眼球から推定）

    Rt = np.asarray([[R[0][0], R[0][1], R[0][2], t[0][0]],
                        [R[1][0], R[1][1], R[1][2], t[1][0]],
                        [R[2][0], R[2][1], R[2][2], t[2][0]],
                        [0, 0, 0, 1.0]])
    print(Rt)
    return Rt




def searchCorresponding(reg_p2p, aftersource, aftertarget):  # 3Dモデル間の距離・対応点同士の色情報の取得
    pointset = np.asarray(reg_p2p.correspondence_set)  # 対応点のセット  公式ページの検索欄から詳細を検索すること

    apoint = np.asarray(aftersource.points)  # 点群の座標
    bpoint = np.asarray(aftertarget.points)  # 点群の座標
    acolor =  np.asarray(aftersource.points)  # 点群の色情報
    bcolor =  np.asarray(aftertarget.points)  # 点群の色情報

    listnorm = []
    for set in pointset :
        dd = []

        # 3Dモデル間の距離を求める
        aa = apoint[set[0]]
        bb = bpoint[set[1]]
        norm = np.linalg.norm(aa-bb)   # ユークリッド距離
        # listnorm.append(norm)
        # 対応点の色情報を求める
        aacolor = acolor[set[0]].tolist()
        bbcolor = bcolor[set[1]].tolist()

        dd.append(norm)
        dd.append(aacolor)
        dd.append(bbcolor)
        listnorm.append(dd)
    listnorm.sort(reverse=True)  # ユークリッド距離が大きい順にソートする
    return listnorm



def certificationColor(listnorm) :   # 色情報による認証

    '''
    # 対応点どうしの距離が大きい順で、上位n番目までのRGB情報の差を2乗して、その和
    listnorm.sort(reverse=False)  # ユークリッド距離が小さい順にソートする

    n = 500
    sumColor = 0
    for count in range(n):
        colorA = np.array(listnorm[count][1])  # 色情報
        colorB = np.array(listnorm[count][2])

        increment = colorA - colorB
        for m in range(3):
            sumColor = sumColor + increment[m]**2

    listnorm.sort(reverse=True)  # ユークリッド距離が大きい順にソートする
    '''


    # 全てのRGB値の差の2乗をとり、その平均を色情報判定だとする  MSEみたいな感じだよね　https://mathwords.net/rmsemae
    n = len(listnorm)
    sumColor = 0
    for count in range(n-1):
        colorA = np.array(listnorm[count][1])  # 色情報
        colorB = np.array(listnorm[count][2])

        increment = colorA - colorB
        for m in range(3):
            sumColor = sumColor + increment[m] ** 2
    sumColor = sumColor / (n-1)

    return sumColor


def certifica(listnorm) :   # 色情報による認証

    # 対応点どうしの距離が大きい順で、上位n番目までのRGB情報の差を2乗して、その和
    # listnorm.sort(reverse=False)  # ユークリッド距離が小さい順にソートする

    n = 500
    sumColor = 0
    for count in range(n):
        colorA = np.array(listnorm[count][1])  # 色情報
        colorB = np.array(listnorm[count][2])

        increment = colorA - colorB
        for m in range(3):
            sumColor = sumColor + increment[m]**2

    # listnorm.sort(reverse=True)  # ユークリッド距離が大きい順にソートする

    return sumColor

def doICP(source, target, threshold, trans_init):
    beforesource, beforetarget = draw_registration_result(source, target, trans_init)  # 位置合わせ処理中のアライメントを可視化
    # o3d.visualization.draw_geometries([beforetarget, beforesource])
    print("Initial alignment")
    evaluation = o3d.pipelines.registration.evaluate_registration(source, target,
                                                                  threshold, trans_init)
    print(evaluation)

    print("Apply point-to-point ICP")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000))  # max_iterationは反復回数
    print(reg_p2p)
    print("Correspondence set between source and target point cloud.")
    print("Transformation is:")
    print(reg_p2p.transformation)
    print("")
    aftersource, aftertarget = draw_registration_result(source, target, reg_p2p.transformation)

    listnorm = searchCorresponding(reg_p2p, aftersource, aftertarget)  # 各対応点同士の距離・色情報の取得
    sumColor = certificationColor(listnorm)
    sumColor22 = certifica(listnorm)

    print("モデル間の距離とその色情報Top5：", listnorm[0:5])
    print("モデル間の距離：", listnorm[0][0])
    print("色情報判定(全部の平均)：", sumColor)
    print("RMSE: ", reg_p2p.inlier_rmse)
    print("色情報判定(TOP5〇〇番目)：", sumColor22)

    return aftersource, aftertarget, listnorm[0][0], sumColor, reg_p2p.inlier_rmse, sumColor22




#　ERRの評価方法


def line_cross_point(P0, P1, Q0, Q1):
    x0, y0 = P0; x1, y1 = P1
    x2, y2 = Q0; x3, y3 = Q1
    a0 = x1 - x0; b0 = y1 - y0
    a2 = x3 - x2; b2 = y3 - y2

    d = a0*b2 - a2*b0
    if d == 0:
        # two lines are parallel
        return None

    # s = sn/d
    sn = b2 * (x2-x0) - a2 * (y2-y0)
    # t = tn/d
    #tn = b0 * (x2-x0) - a0 * (y2-y0)
    return x0 + a0*sn/d, y0 + b0*sn/d


def evaluation(list, nn, mm, couA, oneA, startA, couB, oneB, startB) :  # ERRを求め、評価する   (nn は色情報の番号・mmは距離の番号)

    # 3Dモデル間の距離のみ適用
    start = startA  # 最初の閾値を定義
    listAll = []

    for aa in range(couA) :
        FARCount = 0  # 「他人受入」の個数
        FRRCount = 0  # 「本人拒否」の個数

        for cc in range(len(list)) :  # 認証対象の顔の数（１人につき）
            for base in range(len(list[0])) :  # データベースの顔番号

                for target in range(len(list[0][0])) :  # 認証対象の顔番号
                    if list[cc][base][target][mm] <= start:  # 閾値以下なら、同一人物だとみなす
                        if base != target: # 「他人受入」の組み合わせか定義
                            FARCount = FARCount + 1
                    if base == target :  # 同一人物の組み合わせをピックアップ
                        if list[cc][base][target][mm] <= start :  # 閾値以下なら、同一人物だとみなす
                            pass
                        else:  # 「本人拒否」の場合の処理
                            FRRCount = FRRCount + 1

        FARCount = FARCount / ((len(list) * len(list[0]) * len(list[0][0])) - (len(list) * len(list[0])))  # 確率に直す
        FRRCount = FRRCount / (len(list) * len(list[0]))  # 確率に直す

        # print("FARの母数", (len(list) * len(list[0]) * len(list[0][0])) - len(list))
        # print("FRRの母数", len(list) * len(list[0]))

        # 座標に重複がないように処理
        abc = 0
        if len(listAll) == 0:
            listAll.append([FRRCount, FARCount])
        else :
            for ff in listAll :
                if ff[0] == FRRCount and ff[1] == FARCount :
                    abc = abc + 1
            if abc == 0:
                listAll.append([FRRCount, FARCount])

        start = start + oneA

    FAR_Y = []
    FRR_X = []
    sordToX = sorted(listAll)  # x座標でソートする
    # sordToX = sorted(listAll, key=lambda x: x[1])  # y座標でソートする

    for aaaa in sordToX :
        FRR_X.append(aaaa[0])
        FAR_Y.append(aaaa[1])

    plt.plot(FRR_X, FAR_Y, color="red", label="distance")  # グラフの描画

    # y = x のグラフの描画 (xはFRR_Xと同じ値になるように定義するため、xを新しく定義しない)
    y = []
    for nnn in FRR_X:
        y.append(nnn)
    plt.plot(FRR_X, y, color="black")

    y = np.array(y)
    FRR_X = np.array(FRR_X)
    FAR_Y = np.array(FAR_Y)

    # グラフの交点を求める  https://www.blue-weblog.com/entry/2017/12/24/194413
    idx = np.argwhere(np.diff(np.sign(FAR_Y - y)) != 0)
    xxx, yyy = line_cross_point((FRR_X[idx[0]], FAR_Y[idx[0]]), (FRR_X[idx[0] + 1], FAR_Y[idx[0] + 1]),
                                (0, 0), (10, 10))

    plt.plot(xxx, yyy, 'ms', ms=5)
    print("3Dモデル間のみでのERR：", xxx, yyy)
    # plt.plot(FRR_X[idx[0]], FAR_Y[idx[0]], 'ms', ms=5)
    # print("3Dモデル間のみでのERR：", FRR_X[idx[0]], FAR_Y[idx[0]])



    # 色情報も考慮している場合のみ適用



    start = startA  # 最初の閾値を定義（距離）
    listAll = []

    for aa in range(couA):
        coStart = startB  # 最初の閾値を定義（色情報）
        for bb in range(couB) :

            FARCount = 0  # 「他人受入」の個数
            FRRCount = 0  # 「本人拒否」の個数

            for cc in range(len(list)):  # 認証対象の顔の数（１人につき）
                for base in range(len(list[0])):  # データベースの顔番号

                    for target in range(len(list[0][0])):  # 認証対象の顔番号
                        if list[cc][base][target][mm] <= start and list[cc][base][target][nn] <= coStart:  # 閾値以下なら、同一人物だとみなす
                            if base != target:  # 「他人受入」の組み合わせか定義
                                FARCount = FARCount + 1
                        if base == target:  # 同一人物の組み合わせをピックアップ
                            if list[cc][base][target][mm] <= start and list[cc][base][target][nn] <= coStart:  # 閾値以下なら、同一人物だとみなす
                                pass
                            else:  # 「本人拒否」の場合の処理
                                FRRCount = FRRCount + 1

            FARCount = FARCount / ((len(list) * len(list[0]) * len(list[0][0])) - (len(list) * len(list[0])))  # 確率に直す
            FRRCount = FRRCount / (len(list) * len(list[0]))  # 確率に直す

            # 座標に重複がないように処理
            abc = 0
            if len(listAll) == 0:
                listAll.append([FRRCount, FARCount])
            else :
                for ff in listAll:
                    if ff[0] == FRRCount and ff[1] == FARCount:
                        abc = abc + 1
                if abc == 0:
                    listAll.append([FRRCount, FARCount])

            # 全ての点の平均を適用：0.0001   距離が近い順で1000点対応点のみ適用：0.0001
            coStart = coStart + oneB
        start = start + oneA

    FAR_Y = []
    FRR_X = []
    sordToX = sorted(listAll)  # x座標でソートする
    # sordToX = sorted(listAll, key=lambda x: x[1])  # y座標でソートする

    for aaaa in sordToX:
        FRR_X.append(aaaa[0])
        FAR_Y.append(aaaa[1])

    plt.plot(FRR_X, FAR_Y, color="blue", label="distance and color")  # グラフの描画

    # y = x のグラフの描画 (xはFRR_Xと同じ値になるように定義するため、xを新しく定義しない)
    y = []
    for nnn in FRR_X:
        y.append(nnn)
    plt.plot(FRR_X, y, color="black")

    y = np.array(y)
    FRR_X = np.array(FRR_X)
    FAR_Y = np.array(FAR_Y)

    # グラフの交点を求める  https://www.blue-weblog.com/entry/2017/12/24/194413
    idx = np.argwhere(np.diff(np.sign(FAR_Y - y)) != 0)
    xxx, yyy = line_cross_point((FRR_X[idx[0]], FAR_Y[idx[0]]), (FRR_X[idx[0] + 1], FAR_Y[idx[0] + 1]),
                                (0, 0), (10, 10))

    plt.plot(xxx, yyy, 'ms', ms=5)
    print("色情報を考慮した場合でのERR：", xxx, yyy)
    # plt.plot(FRR_X[idx[0]], FAR_Y[idx[0]], 'ms', ms=5)
    # print("色情報を考慮した場合でのERR：", FRR_X[idx[0]], FAR_Y[idx[0]])

    plt.xlabel("False Rejection Rate [%]")
    plt.ylabel("False Acceptance Rate [%]")

    plt.xlim(0, 3)
    plt.ylim(0, 3)
    plt.legend()
    plt.show()  # プロットを表示



def evaluationALL(list, nn, mm, couA, oneA, startA, couB, oneB, startB) :  # ERRを求め、評価する   (nn は色情報の番号・mmは距離の番号)

    # 3Dモデル間の距離のみ適用
    start = startA  # 最初の閾値を定義
    listAll = []

    for aa in range(couA) :
        FARCount = 0  # 「他人受入」の個数
        FRRCount = 0  # 「本人拒否」の個数

        for cc in range(len(list)) :  # 認証対象の顔の数（１人につき）
            for base in range(len(list[0])) :  # データベースの顔番号

                for target in range(len(list[0][0])) :  # 認証対象の顔番号
                    if list[cc][base][target][mm] <= start:  # 閾値以下なら、同一人物だとみなす
                        if base != target: # 「他人受入」の組み合わせか定義
                            FARCount = FARCount + 1
                    if base == target :  # 同一人物の組み合わせをピックアップ
                        if list[cc][base][target][mm] <= start :  # 閾値以下なら、同一人物だとみなす
                            pass
                        else:  # 「本人拒否」の場合の処理
                            FRRCount = FRRCount + 1

        FARCount = FARCount / ((len(list) * len(list[0]) * len(list[0][0])) - (len(list) * len(list[0])))  # 確率に直す
        FRRCount = FRRCount / (len(list) * len(list[0]))  # 確率に直す

        # print("FARの母数", (len(list) * len(list[0]) * len(list[0][0])) - len(list))
        # print("FRRの母数", len(list) * len(list[0]))

        # 座標に重複がないように処理
        abc = 0
        if len(listAll) == 0:
            listAll.append([FRRCount, FARCount])
        else :
            for ff in listAll :
                if ff[0] == FRRCount and ff[1] == FARCount :
                    abc = abc + 1
            if abc == 0:
                listAll.append([FRRCount, FARCount])

        start = start + oneA

    FAR_Y = []
    FRR_X = []
    sordToX = sorted(listAll)  # x座標でソートする
    # sordToX = sorted(listAll, key=lambda x: x[1])  # y座標でソートする

    for aaaa in sordToX :
        FRR_X.append(aaaa[0])
        FAR_Y.append(aaaa[1])

    plt.plot(FRR_X, FAR_Y, color="red", label="distance")  # グラフの描画

    # y = x のグラフの描画 (xはFRR_Xと同じ値になるように定義するため、xを新しく定義しない)
    y = []
    for nnn in FRR_X:
        y.append(nnn)
    plt.plot(FRR_X, y, color="black")

    y = np.array(y)
    FRR_X = np.array(FRR_X)
    FAR_Y = np.array(FAR_Y)

    # グラフの交点を求める  https://www.blue-weblog.com/entry/2017/12/24/194413
    idx = np.argwhere(np.diff(np.sign(FAR_Y - y)) != 0)
    xxx, yyy = line_cross_point((FRR_X[idx[0]], FAR_Y[idx[0]]), (FRR_X[idx[0] + 1], FAR_Y[idx[0] + 1]),
                                (0, 0), (10, 10))

    plt.plot(xxx, yyy, 'ms', ms=5)
    print("3Dモデル間のみでのERR：", xxx, yyy)
    # plt.plot(FRR_X[idx[0]], FAR_Y[idx[0]], 'ms', ms=5)
    # print("3Dモデル間のみでのERR：", FRR_X[idx[0]], FAR_Y[idx[0]])



    # 色情報も考慮している場合のみ適用



    start = startA  # 最初の閾値を定義（距離）
    listAll = []

    coStart = startB  # 最初の閾値を定義（色情報）
    for bb in range(couB) :

        FARCount = 0  # 「他人受入」の個数
        FRRCount = 0  # 「本人拒否」の個数

        for cc in range(len(list)):  # 認証対象の顔の数（１人につき）
            for base in range(len(list[0])):  # データベースの顔番号

                for target in range(len(list[0][0])):  # 認証対象の顔番号
                    if list[cc][base][target][mm] <= start and list[cc][base][target][nn] <= coStart:  # 閾値以下なら、同一人物だとみなす
                        if base != target:  # 「他人受入」の組み合わせか定義
                            FARCount = FARCount + 1
                    if base == target:  # 同一人物の組み合わせをピックアップ
                        if list[cc][base][target][mm] <= start and list[cc][base][target][nn] <= coStart:  # 閾値以下なら、同一人物だとみなす
                            pass
                        else:  # 「本人拒否」の場合の処理
                            FRRCount = FRRCount + 1

        FARCount = FARCount / ((len(list) * len(list[0]) * len(list[0][0])) - (len(list) * len(list[0])))  # 確率に直す
        FRRCount = FRRCount / (len(list) * len(list[0]))  # 確率に直す

        # 座標に重複がないように処理
        abc = 0
        if len(listAll) == 0:
            listAll.append([FRRCount, FARCount])
        else :
            for ff in listAll:
                if ff[0] == FRRCount and ff[1] == FARCount:
                    abc = abc + 1
            if abc == 0:
                listAll.append([FRRCount, FARCount])

        # 全ての点の平均を適用：0.0001   距離が近い順で1000点対応点のみ適用：0.0001
        coStart = coStart + oneB
        start = start + oneA

    FAR_Y = []
    FRR_X = []
    sordToX = sorted(listAll)  # x座標でソートする
    # sordToX = sorted(listAll, key=lambda x: x[1])  # y座標でソートする

    for aaaa in sordToX:
        FRR_X.append(aaaa[0])
        FAR_Y.append(aaaa[1])

    plt.plot(FRR_X, FAR_Y, color="blue", label="distance and color")  # グラフの描画

    # y = x のグラフの描画 (xはFRR_Xと同じ値になるように定義するため、xを新しく定義しない)
    y = []
    for nnn in FRR_X:
        y.append(nnn)
    plt.plot(FRR_X, y, color="black")

    y = np.array(y)
    FRR_X = np.array(FRR_X)
    FAR_Y = np.array(FAR_Y)

    # グラフの交点を求める  https://www.blue-weblog.com/entry/2017/12/24/194413
    idx = np.argwhere(np.diff(np.sign(FAR_Y - y)) != 0)
    xxx, yyy = line_cross_point((FRR_X[idx[0]], FAR_Y[idx[0]]), (FRR_X[idx[0] + 1], FAR_Y[idx[0] + 1]),
                                (0, 0), (10, 10))

    plt.plot(xxx, yyy, 'ms', ms=5)
    print("色情報を考慮した場合でのERR：", xxx, yyy)
    # plt.plot(FRR_X[idx[0]], FAR_Y[idx[0]], 'ms', ms=5)
    # print("色情報を考慮した場合でのERR：", FRR_X[idx[0]], FAR_Y[idx[0]])

    plt.xlabel("False Rejection Rate [%]")
    plt.ylabel("False Acceptance Rate [%]")

    plt.xlim(0, 2)
    plt.ylim(0, 2)
    plt.legend()
    plt.show()  # プロットを表示

# main




'''
source = o3d.io.read_point_cloud("stereo_camera/3Dmodel/model_1.ply")  # 対象となる点群（数すくなめ）
target = o3d.io.read_point_cloud("stereo_camera/3Dmodel/model_2.ply")  # データベースの点群（数おおめ）

# demo_colored_icp_pcds = o3d.data.DemoColoredICPPointClouds()
# source = o3d.io.read_point_cloud(demo_colored_icp_pcds.paths[0])
# target = o3d.io.read_point_cloud(demo_colored_icp_pcds.paths[1])

# 2つのRGBD画像の初期位置はほぼ同じ位置（加減はネットのICPの実行結果を参照）にして、閾値をなるべく小さくする
# 顔領域以外を1つにまとめると位置合わせが上手く行かなかった。。。。
# 初期位置の行列で3Dモデル間の距離を広げると、閾値も大きくしないと死ぬ

time_sta = time.time()  # ICPの実行時間を計測する

threshold = 10.0  # 閾値    RGBD使うとき：0.1 使わないとき:
trans_init = startRevolution()  # 初期位置の剛体行列の作成
source, target, distance, color = doICP(source, target, threshold, trans_init)


# threshold = 100.0  # 閾値
# trans_init = np.asarray([[1, 0, 0, 0],
#                         [0, 1, 0, 0],
#                         [0, 0, 1, 0],
#                         [0, 0, 0, 1.0]])
# source, target = doICP(source, target, threshold, trans_init)

time_end = time.time()
tim = time_end - time_sta
print("実行時間：", tim, "秒")

print("end played")
'''

