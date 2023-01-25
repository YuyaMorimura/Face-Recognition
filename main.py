import cv2
import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d

from university import ImagesToMovie
from university import rectification
from university import SkinToneDetevtion
from university import Disparity
from university import ICP
from university import faceParts


faceNumber = 7   # 対象の顔を１人につき何枚の画像に分けるか  デフォルト：7
count = 12  # 顔の数
solution = []  # 全ての結果

for cc in range(2, faceNumber + 1) :  #  対象の顔を１人につき何枚の画像を認証するか

    allDetabace = []  # 全てのデータベースでの認証結果
    for y in range(1, count + 1) :  # データベースの顔数
        oneDetabace = []  # 一つのデータベースでの結果を記録
        for x in range(1, count + 1) :  # 対象の顔数

            print("ループ回数：", cc)
            print("データベースの顔番号：", y)
            print("認証対象の顔番号：", x)

            ImagesToMovie.targetMovieToImage('stereo_camera/original/subjectImage/{}.mp4'.format(str(x)),
                            'stereo_camera/face_images', 1, cc)  # 対象の顔の保存
            ImagesToMovie.save_all_frames('stereo_camera/original/detabaseImage/{}.mp4'.format(str(y)),
                            'stereo_camera/face_images', 2)  # データベースの顔の保存

            rectification.rectification()  # 画像の平行化

            #  SkinToneDetevtion
            # 画像データの読み込み
            lmg1 = cv2.imread("stereo_camera/face_after/left/left1.jpg")
            lmg2 = cv2.imread("stereo_camera/face_after/left/left2.jpg")

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

            SkinToneDetevtion.makeFace(lmg1, l_adress1, erosion1, lmg11, rmg11)  # mainの実行関数
            SkinToneDetevtion.makeFace(lmg2, l_adress2, erosion2, lmg22, rmg22)  # mainの実行関数

            # Dispatity
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
            mmg1, ddpp1, faceArea1, neck1, mousePoint1 = Disparity.useNotihgRGBD(lmg1, rmg1, erosion1,
                                                                       AfterAddress1)  # RGBDを利用しないDepth画像作成・首の除去
            mmg2, ddpp2, faceArea2, neck2, mousePoint2 = Disparity.useNotihgRGBD(lmg2, rmg2, erosion2,
                                                                       AfterAddress2)  # RGBDを利用しないDepth画像作成・首の除去

            mmg1, ddpp1, faceArea1 = Disparity.adjustmentModels(mmg1, faceArea1, neck1, mmg2, mousePoint1, ddpp1, faceArea2)

            # 顔のパーツごとの認証コーナー
            # mmg1 = faceParts.mouse(mmg1, mousePoint1)  # 口の領域の抽出
            # mmg1 = faceParts.nose(mmg1, mousePoint1)  # 鼻の領域の抽出
            mmg1 = faceParts.eyes(mmg1, faceArea1, mousePoint1)  # 目の領域の抽出

            pcd1 = Disparity.view_pcd2(ddpp1, mmg1, faceArea1)  # 3Dモデルの作成（背景を削除するのも含まれる）
            pcd2 = Disparity.view_pcd2(ddpp2, mmg2, faceArea2)  # 3Dモデルの作成（背景を削除するのも含まれる）

            # watchModel(pcd)  # 3Dモデルの表示
            # 「1」は対象となる点群（点少なめ） / 「2」はデータベースとなる点群（点多め）
            o3d.io.write_point_cloud("stereo_camera/3Dmodel/model_1.ply", pcd1)  # 3Dモデルの保存
            o3d.io.write_point_cloud("stereo_camera/3Dmodel/model_2.ply", pcd2)  # 3Dモデルの保存

            # ICP
            source = o3d.io.read_point_cloud("stereo_camera/3Dmodel/model_1.ply")  # 対象となる点群（数すくなめ）
            target = o3d.io.read_point_cloud("stereo_camera/3Dmodel/model_2.ply")  # データベースの点群（数おおめ）

            threshold = 10.0  # 閾値    RGBD使うとき：0.1 使わないとき:
            trans_init = ICP.startRevolution()  # 初期位置の剛体行列の作成
            source, target, distance, color, RMSE, color22 = ICP.doICP(source, target, threshold, trans_init)

            oneDetabace.append([distance, color, RMSE, color22])  # distance = 3Dモデル間の距離, color = 色判定

        allDetabace.append(oneDetabace)  # デバックして確認すること
    solution.append(allDetabace)

# print("RMSEと全体の平均での判定")
# ICP.evaluation(solution, 1, 2, 1000, 0.001, 0, 1000, 0.00001, 0)  # 評価する
# print("RMSEとTop〇〇番目")
# ICP.evaluation(solution, 3, 2, 1000, 0.001, 0, 1000, 0.01, 0)  # 評価する


# print("距離が一番遠いと全体の平均")
# ICP.evaluation(solution, 1, 0, 1000, 0.001, 0, 1000, 0.00001, 0)  # 評価する
# print("距離が一番遠いとTop〇〇番目")
# ICP.evaluation(solution, 3, 0, 1000, 0.001, 0, 1000, 0.01, 0)  # 評価する


print("距離が一番遠いとTop〇〇番目（色情報を含めた場合は、1度に両方の閾値を変更）")
ICP.evaluationALL(solution, 3, 0, 10000, 0.001, 0, 10000, 0.01, 0)  # 評価する
print("距離が一番遠いと全体の平均(色情報を含めた場合は、1度に両方の閾値を変更)")
ICP.evaluationALL(solution, 1, 0, 10000, 0.001, 0, 10000, 0.00001, 0)  # 評価する

solution = np.array(solution)  # デバックしたときにみやすくするため
print("finish")







