#　ステレオカメラで撮影した動画から、左右のそれぞれの画像を作成


import cv2
import os

#  動画パス、保存する画像の場所、画像名前
def save_all_frames(video_path, dir_path, n, ext='jpg'):    # データベースのみの処理

    os.makedirs(dir_path, exist_ok=True)  # ディレクトリの作成
    dir_path_left = '{}/{}'.format(dir_path, 'left')
    dir_path_right = '{}/{}'.format(dir_path, 'right')
    os.makedirs(dir_path_left, exist_ok=True)
    os.makedirs(dir_path_right, exist_ok=True)

    base_path_right = os.path.join(dir_path_right, "right")  # パス文字列の結合
    base_path_left = os.path.join(dir_path_left, "left")

    cap = cv2.VideoCapture(video_path)  # 動画の読み込み
    if not cap.isOpened():
        return

    totalframecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 動画の合計フレーム数

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(totalframecount/2))  # 動画の中間地点のフレーム情報を読み込む

    ret, frame = cap.read()  # 画像情報があるかないか、画像情報（行列）
    height, width, channels = frame.shape  # 画像の寸法・チャンネル数の取得
    frame_right = frame[:, width//2:width]
    frame_left = frame[:, 0:width//2]
    cv2.imwrite('{}{}.{}'.format(base_path_right, str(n).zfill(1), ext), frame_right)
    cv2.imwrite('{}{}.{}'.format(base_path_left, str(n).zfill(1), ext), frame_left)



def targetMovieToImage(video_path, dir_path, n, cc, ext='jpg'):  # 認証対象のみの処理
    os.makedirs(dir_path, exist_ok=True)  # ディレクトリの作成
    dir_path_left = '{}/{}'.format(dir_path, 'left')
    dir_path_right = '{}/{}'.format(dir_path, 'right')
    os.makedirs(dir_path_left, exist_ok=True)
    os.makedirs(dir_path_right, exist_ok=True)

    base_path_right = os.path.join(dir_path_right, "right")  # パス文字列の結合
    base_path_left = os.path.join(dir_path_left, "left")

    cap = cv2.VideoCapture(video_path)  # 動画の読み込み
    if not cap.isOpened():
        return

    totalframecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 動画の合計フレーム数

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cc * 10))  # cc * 10のフレームを読み込み

    ret, frame = cap.read()  # 画像情報があるかないか、画像情報（行列）
    height, width, channels = frame.shape  # 画像の寸法・チャンネル数の取得
    frame_right = frame[:, width // 2:width]
    frame_left = frame[:, 0:width // 2]
    cv2.imwrite('{}{}.{}'.format(base_path_right, str(n).zfill(1), ext), frame_right)
    cv2.imwrite('{}{}.{}'.format(base_path_left, str(n).zfill(1), ext), frame_left)

'''
path, dirs, files = next(os.walk("stereo_camera/original/on_face"))  # 入力動画があるディレクトリ
file_count = len(files)  # フォルダ内のファイル数
# for number in range(1, file_count):
for number in range(1, 3):
    save_all_frames('stereo_camera/original/on_face/onface_{}.mp4'.format(str(number)),
                    'stereo_camera/face_images', number)  # 入力する動画アドレス、画像を保存するディレクトリ
'''

