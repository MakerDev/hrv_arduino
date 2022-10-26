import os
import argparse

import cv2
print(cv2.__version__)


def extractImages(source_path, dest_path):
    count = 0
    vidcap = cv2.VideoCapture(source_path)
    success, image = vidcap.read()
    success = True
    while success:
        success, image = vidcap.read()
        path = os.path.join(dest_path, f'frame{count}.png')
        # cv2.imwrite(path, image)
        #경로 한글 문제
        extension = '.png'
        result, encoded_img = cv2.imencode(extension, image)
 
        if result:
            with open(path, mode='w+b') as f:
                encoded_img.tofile(f)
        count += 1

#TODO: 박수 프레임 기준, ppg timestamp에 맞게 파일 이름 바꿔주기. 그건 별도의 py로 만들어야할 듯.
#박수 프레임
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", default='D:\AESPA Dataset\webcams\WIN_20220824_21_22_27_Pro_홍요한.mp4', help="path to video")
    #parser.add_argument("--dest_path", help="path to images")
    parser.add_argument("--fps", default=30, help="Video fps")

    args = parser.parse_args()
    print(args)
    dest_path = args.source_path.replace(".mp4", "_webcam")

    if not os.path.exists(dest_path):
        os.mkdir(dest_path)

    extractImages(args.source_path, dest_path)
