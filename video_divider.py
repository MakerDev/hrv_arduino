from glob import glob
import os
import numpy as np
import csv
import random
import math
import utils
import datetime
from moviepy.editor import *



def extract_second(timestamp:str):
    '''
    18:27:43.968400 형태의 timestamp에서 초를 추출함
    '''
    return int(timestamp.split(':')[-1].split('.')[0])


def tick_second(timestamp_str, dateformat="%H:%M:%S.%f"):
    '''
    현재 timestampe에서 1초 증가한 timestamp문자열 얻기
    '''
    datetime_convert = datetime.datetime.strptime(timestamp_str, dateformat)
    next_second = datetime_convert + datetime.timedelta(seconds=1)

    return datetime.datetime.strftime(next_second, dateformat)

def step_time(start:float, step_amount:float, fps: int):
    result = start + step_amount
    frames = int(result*100)% 100

    seconds = int(result) + frames // fps 
    frames = frames % fps

    return seconds + frames * 0.01


'''
csv파일과 첫 clip의 시작 row 넘버를 입력받아 predefine된 영상의 정보를 이용하여, csv를 잘라낸다.
'''
if __name__ == "__main__":
    folder = 'D://AESPA Dataset/webcams'
    output_folder = 'D://AESPA Dataset/webcams_splitted'

    videos = os.listdir(folder)

    #Clip1 시작 시간
    start_times = {
        "홍요한": 19,
        "정윤하": 25,
        "손혜림": 33,
        "윤보현": 30,
        "이영준": 18,
        "문서현": 17,
        "김정우": 35,
        "suan": 12,        
        "김유겸": 22,
        "이준서": 13,
        "박정민": 11,
        "김도완": 16,
        "채지원": 14,
        "김지현": 7,
        "김도운": 17,
        "김윤태": 12,
        "조은정": 17,
        "이기창": 13,
        "한보람": 19,
        "김기현": 11,
    }
    
    clip_lens = [201.06, 148.0, 54.11, 208.14, 96.18, 142.20,
                 111.23, 131.12, 106.06, 64.22, 126.20, 128.09, 151.09, 40.06]
    clip_emotion_label = ['neutral1', 'fear1', 'suprise1', 'sad1', 'disgust1', 'fear2',
                          'anger1', 'happy1', 'neutral2', 'disgust2', 'anger2', 'happy2', 'sad2', 'suprise2']

    #clip label 2부터 14까지 존재
    clip_label_lens     = [3.24, 3.04, 4.0, 3.24, 3.24, 3.23, 3.24, 3.24, 3.24, 3.24, 3.24, 3.24, 4.0]
    #쉬어가기 1~13까지 존재
    cooldown_lable_lens = [2.15, 2.19, 2.19, 2.19, 2.14, 2.18, 2.20, 2.17, 2.19, 2.20, 2.19, 2.19, 2.19]

    fs_video = 25

    for filename in videos:
        subject_name = filename.split('_')[-1].split('.')[0]

        if subject_name not in start_times:
            print(f"Skipped {filename}")
            continue

        start_time = start_times[subject_name]
        file_path = os.path.join(folder, filename)
        clip = VideoFileClip(file_path)

        #중간에 길이 모자란 웹캠 영상 처리용도
        try:
            for i, clip_len in enumerate(clip_lens):
                #프레임단위는 그냥 올리기.
                subclip = clip.subclip(start_time, start_time + math.ceil(clip_len))
                output_filename = f"{clip_emotion_label[i]}_{filename}"
                output_file_path = os.path.join(output_folder, output_filename)

                #TODO: Remove this. 
                if not os.path.exists(output_file_path):
                    subclip.write_videofile(output_file_path)

                start_time = step_time(start_time, clip_len, fs_video)

                if i < len(clip_lens) - 1:
                    skip_amount = cooldown_lable_lens[i] + 35.01 + clip_label_lens[i]
                    start_time = step_time(start_time, skip_amount, fs_video)

                print(f'Exported {output_filename}')
        except:
            print(f'Failed to process {filename}\n')
