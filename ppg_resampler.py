import os
import numpy as np
import csv
import random
import math
import utils

'''
1초에 들어온 샘플 수가 제각각이므로 fs에 맞게 세팅.
만약 1초에 있는 샘플 수가 fs 미만이면 uniform 하게 샘플해서 늘리고, 넘치면 마찬가지로 uniform샘플하기.
그런데 샘플을 보면 1초에 몇 백개씩 들어가있어서 일단 넘치는거 기준으로 먼저보면 좋을듯.
Params:
    start_row: This should be index of the row containing the first frame timestamp.
Return:
    start_row of this second and the start_row of the next second.
'''
def capture_second_range(start_row, timestamps):
    start_time = timestamps[start_row]
    start_second = start_time.split(':')[-1].split('.')[0]
    for i, timestamp in enumerate(timestamps[start_row:]):
        current_second = timestamp.split(':')[-1].split('.')[0]
        if current_second != start_second:
            break

    return start_row, start_row + i

def resample_csv(timestamps, readings, target_fs):
    timestamps_resampled = []
    readings_resampled = []

    current_row = 0

    stat = []
    while True:
        current_row, next_sceond_row = capture_second_range(
            current_row, timestamps)

        frames_in_second = next_sceond_row - current_row
        stat.append(frames_in_second)
        if frames_in_second<=0:
            break
        
        # Sampling
        indices = list(range(current_row, next_sceond_row))
        if frames_in_second < target_fs:
            #랜덤하게 중복으로 뽑기
            indices.extend(list(np.random.choice(indices, target_fs - frames_in_second)))
            indices.sort()
        elif frames_in_second > target_fs:
            indices = sorted(random.sample(indices, target_fs))

        timestamps_resampled.extend([timestamps[i] for i in indices])
        readings_resampled.extend([readings[i] for i in indices])

        current_row = next_sceond_row
    return timestamps_resampled, readings_resampled

'''
시작 행으로부터, clip_len만큼 앞으로 가기
return 그만큼 앞으로 간 결과
'''
def jump_clip_by(start_row, clip_len, fs_csv, fs_video=25):
    seconds = int(math.trunc(clip_len))
    frames = int(round(clip_len, 2) * 100 % 100) #소수점 계산 오류로 30.00 같은데 29.99999로 나올때 버그 발생
    fs_ratio = fs_csv // fs_video
    
    skip = seconds * fs_csv + frames * fs_ratio
    return start_row + skip


'''
csv파일과 첫 clip의 시작 row 넘버를 입력받아 predefine된 영상의 정보를 이용하여, csv를 잘라낸다.
'''
if __name__ == "__main__":
    timestamps = []
    readings = []
    folder = 'ppgs'
    filename = '2022-08-24 21-19-34_홍요한.csv'
    record_date = filename.split(' ')[0]

    timestamps, readings = utils.load_readings(os.path.join(folder, filename), apply_filter=False)

    #TODO: clip_start_timestamp를 21:22:47.16 이런식으로 받아서 샘플링 후 fs를 기준으로 굳이 우리가 계산안해도 어디가 시작 frame인지 알도록 하기
    #만약 120으로 샘플링 했는데 위처럼 timestamp가 주어지면 21:22:47의 64번째 row가 시작 frame이 될 것.
    #start_row에서 보험용으로 200개 더.
    start_row = 65432
    timestamps = timestamps[start_row-200:]
    readings = readings[start_row-200:]
    fs = 300
    fs_video = 25
    timestamps, readings = resample_csv(timestamps, readings, fs)

    #export resampled readings
    filename_resampled = filename.replace('.csv', '_resampled.csv')
    with open(os.path.join(folder, filename_resampled), 'w', newline='') as f:
        wr = csv.writer(f)
        for i in range(len(readings)):
            wr.writerow([f'{record_date} {timestamps[i]}', readings[i]])


    clip_label_len = 4
    cooldown_label_len = 2.15
    cooldown_clip_len = 35
    #NOTICE: 아래의 frame들은 fps 25를 기준으로 맞춰진것임. 따라서 25의 배수로 sampling 하는것이 추후 계산시에 편할 것.
    clip_lens = [201.06, 148.0, 54.11, 208.14, 96.18, 142.20,
                 111.23, 131.12, 106.06, 64.22, 126.20, 128.09, 151.09, 40.06]
    clip_emotion_label = ['neutral1', 'fear1', 'suprise1', 'sad1', 'disgust1', 'fear2',
                          'anger1', 'happy1', 'neutral2', 'disgust2', 'anger2', 'happy2', 'sad2', 'suprise2']

    #clip label 2부터 14까지 존재
    clip_label_lens     = [3.24, 3.04, 4.0, 3.24, 3.24, 3.23, 3.24, 3.24, 3.24, 3.24, 3.24, 3.24, 4.0]
    #쉬어가기 1~13까지 존재
    cooldown_lable_lens = [2.15, 2.19, 2.19, 2.19, 2.14, 2.18, 2.20, 2.17, 2.19, 2.20, 2.19, 2.19, 2.19]
    
    #.csv 확장자 제거를 위해 뒤에서 4개는 제거.
    folder = f"{filename.split('_')[1][:-4]}"

    if not os.path.exists(folder):
        os.mkdir(folder)

    start_row = 314
    for i, clip_len in enumerate(clip_lens):
        next_clip_start_row = jump_clip_by(start_row, clip_len, fs, fs_video)
        clip_readings = readings[start_row:next_clip_start_row]
        clip_timestamps = timestamps[start_row:next_clip_start_row]
        clip_name = f'clip_{clip_emotion_label[i]}.csv'

        with open(os.path.join(folder, clip_name), 'w', newline='') as f:
            wr = csv.writer(f)
            for row in range(len(clip_readings)):
                wr.writerow([f'{record_date} {clip_timestamps[row]}', clip_readings[row]])

        #skip label and cooldown. 
        #쉬어가기 영상 35.01초
        #마지막 영상 처리 후에는 스킵 없음.
        if i < len(clip_lens) - 1:
            skip_amount = cooldown_lable_lens[i] + 35.01 + clip_label_lens[i]
            start_row = jump_clip_by(next_clip_start_row, skip_amount, fs, fs_video)