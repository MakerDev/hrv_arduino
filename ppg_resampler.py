import os
import numpy as np
import csv
import random
import math
import utils
import datetime

def capture_second_range(start_row, timestamps):
    '''
    1초에 들어온 샘플 수가 제각각이므로 fs에 맞게 세팅.
    만약 1초에 있는 샘플 수가 fs 미만이면 uniform 하게 샘플해서 늘리고, 넘치면 마찬가지로 uniform샘플하기.
    그런데 샘플을 보면 1초에 몇 백개씩 들어가있어서 일단 넘치는거 기준으로 먼저보면 좋을듯.

    Params:
        start_row: This should be index of the row containing the first frame timestamp.
    Return:
        start_row of this second and the start_row of the next second.
        If this is the last second of the given timestamps, both return values are the same as the input start_row
    '''
    start_time = timestamps[start_row]
    # start_second = start_time.split(':')[-1].split('.')[0]
    start_second = extract_second(start_time)
    end_of_line = True

    for i, timestamp in enumerate(timestamps[start_row:]):
        current_second = extract_second(timestamp)
        # current_second = timestamp.split(':')[-1].split('.')[0]
        if current_second != start_second:
            end_of_line = False
            break
    
    if end_of_line:
        return start_row, start_row

    return start_row, start_row + i



def jump_clip_by(start_row, clip_len, fs_csv, fs_video=25):
    '''
    시작 행으로부터, clip_len만큼 앞으로 가기

    return 그만큼 앞으로 간 결과
    '''
    seconds = int(math.trunc(clip_len))
    frames = int(round(clip_len, 2) * 100 % 100) #소수점 계산 오류로 30.00 같은데 29.99999로 나올때 버그 발생
    fs_ratio = fs_csv // fs_video
    
    skip = seconds * fs_csv + frames * fs_ratio
    return start_row + skip



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


def fill_empty_timestamps(timestamps, readings, target_fs):
    '''
    중간에 녹화가 안된 부분들을 감지하고, 해당 부분들을 -1로 채우거나 옵션에 따라 보간하기
    return: 보간 완료된 timestamps와 readings
    '''
    start_row = 0
    cnt = 0

    while True:
        _, next_start_row = capture_second_range(start_row, timestamps)

        #The last second.
        if start_row == next_start_row:
            print(cnt)
            return timestamps, readings

        next_second_truth = tick_second(timestamps[start_row])
        next_second = timestamps[next_start_row]
        #ms는 비교하지 않음.
        if extract_second(next_second_truth) != extract_second(next_second):
            #18:27:43.968400에서 18:27:44.968400 얻기.
            #TODO: ms 단위도 interpolate하기.
            timestamps_interpolation = [next_second_truth] * target_fs
            #중간에 빠진 부분 있으면 -1로 채우기.
            reading_interpolation = [-1] * target_fs
            timestamps_interpolated = timestamps[:next_start_row] + timestamps_interpolation + timestamps[next_start_row:]
            readings_interpolated = readings[:next_start_row] + reading_interpolation + readings[next_start_row:]

            #바로 넣으면 timestamps가 변형되면서 밀리는 현상 발생하는 걸로 보임.
            timestamps = timestamps_interpolated
            readings = readings_interpolated
            cnt+=1
        start_row = next_start_row
            

def resample_csv_internal(timestamps, readings, target_fs):
    timestamps_resampled = []
    readings_resampled = []

    current_row = 0

    while True:
        current_row, next_sceond_row = capture_second_range(
            current_row, timestamps)

        frames_in_second = next_sceond_row - current_row
        if frames_in_second<=0:
            break

        #TODO : target_fs대비 일정 비율 만큼도 없으면 아예 그 second는 날려버리기?
        # 20%보다도 적은 샘플밖에 없으면 녹화 안된걸로 친다.
        if next_sceond_row - current_row < target_fs * 0.1:
            current_row = next_sceond_row
            continue

        # Sampling
        indices = list(range(current_row, next_sceond_row))
        if frames_in_second < target_fs:
            #랜덤하게 중복으로 뽑기
            indices.extend(list(np.random.choice(indices, target_fs - frames_in_second)))
            indices.sort()
        elif frames_in_second > target_fs:
            indices = sorted(random.sample(indices, target_fs))

        timestamps_resampled.extend([timestamps[i] for i in indices])
        # readings_resampled.extend([readings[i] for i in indices])

        if next_sceond_row - current_row >= 3:
            current_readings_resampled = utils.up_down_sampling(readings[current_row:next_sceond_row], up_size=target_fs)
        else:
            current_readings_resampled = [readings[i] for i in indices]
        readings_resampled.extend(current_readings_resampled)
        
        current_row = next_sceond_row
    return timestamps_resampled, readings_resampled


def resample_csv(path, resampled_csv_path, fs=300):
    timestamps, readings = utils.load_readings(path, offset=400, apply_filter=False)

    start_row = 200
    timestamps = timestamps[start_row-200:]
    readings = readings[start_row-200:]
    timestamps, readings = resample_csv_internal(timestamps, readings, fs)
    timestamps, readings = fill_empty_timestamps(timestamps, readings, fs)
    record_date = os.path.basename(path).split(' ')[0]

    #export resampled readings    
    with open(resampled_csv_path, 'w', newline='') as f:
        wr = csv.writer(f)
        for i in range(len(readings)):
            wr.writerow([f'{record_date} {timestamps[i]}', readings[i]])

    return timestamps, readings

'''
csv파일과 첫 clip의 시작 row 넘버를 입력받아 predefine된 영상의 정보를 이용하여, csv를 잘라낸다.
'''
if __name__ == "__main__":
    timestamps = []
    readings = []
    folder = 'ppgs'
    filename = '2022-08-26 21-21-20_김기현.csv'
    
    csv_files = os.listdir(folder)
    record_date = filename.split(' ')[0]

    path = os.path.join(folder, filename)
    resampled_file_path = path.replace('.csv', '_resampled.csv')
    fs = 300
    if not os.path.exists(resampled_file_path):
        timestamps, readings = resample_csv(path, resampled_file_path, fs)
    else:
        timestamps, readings = utils.load_readings(resampled_file_path, offset=0, apply_filter=False)

    fs_video = 25

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
    folder = f"ppgs_sep/{filename.split('_')[1][:-4]}"

    if not os.path.exists(folder):
        os.mkdir(folder)

    #Divide PPG raw datas.
    start_row = int(input("Enter the first clip row: "))
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