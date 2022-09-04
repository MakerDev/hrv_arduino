import numpy as np
import csv
import random

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

#TODO: 일반적으로 raw sampling rate 얼마인지 체크
def resample_csv(timestamps, readings, fs=30):
    timestamps_resampled = []
    readings_resampled = []

    current_row = 0
    while True:
        current_row, next_sceond_row = capture_second_range(
            current_row, timestamps)

        frames_in_second = next_sceond_row - current_row

        if frames_in_second<=0:
            break
        
        # Sampling
        indices = list(range(current_row, next_sceond_row))
        if frames_in_second < fs:
            #랜덤하게 중복으로 뽑기
            indices.extend(list(np.random.choice(indices, fs - frames_in_second)))
            indices.sort()
        elif frames_in_second > fs:
            indices = sorted(random.sample(indices, fs))

        timestamps_resampled.extend([timestamps[i] for i in indices])
        readings_resampled.extend([readings[i] for i in indices])

        current_row = next_sceond_row

    return timestamps_resampled, readings_resampled


'''
csv파일과 첫 clip의 시작 row 넘버를 입력받아 predefine된 영상의 정보를 이용하여, csv를 잘라낸다.
'''
if __name__ == "__main__":
    timestamps = []
    readings = []
    filename = '2022-08-24 21-19-34_홍요한.csv'
    record_date = filename.split(' ')[0]
    with open(filename) as f:
        rdr = csv.reader(f)
        first = True
        for line in rdr:
            if first:
                first = False                
                continue

            timestamp, reading = line

            try:
                reading = float(reading)
            except:
                reading = readings[-1]

            readings.append(reading)
            timestamps.append(timestamp[11:])  # 요일은 제거

    start_row = 65432
    clip_label_len = 4
    cooldown_label_len = 2.15
    cooldown_clip_len = 35
    clip_lens = [201.06, 148.0, 54.11, 208.14, 96.18, 142.20,
                 111.23, 131.12, 106.06, 64.22, 126.20, 128.09, 151.09, 40.06]
    clip_emotion_label = ['neutral1', 'fear1', 'suprise1', 'sad1', 'disgust1', 'fear2',
                          'anger1', 'happy1', 'neutral2', 'disgust2', 'anger2', 'happy2', 'sad2', 'suprise2']
    # start_row에서 보험용으로 200개 더.
    timestamps = timestamps[start_row-200:]
    readings = readings[start_row-200:]
    timestamps, readings = resample_csv(timestamps, readings, 100)

    #export resampled readings
    filename_resampled = filename.replace('.csv', '_resampled.csv')
    with open(filename_resampled, 'w', newline='') as f:
        wr = csv.writer(f)
        for i in range(len(readings)):
            wr.writerow([f'{record_date} {timestamps[i]}', readings[i]])
        
