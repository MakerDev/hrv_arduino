import matplotlib.pyplot as plt
import csv
import serial
import time
import datetime
import pickle

def get_timestamp():
    current_time = datetime.datetime.now()
    current_time_ms = int(time.time()*1000)

    return current_time, current_time_ms

def save_csv(filename, logs):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(logs)


if __name__ == '__main__':
    baud_rate = 9600
    ser = serial.Serial('COM7', baud_rate)
    ser.close()
    ser.open()
    logs = [['index', "timestamp", "millisec", "raw-value", 'bpm']]
    
    fs = 80
    '''
    1분에 한 파일 나오도록 설계
    '''
    save_period = fs * 60 * 1

    i = 0
    while True:
        try:
            data = str(ser.readline().decode('utf-8')).strip()
            data = data.split(',')

            if len(data) <= 1:
                continue

            value, bpm = data
            dt, ms = get_timestamp()
            log = [i,dt, ms, value, bpm]
            logs.append(log)
            print(log)
            i += 1

            if i % save_period == 0:
                save_csv(f'ppg_logs_{i/save_period}.csv', logs[i-save_period:])
        except KeyboardInterrupt:
            break

    ser.close()

    with open('raw_dump.pkl', 'wb')  as f:
        pickle.dump(logs, f)

    save_csv('ppg_logs_final.csv', logs=logs)

