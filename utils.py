import matplotlib.pyplot as plt
import csv
import numpy as np
import scipy
from scipy.signal import filtfilt, butter, find_peaks



'''
Filter not applied

Return: 
    timestamps, readings
'''
def load_readings(filename, offset=400, apply_filter=True):
    timestamps = []
    readings = []

    with open(filename) as f:
        rdr = csv.reader(f)
        first = True
        #수집된 csv의 첫줄에 reading이 없는 버그가 있어서 첫 줄은 스킵
        for line in rdr:
            if first:
                first = False
                continue

            timestamp, reading = line

            try:
                reading = float(reading) - offset
            except:
                reading = readings[-1]
            
            readings.append(reading)
            timestamps.append(timestamp[11:])
            
    if apply_filter:
        b, a = butter(5, 0.1)
        readings = filtfilt(b, a, readings)    
        readings = np.asarray(readings)

    return timestamps, readings
