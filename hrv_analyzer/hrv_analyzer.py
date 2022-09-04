import matplotlib.pyplot as plt
import csv
import numpy as np
from ecgdetectors import Detectors
from hrv import HRV
import scipy
from scipy.signal import filtfilt, butter, find_peaks



if __name__ == "__main__":
    max_pt = 20000
    b, a = butter(5, 0.1)
    timestamps = []
    readings = []
    hrvs = []
    #filename = 'hrv_analyzer/raw_readings.csv'
    filename = '2022-08-24 21-19-34_홍요한_resampled.csv'
    offset = 400
    with open(filename) as f:
        rdr = csv.reader(f)
        first = True
        for line in rdr:
            if first:
                first = False
                continue
            
            #index, timestamp, ms, reading, hrv = line
            hrv = '100.0'
            timestamp, reading = line

            try:
                reading = float(reading) - offset
            except:
                reading = readings[-1]
            
            readings.append(reading)
            timestamps.append(timestamp[14:])
            hrvs.append(0 if 'inf' in hrv else float(hrv))
    
    readings = filtfilt(b, a, readings)
    t = np.arange(0., len(readings), 1)
    hrvs = np.asarray(hrvs)
    readings = np.asarray(readings)

    plt.plot(t, readings, 'g')
    #plt.xticks(t, timestamps)
    plt.xlim([0, max_pt])
    plt.ylim([40, 220])


    '''
    ECG DETECTOR TEST
    '''
    fs = 380 #CSV 파일 보고 역산
    detectos = Detectors(fs)
    #r_peaks = detectos.two_average_detector(readings)
    r_peaks, props = find_peaks(readings, distance=200)
    rr_intervals = np.diff(r_peaks)
    heart_rates = 60.0/(rr_intervals/fs)

    heart_rates_interp = np.interp(t, r_peaks[1:], heart_rates)

    plt.plot(r_peaks, readings[r_peaks], 'bo')
    plt.plot(t, heart_rates_interp, color='violet')
    plt.show()

    