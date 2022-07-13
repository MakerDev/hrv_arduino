import matplotlib.pyplot as plt
import csv
import numpy as np
from ecgdetectors import Detectors
from hrv import HRV

if __name__ == "__main__":
    readings = []
    hrvs = []
    #filename = 'hrv_analyzer/raw_readings.csv'
    filename = '../ppg_logs_11.0.csv'
    with open(filename) as f:
        rdr = csv.reader(f)
        first = True
        for line in rdr:
            if first:
                first = False
                continue
            
            index, timestamp, ms, reading, hrv = line
            readings.append(float(reading))
            hrvs.append(0 if 'inf' in hrv else float(hrv))
    

    t = np.arange(0., len(readings), 1)
    hrvs = np.asarray(hrvs)
    readings = np.asarray(readings)

    plt.plot(t, hrvs, 'r--', t, readings, 'g')
    plt.ylim([0, 200])


    '''
    ECG DETECTOR TEST
    '''
    fs = 150 #CSV 파일 보고 역산
    detectos = Detectors(fs)
    r_peaks = detectos.two_average_detector(readings)
    rr_intervals = np.diff(r_peaks)
    heart_rates = 60.0/(rr_intervals/100)

    heart_rates_interp = np.interp(t, r_peaks[1:], heart_rates)

    plt.plot(r_peaks, readings[r_peaks], 'bo')
    plt.plot(t, heart_rates_interp, color='violet')
    plt.show()

    