import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import utils


if __name__ == "__main__":
    filename = '2022-08-24 21-19-34_홍요한_resampled.csv'
    timestamps, readings = utils.load_readings(filename)

    t = np.arange(0., len(readings), 1)
    max_pt = 20000

    plt.plot(t, readings, 'g')
    #plt.xticks(t, timestamps)
    plt.xlim([0, max_pt])
    plt.ylim([40, 220])


    '''
    ECG DETECTOR TEST
    '''
    fs = 380 #CSV 파일 보고 역산
    r_peaks, props = find_peaks(readings, distance=200)
    rr_intervals = np.diff(r_peaks)
    heart_rates = 60.0/(rr_intervals/fs)

    heart_rates_interp = np.interp(t, r_peaks[1:], heart_rates)

    plt.plot(r_peaks, readings[r_peaks], 'bo')
    plt.plot(t, heart_rates_interp, color='violet')
    plt.show()

    