import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from utilities.utils import *
import glob
import os

if __name__ == "__main__":
    dat_files = glob.glob(os.path.join("vreed_dataset", "**/*.dat"))

    for dat_file in dat_files:
        # 119번의 5번 샘플이 1/x임. peek가 없어
        targets, ecg_datas = load_vreed_data(dat_file)

        readings = []

        for i in range(0, len(ecg_datas)):
            readings.extend(ecg_datas[i])

        readings = np.asarray(readings)
        t = np.arange(0., len(readings), 1)

        plt.plot(t, readings, 'g')
        #plt.xticks(t, timestamps)
        #plt.xlim([0, max_pt])
        plt.ylim([-1, 1])

        '''
        ECG DETECTOR TEST
        '''
        try:
            fs = 1000 #CSV 파일 보고 역산
            r_peaks, props = find_peaks(readings, distance=800)
            rr_intervals = np.diff(r_peaks)
            heart_rates = 60.0/(rr_intervals/fs)

            heart_rates_interp = np.interp(t, r_peaks[1:], heart_rates)
            heart_rates_interp = normalize_data(heart_rates_interp)

            plt.plot(r_peaks, readings[r_peaks], 'bo')
            plt.plot(t, heart_rates_interp, color='violet')
        except:
            pass
        
        # plt.savefig(f"hrv_pictures/vreed/{os.path.basename(dat_file)}.png")
        plt.show()
    