import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import utils
import os

if __name__ == "__main__":
    # filename = 'ppgs/2022-08-19 17-37-33_박준하_resampled.csv'
    # timestamps, readings = utils.load_readings(filename, offset=0)

    # t = np.arange(0., len(readings), 1)
    # max_pt = 20000

    # plt.plot(t, readings, 'g')
    # #plt.xticks(t, timestamps)
    # plt.xlim([0, max_pt])
    # plt.ylim([40, 220])


    # '''
    # ECG DETECTOR TEST
    # '''
    # fs = 300 #CSV 파일 보고 역산
    # r_peaks, props = find_peaks(readings, distance=150)
    # rr_intervals = np.diff(r_peaks)
    # heart_rates = 60.0/(rr_intervals/fs)

    # heart_rates_interp = np.interp(t, r_peaks[1:], heart_rates)

    # plt.plot(r_peaks, readings[r_peaks], 'bo')
    # plt.plot(t, heart_rates_interp, color='violet')
    # plt.show()
    
    #TODO: 14개 클립 각각에 대해 위 내용들을 한 figure로 plot하기.
    folders = os.listdir('ppgs_sep')
    for folder in folders:
        individual_folder_path = os.path.join('ppgs_sep', folder)
        ppgs = os.listdir(individual_folder_path)

        fig = plt.figure()

        for i, ppg in enumerate(ppgs):
            ppg_path = os.path.join(individual_folder_path, ppg)
            timestamps, readings = utils.load_readings(ppg_path, offset=0)

            ax = fig.add_subplot(2, 7, i+1)
            t = np.arange(0., len(readings), 1)
            max_pt = 20000
            ax.set_title(ppg.split('_')[1][:-4])
            ax.plot(t, readings, 'g')
            #plt.xticks(t, timestamps)
            # ax.xlim([0, max_pt])
            ax.set_ylim([60, 160])

            fs = 300 #CSV 파일 보고 역산
            r_peaks, props = find_peaks(readings, distance=150)
            rr_intervals = np.diff(r_peaks)
            heart_rates = 60.0/(rr_intervals/fs)

            heart_rates_interp = np.interp(t, r_peaks[1:], heart_rates)

            ax.plot(r_peaks, readings[r_peaks], 'bo')
            ax.plot(t, heart_rates_interp, color='violet')

        fig.tight_layout()
        plt.show()