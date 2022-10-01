import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import utils
import os
import glob

# 감정 별로 쪼개놓은 PPG 합쳐서 출력하기.
def plot_all_ppgs_in_one(individual_folder_path, fs=300):
    ppgs = os.listdir(individual_folder_path)

    fig = plt.figure()

    for i, ppg in enumerate(ppgs):
        ppg_path = os.path.join(individual_folder_path, ppg)
        _, readings = utils.load_readings(ppg_path, offset=0)

        ax = fig.add_subplot(2, 7, i+1)
        t = np.arange(0., len(readings), 1)

        ax.set_title(ppg.split('_')[1][:-4])
        ax.plot(t, readings, 'g')
        # plt.xticks(t, timestamps)
        # ax.xlim([0, max_pt])
        ax.set_ylim([60, 160])

        r_peaks, rr_intervals = utils.calc_rr_intervals(readings, 150)
        heart_rates = 60.0/(rr_intervals/fs)
        heart_rates_interp = np.interp(t, r_peaks[1:], heart_rates)

        ax.plot(r_peaks, readings[r_peaks], 'bo')
        ax.plot(t, heart_rates_interp, color='violet')

    fig.tight_layout()
    plt.show()

def plot_by_all_params(readings, N_list, Wn_list, distances, heights, fs=300):
    combinations = [(N, Wn, distance, height)   for N in N_list 
                                                for Wn in Wn_list 
                                                for distance in distances 
                                                for height in heights]

    for N, Wn, distance, height in combinations:
        fig = plt.figure()

        t = np.arange(0., len(readings), 1)

        readings_filtered = utils.apply_butter_filter(readings, N, Wn)
        plt.title(f'{N} {Wn} {distance} {height}')
        plt.plot(t, readings_filtered, 'g')
        plt.ylim([60, 160])

        r_peaks, rr_intervals = utils.calc_rr_intervals(readings_filtered, distance, height)

        heart_rates = 60.0/(rr_intervals/fs)
        heart_rates_interp = np.interp(t, r_peaks[1:], heart_rates)

        plt.plot(r_peaks, readings_filtered[r_peaks], 'bo')
        plt.plot(t, heart_rates_interp, color='violet')

        fig.tight_layout()
        plt.show()


def get_aggregated_ppg(root_folder):
    '''
    ppgs_sep에 감정별로 분리된 csv들을 하나의 reading으로 합쳐서 출력하기.
    '''
    ppg_files = glob.glob(os.path.join(root_folder, "*.csv"))
    readings_aggregated = []
    for ppg_file in ppg_files:
        _, readings = utils.load_readings(ppg_file, load_only_valid=True)
        readings_aggregated.extend(list(readings))

    return np.array(readings_aggregated)


def plot_ecg_readings(readings, fs=300, distance=140, height=120, max_pt=50000, normalize=False):
    t = np.arange(0., len(readings), 1)
    r_peaks, rr_intervals = utils.calc_rr_intervals(readings, distance, height)
    heart_rates = 60.0/(rr_intervals/fs)

    heart_rates_interp = np.interp(t, r_peaks[1:], heart_rates)
    if normalize:
        heart_rates_interp = utils.normalize_data(heart_rates_interp)
        readings = utils.normalize_data(readings, 60, 140)
        plt.ylim([-0.8, 0.8])
    else:
        plt.ylim([40, 160])

    plt.xlim([0, max_pt])
    plt.plot(t, readings, 'g')
    plt.plot(r_peaks, readings[r_peaks], 'bo')
    plt.plot(t, heart_rates_interp, color='violet')
    plt.show()


def plot_ecg_file(file_path, fs=300, N=5, Wn=0.1, distance=140, height=125, max_pt=50000):
    _, readings = utils.load_readings(file_path, apply_filter=False, offset=0, N=N, Wn=Wn, load_only_valid=True)    
    plot_ecg_readings(readings, fs, distance, height, max_pt)

if __name__ == "__main__":
    # filename = 'ppgs/2022-08-19 17-37-33_박준하_resampled.csv'
    clip_emotion_label = ['neutral1', 'fear1', 'suprise1', 'sad1', 'disgust1', 'fear2',
                        'anger1', 'happy1', 'neutral2', 'disgust2', 'anger2', 'happy2', 'sad2', 'suprise2']

    filename = f'ppgs_sep/suan/clip_{clip_emotion_label[0]}.csv'
    file_dir = 'ppgs'
    filename = f'{file_dir}/2022-08-20 20-34-23_suan_resampled_interp.csv'
    
    # plot_ecg(filename)
    # plot_all_ppgs_in_one('ppgs_sep/suan')
    readings_aggregated = get_aggregated_ppg('ppgs_sep/김연재')
    plot_ecg_readings(readings_aggregated, height=115)

    # plot_by_all_params(readings, [2,3,5], Wn_list=[0.1,0.2,0.5], distances=[145, 140], heights=[120, 125])

    # #TODO: 14개 클립 각각에 대해 위 내용들을 한 figure로 plot하기.
    # folders = os.listdir('ppgs_sep')
    # for folder in folders:
    #     individual_folder_path = os.path.join('ppgs_sep', folder)
    #     plot_all_ppgs_in_one(individual_folder_path)