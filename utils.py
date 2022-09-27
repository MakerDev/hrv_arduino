import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns
import config

from scipy.signal import filtfilt, butter, find_peaks

def plot_confusion_matrix(cf_matrix, normalize=False, to_show=False, savefile_path=None):
    plt.figure(figsize=(10, 9))

    if normalize:
        ax: Axes = sns.heatmap(cf_matrix/np.sum(cf_matrix[0]), annot=True,
                               fmt='.2%', cmap='Blues')
    else:
        ax: Axes = sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Blues')

    ax.set_title('Confusion Matrix with labels')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Ground Truth')

    ax.xaxis.set_ticklabels(config.LABELS, rotation=30)
    ax.yaxis.set_ticklabels(config.LABELS, rotation=30)
    plt.tight_layout()

    if to_show:
        plt.show()

    if savefile_path != None:
        plt.savefig(savefile_path, dpi=150)

def calc_rr_intervals(readings, distance):
    r_peaks, _ = find_peaks(readings, distance=distance)
    rr_intervals = np.diff(r_peaks)

    return rr_intervals

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
