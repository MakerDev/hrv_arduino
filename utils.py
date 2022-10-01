import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns
import config
import torch
from scipy.interpolate import interp1d
from scipy.signal import filtfilt, butter, find_peaks

def load_vreed_data(dat_file):
    dat = np.load(dat_file, allow_pickle=True)
    targets = dat["Labels"]
    '''
    Data = [13, 2, Length of Data]
    '''
    data = dat["Data"]
    ecg_datas = []

    for data_for_a_video in data:
        ecg_readings = data_for_a_video[:, 1]
        ecg_datas.append(ecg_readings)

    # 데이터가 13개가 다 안들어있는 경우가 있다. => 아니 그럼 뭐가 뭔지 어떻게 알아..?
    # print(len(targets), len(ecg_datas))

    return targets, ecg_datas


def up_down_sampling(input_data, up_size=300):   
    input_len = len(input_data)
   
    input_x = np.linspace(0, input_len, input_len, dtype=np.int32)
    input_y = input_data

    func_quad = interp1d(input_x, input_y, kind='quadratic')
    x_interp  = np.linspace(input_x.min(), input_x.max(), up_size, dtype=np.int32)
    y_interp  = func_quad(x_interp)
       
    return y_interp


def normalize_data(data, val_min=None, val_max=None):
    if val_max == None:
        val_max = np.max(data)
    
    if val_min == None:
        val_min = np.min(data)

    return (data - val_min) / (val_max - val_min) - 0.5


def calculate_accuracy(outputs, targets):
    with torch.no_grad():
        batch_size = targets.size(0)

        _, pred = outputs.topk(1, 1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1))
        n_correct_elems = correct.float().sum().item()

        return n_correct_elems / batch_size


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

def calc_rr_intervals(readings, distance=None, height=None):
    '''
    Return: r_peaks, rr_intervals
    '''
    r_peaks, _ = find_peaks(readings, distance=distance, height=height)
    rr_intervals = np.diff(r_peaks)

    return r_peaks, rr_intervals

def apply_butter_filter(raw_readings, N, Wn):
    b, a = butter(N, Wn)
    readings = filtfilt(b, a, raw_readings)    
    readings = np.asarray(readings)

    return readings

'''
Filter not applied

Return: 
    timestamps, readings
'''
def load_readings(filename, offset=400, apply_filter=True, N=5, Wn=0.1, load_only_valid=False):
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
            
            if load_only_valid and reading < 0:
                continue

            readings.append(reading)
            timestamps.append(timestamp[11:])
            
    if apply_filter:
        b, a = butter(N, Wn)
        readings = filtfilt(b, a, readings)    

    readings = np.asarray(readings)
    return timestamps, readings
