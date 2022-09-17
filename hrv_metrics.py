from turtle import distance
import numpy as np
import utils
from scipy.signal import find_peaks

def get_rr_intervals(readings, distance):
    r_peaks, _ = find_peaks(readings, distance=distance)
    rr_intervals = np.diff(r_peaks)

    return rr_intervals

'''
The average of RR intervals (ms)

Return: 
    Average interval in ms. 
'''
def mRR(readings, fs, distance=200):
    rr_intervals = get_rr_intervals(readings, distance=distance)
    mRR = np.average(rr_intervals) / fs * 1000

    return mRR

'''
Standard deviation of RR intervals(ms)
'''
def SDRR(readings, fs, distance=200):
    rr_intervals = get_rr_intervals(readings, distance=distance)
    sdrr = np.std(rr_intervals) / fs * 1000

    return sdrr

def NN50_threshold(fs):
    ms_in_count = fs / 1000
    threshold = ms_in_count * 50 #50ms를 count로 바꾼것.

    return threshold

def NN50_internal(rr_intervals, fs):
    threshold = NN50_threshold(fs)

    count = 0
    for interval in rr_intervals:
        if interval > threshold:
            count+=1

    return count

'''
Number of successive RR interval pairs that differ more than 50ms (count)
'''
def NN50(readings, fs, distance=200):
    rr_intervals = get_rr_intervals(readings, distance=distance)

    return NN50_internal(rr_intervals, fs)

'''
NN50 divieded by the total number of RR intervals (%)
'''
def pNN50(readings, fs, distance=200):
    rr_intervals = get_rr_intervals(readings, distance=distance)
    nn50 = NN50_internal(rr_intervals, fs)

    return nn50 / len(rr_intervals) * 100


if __name__ == "__main__":
    filename = 'anger1.csv'
