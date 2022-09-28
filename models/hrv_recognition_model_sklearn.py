
import pandas as pd
from gesture_dataset import load_kinematic_dataset, get_file_infos
import joblib
import utils
import time
import numpy as np
import glob
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


FEATURE_NAMES = ["Quad_Cat", 'Mean', 'Min', 'Max', 'MeanRR', 'MedianRR', 'MinRR', 'MaxRR',
                 'LF', 'HF', 'VLF', 'Ibi', 'Bpm', 'Sdnn', 'Sdsd', 'Rmssd', 'Pnn50', 'pnn20', 'Pnn50pnn20']

def find_indices(lst, values):
    indices = []
    for value in values:
        indices.append(lst.index(value))

    return indices

if __name__ == "__main__":
    features_csv = "vreed_dataset/ECG_FeaturesExtracted.csv"
    df_features = pd.read_csv(features_csv, skiprows=1)  # skip header
    np_features = df_features.to_numpy()

    targets = np_features[:, 0]

    # 단일 feature로 했을때, 4-class에서 15~31% 정도 정확도임
    # 단일 feature로 했을때, arousal에서 40~56%정도 나옴. -> ['MeanRR', 'MedianRR', 'MaxRR', 'Ibi', 'Pnn50', 'Min']
    # 단일 feature로 했을때, valence에서 46~54%정도 나옴. -> ['MeanRR', 'MedianRR', 'LF', 'HF', 'Bpm', 'Sdnn', 'Sdsd', 'Pnn50pnn20']

    targets_valence = [ 0 if target==0 or target==1 else 1 for target in targets]
    targets_arousal = [ 0 if target==1 or target==2 else 1 for target in targets]

    acc_over_50 = []

    # features_to_use = find_indices(FEATURE_NAMES, [feature_name])
    features_to_use = find_indices(FEATURE_NAMES, ['MeanRR', 'MedianRR', 'MaxRR', 'LF', 'HF', 'Bpm', 'Sdnn', 'Sdsd', 'Pnn50pnn20', 'Ibi', 'Pnn50'])
    datas = np_features[:, features_to_use]


    x_train, x_test, y_train, y_test = train_test_split(
        datas, targets, test_size=0.2, stratify=targets)

    hidden_layer_sizes = (64, 64)
    model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes)
    # model = SVC()
    model.fit(x_train, y_train)
    # joblib.dump(model, f'pretrained_models/hrv_model_{hidden_layer_sizes}.pkl')

    start = time.time()
    start = time.time()
    preds = model.predict(x_test)
    end = time.time()

    print('Inference time:', end - start)
    print("Train score:", model.score(x_train, y_train))
    print("Test score:", model.score(x_test, y_test))
    conf_matrix = confusion_matrix(y_test, preds)
    # print(conf_matrix)
    print(acc_over_50)
    #utils.plot_confusion_matrix(conf_matrix, to_show=True)
