
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import csv
import glob
import numpy as np
from tqdm import tqdm
import random
import pickle
import time
import utils
import config
import joblib


def load_kinematic_dataset(threshold=700):
    root = "D:/AESPA Dataset/kinematic-dataset"
    #1. file_info.csv로부터, 각 파일의 이름을 키로 하여, 그 파일의 정보를 담고있는 dict를 생성한다.
    file_info_path = f"{root}/file-info.csv"
    file_infos = {}
    with open(file_info_path) as f:
        reader = csv.reader(f)
        
        header = next(reader)

        for line in reader:
            filename = line[0]
            file_info = {}
            for i, info in enumerate(line[1:]):
                file_info[header[i+1]] = info
            file_infos[filename] = file_info
    x_data = []
    y_data = []

    
    file_list = glob.glob(os.path.join(root, "BVH_only_motion", '**/*.bvh'), recursive=True)

    for file in tqdm(file_list):
        with open(file) as f:
            reader = csv.reader(f, delimiter=' ')
            data_np = np.genfromtxt(file, delimiter=' ', skip_header=2, skip_footer=1)
            if data_np.shape[0] <= threshold:
                continue

            data_np = data_np[:threshold][:]
            filename = os.path.basename(file)[:-4] #Remove extension.
            label = file_infos[filename]["emotion"]
            target = config.LABEL_TO_INDEX[label]
            x_data.append(data_np)
            y_data.append(target)

    return np.asarray(x_data, dtype=float), np.asarray(y_data, dtype=float)


if __name__ == "__main__":
    x_data, y_data = load_kinematic_dataset()

    x_data = np.average(x_data, axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, stratify=y_data)

    model = MLPClassifier(hidden_layer_sizes=(584, 512, 512))
    model.fit(x_train, y_train)
    joblib.dump(model, f'pretrained_models/gesture_model_584 512 512.pkl')

    start = time.time()
    preds = model.predict(x_test)
    end = time.time()
    print('Inference time:', end - start)

    print("Train score:", model.score(x_train, y_train))
    print("Test score:", model.score(x_test, y_test))
    conf_matrix = confusion_matrix(y_test, preds)
    print(conf_matrix)
    utils.plot_confusion_matrix(conf_matrix, to_show=True)

    
