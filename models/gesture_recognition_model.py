import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np

import time
import utilities.utils as utils
import joblib
from gesture_dataset import load_kinematic_dataset, get_file_infos


if __name__ == "__main__":
    root = "D:/AESPA Dataset/kinematic-dataset"
    info_file_path = f"{root}/file-info.csv"
    file_infos = get_file_infos(info_file_path)
    bvh_files_root = os.path.join(root, "BVH_only_motion")
    x_data, y_data = load_kinematic_dataset(file_infos, bvh_files_root)

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

    
