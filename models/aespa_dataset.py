from torch.utils.data import DataLoader
from spatial_transforms import ToTensor, Compose
import torch.utils.data as data
import glob
import os
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from sklearn.model_selection import train_test_split
import utilities.utils as utils
import numpy as np
import torch
import pandas as pd

# 중간에 suprise오타 처리를 위해 두 개 넣음
#TODO: config 것을 사용하도록 바꿔라
LABEL_TO_INDEX = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
    "suprise": 6,
    "surprise": 6
}

def to_sam_label(emotion_label):
    '''
    Valence, Arousal
    LL: Disgust, Sad
    LH: Angry, Fear
    HL: Happy
    HH: Surprised
    Baseline: Neutral
    '''
    #TODO: 다르게 분류한 것도 있어서 그런 레이블로도 시도해보기.
    if emotion_label in ["sad"]:
        return 0
    elif emotion_label in ["anger", "fear", "disgust"]:
        return 1
    elif emotion_label in ["happy"]:
        return 2
    elif emotion_label in ["suprise"]:
        return 3
    else:
        return 4
    
def load_survey_labels(label_file_path="D://AESPA Dataset/data summary_by subjects.csv"):
    df_responses = pd.read_csv(label_file_path, skiprows=0, encoding='utf-8')
    dict_responses = df_responses.set_index('이름').to_dict("index")

    #TODO: 데이터 가공하기.

    return dict_responses
    

class AESPADataset(data.Dataset):
    def __init__(self, inputs, targets):
        super().__init__()
        self.inputs, self.targets = inputs, targets
    
    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return len(self.inputs)

    def get_all_items(self):
        return self.inputs, self.targets

# VREED HRV data manager
class AESPADataManager():
    def __init__(self, root_path, config_file=None, batch_size=4):
        self.root_path = root_path
        self.config_file = config_file
        self.batch_size  = batch_size
        self.dict_responses = load_survey_labels(label_file_path="D://AESPA Dataset/data summary_by subjects.csv")
    
    def convert_to_hrv(self, ecg_data, fs=300, distance=150, height=100):
        t = np.arange(0., len(ecg_data), 1)
        r_peaks, rr_intervals = utils.calc_rr_intervals(ecg_data, distance=distance, height=height)

        heart_rates = 60.0/(rr_intervals/fs)
        heart_rates_interp = np.interp(t, r_peaks[1:], heart_rates)
        heart_rates_interp = utils.normalize_data(heart_rates_interp)

        return heart_rates_interp

    def interpolate_by_padding(self, readings, target_seq_len):
        seq_len = len(readings)
        if seq_len >= target_seq_len:
            return readings[:target_seq_len]
        else:
            diff = target_seq_len - seq_len                    
            return np.concatenate(([-1] * diff, readings))
    
    #Total sample count = 350
    def load_dataset(self, target_seq_len = 60000, as_hrv = False, as_sam=False, interpolation=False, use_survey=False):
        spatial_transform = [ToTensor()]
        spatial_transform = Compose(spatial_transform)

        csv_files = glob.glob(os.path.join(self.root_path, "**/*.csv"))

        inputs = []
        targets = []
        # TODO: 이거 리팩토링
        for csv_file in csv_files:
            _, readings = utils.load_readings(csv_file, load_only_valid=True)

            if as_hrv:
                # TODO: config file 반영하기.
                readings = self.convert_to_hrv(readings)
            else:
                readings = utils.normalize_data(readings, 60, 140)
            
            if interpolation:
                readings = utils.up_down_sampling(readings, target_seq_len)
                inputs.append(readings)
            else:
                inputs.append(self.interpolate_by_padding(readings, target_seq_len))

            if use_survey:
                user_name = os.path.basename(os.path.dirname(csv_file))
                clip_name = os.path.basename(csv_file).split('_')[-1][:-4]
                label = self.dict_responses[user_name][f'clip_{clip_name}']
            else:
                #[:-5]는 clip number와 extension 제거용.
                label = csv_file.split('_')[-1][:-5]

            if as_sam:
                targets.append(to_sam_label(label))
            else:
                targets.append(LABEL_TO_INDEX[label])

        inputs = torch.Tensor(inputs).float().reshape(-1, target_seq_len, 1)
        targets = torch.Tensor(targets).long()

        self.inputs = inputs
        self.targets = targets
        
        x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.2, shuffle=False)

        return AESPADataset(x_train, y_train), AESPADataset(x_test, y_test)
    
    def get_all_items(self):
        return self.inputs, self.targets

    def load_dataloader(self, train, test, batch_size=0):
        if batch_size == 0:
            batch_size = self.batch_size

        return DataLoader(train, num_workers=4, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True), \
                DataLoader(test, num_workers=4, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=True)

if __name__ == "__main__":
    DM = AESPADataManager("./ppgs_sep")
    TRAIN_DATASET, TEST_DATASET = DM.load_dataset(as_hrv=True, use_survey=True)
