from torch.utils.data import DataLoader
from spatial_transforms import ToTensor, Compose
import torch.utils.data as data
import glob
import os
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from sklearn.model_selection import train_test_split
import utils
import numpy as np
import torch

LABEL_TO_INDEX = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
    "suprise": 6
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
    if emotion_label in ["disgust", "sad"]:
        return 0
    elif emotion_label in ["anger", "fear"]:
        return 1
    elif emotion_label in ["happy"]:
        return 2
    elif emotion_label in ["suprise"]:
        return 3
    else:
        return 4
        

class AESPADataset(data.Dataset):
    def __init__(self, inputs, targets):
        super().__init__()
        self.inputs, self.targets = inputs, targets
    
    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return len(self.inputs)

# VREED HRV data manager
class AESPADataManager():
    def __init__(self, root_path, config_file=None, batch_size=4):
        self.root_path = root_path
        self.config_file = config_file
        self.batch_size  = batch_size
    
    def convert_to_hrv(self, ecg_data, fs=300, distance=150, height=115):
        t = np.arange(0., len(ecg_data), 1)
        r_peaks, rr_intervals = utils.calc_rr_intervals(ecg_data, distance=distance, height=height)

        heart_rates = 60.0/(rr_intervals/fs)
        heart_rates_interp = np.interp(t, r_peaks[1:], heart_rates)
        heart_rates_interp = utils.normalize_data(heart_rates_interp)

        return heart_rates_interp

    
    def load_dataset(self, target_seq_len = 60000, pad_infront = True, as_hrv = False):        
        spatial_transform = [ToTensor()]
        spatial_transform = Compose(spatial_transform)

        csv_files = glob.glob(os.path.join(self.root_path, "**/*.csv"))

        inputs = []
        targets = []

        for csv_file in csv_files:
            # TODO: config file 반영하기.
            _, readings = utils.load_readings(csv_file, load_only_valid=True)

            if as_hrv:
                readings = self.convert_to_hrv(readings)
            else:
                readings = utils.normalize_data(readings, 60, 140)
            seq_len = len(readings)

            if seq_len >= target_seq_len:
                inputs.append(readings[:target_seq_len])
            else:
                diff = target_seq_len - seq_len                    
                inputs.append(np.concatenate(([-1] * diff, readings)))

            #[:-4]는 clip number와 extension 제거용.
            label = csv_file.split('_')[-1][:-5]
            targets.append(LABEL_TO_INDEX[label])
            # targets.append(to_sam_label(label))

        inputs = torch.Tensor(inputs).float().reshape(-1, target_seq_len, 1)
        targets = torch.Tensor(targets).long()
        x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.2, shuffle=False)

        return AESPADataset(x_train, y_train), AESPADataset(x_test, y_test)
    
    def load_dataloader(self, train, test):
        return DataLoader(train, num_workers=4, batch_size=self.batch_size, shuffle=True, pin_memory=True, drop_last=True), \
                DataLoader(test, num_workers=4, batch_size=self.batch_size, shuffle=False, pin_memory=True, drop_last=True)

if __name__ == "__main__":
    DM = AESPADataManager("./ppgs_sep")
    TRAIN_DATASET, TEST_DATASET = DM.load_dataset(as_hrv=True)
