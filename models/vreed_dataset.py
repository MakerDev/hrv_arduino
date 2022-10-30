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

class HRVDataset(data.Dataset):
    def __init__(self, inputs, targets):
        super().__init__()
        self.inputs, self.targets = inputs, targets
    
    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return len(self.inputs)

# VREED HRV data manager
class HRVDataManager():
    def __init__(self, root_path, batch_size=4):
        self.root_path = root_path
        self.batch_size  = batch_size
    
    def convert_to_hrv(self, ecg_data, fs=1000):
        t = np.arange(0., len(ecg_data), 1)
        r_peaks, rr_intervals = utils.calc_rr_intervals(ecg_data, distance=500)

        heart_rates = 60.0/(rr_intervals/fs)
        heart_rates_interp = np.interp(t, r_peaks[1:], heart_rates)
        heart_rates_interp = utils.normalize_data(heart_rates_interp)

        return heart_rates_interp

    
    def load_dataset(self, target_seq_len = 350000, pad_infront = True, as_hrv = False, interpolation=True):        
        spatial_transform = [ToTensor()]
        spatial_transform = Compose(spatial_transform)

        dat_files = glob.glob(os.path.join(self.root_path, "**/*.dat"))

        inputs = []
        targets = []

        for dat_file in dat_files:
            labels, ecg_datas = utils.load_vreed_data(dat_file=dat_file)            

            if len(labels) != len(ecg_datas):
                continue

            for i, ecg_data in enumerate(ecg_datas):
                if as_hrv:
                    try:
                        # 119_ECG_GSR_PreProcessed의 5번 인덱스가 
                        # 모양이 이상해서 peek detection이 안됨
                        ecg_data = self.convert_to_hrv(ecg_data)
                    except:
                        continue

                data_seq_len = len(ecg_data)

                if interpolation:
                    ecg_data = utils.up_down_sampling(ecg_data, target_seq_len)
                    inputs.append(ecg_data)
                else:
                    if data_seq_len >= target_seq_len:
                        inputs.append(ecg_data[:target_seq_len])
                    else:
                        diff = target_seq_len - data_seq_len                    
                        inputs.append(np.concatenate(([-1] * diff, ecg_data)))
            
                targets.append(labels[i])

        # print(targets.count(4))

        inputs = torch.Tensor(inputs).float().reshape(-1, target_seq_len, 1)
        targets = torch.Tensor(targets).long()
        x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.2, shuffle=False)

        return HRVDataset(x_train, y_train), HRVDataset(x_test, y_test)
    
    def load_dataloader(self, train, test):
        return DataLoader(train, num_workers=4, batch_size=self.batch_size, shuffle=True, pin_memory=True, drop_last=True), \
                DataLoader(test, num_workers=4, batch_size=self.batch_size, shuffle=False, pin_memory=True, drop_last=True)

if __name__ == "__main__":
    DM = HRVDataManager("./vreed_dataset")
    TRAIN_DATASET, TEST_DATASET = DM.load_dataset(as_hrv=True)
