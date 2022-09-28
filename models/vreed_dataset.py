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
    
    
    def Load_Dataset(self, target_seq_len = 350000, pad_infront=True):        
        spatial_transform = [ToTensor()]
        spatial_transform = Compose(spatial_transform)

        dat_files = glob.glob(os.path.join(self.root_path, "**/*.dat"))

        inputs = []
        targets = []

        for dat_file in dat_files:
            labels, ecg_datas = utils.load_vreed_data(dat_file=dat_file)            

            if len(labels) != len(ecg_datas):
                continue

            for ecg_data in ecg_datas:
                data_seq_len = len(ecg_data)
                if data_seq_len >= target_seq_len:
                    inputs.append(ecg_data[:target_seq_len])
                else:
                    diff = target_seq_len - data_seq_len                    
                    inputs.append(np.concatenate(([-1] * diff, ecg_data)))
            
            targets.extend(labels)

        inputs = torch.Tensor(inputs).float().reshape(-1, target_seq_len, 1)
        targets = torch.Tensor(targets).long()
        x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.2, shuffle=False)

        return HRVDataset(x_train, y_train), HRVDataset(x_test, y_test)
    
    def Load_DataLoader(self, train, test):
        return DataLoader(train, num_workers=4, batch_size=self.batch_size, shuffle=True, pin_memory=True, drop_last=True), \
                DataLoader(test, num_workers=4, batch_size=self.batch_size, shuffle=False, pin_memory=True, drop_last=True)

if __name__ == "__main__":
    DM = HRVDataManager("./vreed_dataset")
    TRAIN_DATASET, TEST_DATASET = DM.Load_Dataset()
