from torch.utils.data import Dataset, DataLoader
from spatial_transforms import Resize, ToTensor, Compose
import torch.utils.data as data
import glob
import csv
import numpy as np
import os
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import config
import torch
from sklearn.model_selection import train_test_split

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

    for file in tqdm(file_list[:1000]):
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

    return np.asarray(x_data, dtype=float), np.asarray(y_data, dtype=int)        

class GestureDataset(data.Dataset):
    def __init__(self, inputs, targets):
        super().__init__()
        self.inputs, self.targets = inputs, targets
    
    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return len(self.inputs)
        

# Gesture data manager
class GestureDataManager():
    def __init__(self, batch_size=4):
        self.batch_size  = batch_size        
    
    
    def Load_Dataset(self):
        global BATCH_SIZE
        
        #spatial_transform=[Resize((160, 140))]
        spatial_transform = []
        spatial_transform.append(ToTensor())
        spatial_transform = Compose(spatial_transform)

        inputs, targets = load_kinematic_dataset()

        inputs = torch.Tensor(inputs)
        inputs = inputs.reshape(-1, 1, 700, 59, 6)
        # target_onehot = torch.Tensor(np.eye(7)[targets]).long()
        targets = torch.Tensor(targets).long()

        x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.2)

        return GestureDataset(x_train, y_train), GestureDataset(x_test, y_test)
    
    def Load_DataLoader(self, train, test):
        return DataLoader(train, num_workers=4, batch_size=self.batch_size, shuffle=True, pin_memory=True, drop_last=True), \
                DataLoader(test, num_workers=4, batch_size=self.batch_size, shuffle=False, pin_memory=True, drop_last=True)
