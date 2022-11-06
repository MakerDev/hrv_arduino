import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from models.aespa_dataset import AESPADataManager
from models.gesture_dataset import GestureDataManager
import torch.utils.data as data
import numpy as np
import random

def sample_indices(targets, label, count):
    indices = [i for i, x in enumerate(targets) if x == label]

    return np.asarray([random.choice(indices) for _ in range(count)])

def make_set(input_aespa, target_aespa, input_gesture, target_gesture, target_label, num_set):
    indices_aespa = sample_indices(target_aespa, target_label, num_set)
    indices_gesture = sample_indices(target_gesture, target_label, num_set)

    #최종 결과들
    inputs = []
    targets = [target_label] * num_set

    for i, index_aespa in enumerate(indices_aespa):
        index_gesture = indices_gesture[i]
        inputs.append((input_aespa[index_aespa], input_gesture[index_gesture]))
        
    return inputs, targets

#TODO: 지금 dataset 클래스가 세 개인데 그럴 필요가 있을까. 그냥 기본 하나만 있으면 될 거 같은데
class MultimodalDataset(data.Dataset):
    def __init__(self, inputs, targets):
        super().__init__()

        self.inputs = inputs
        self.targets = targets
    
    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return len(self.inputs)

# VREED HRV data manager
class MultimodalDataManager():
    #TODO: DM_aespa 파라미터 수정 가능하게 만들기.
    def __init__(self, batch_size, aespa_dataset_path):
        self.batch_size = batch_size
        
        DM_AESPA = AESPADataManager(aespa_dataset_path, batch_size=batch_size)
        self.aespa_train_dataset, self.aespa_test_dataset = DM_AESPA.load_dataset()
        self.aespa_inputs, self.aespa_targets = DM_AESPA.get_all_items()

        DM_GESTURE = GestureDataManager(batch_size)
        self.gesture_train_dataset, self.gesture_test_dataset = DM_GESTURE.Load_Dataset()
        self.gesture_inputs, self.gesture_targets = DM_GESTURE.get_all_items()

    def load_dataset(self):        
        all_inputs = []
        all_targets = []

        # The number of sets for each label. -> 각 레이블 별로 300세트가 나오는것.
        #TODO: random choice 결과 저장해서 잘되는걸로 계속 불러오게 하기.
        num_set = 300

        for label in range(7):
            inputs, targets = make_set(self.aespa_inputs, self.aespa_targets, self.gesture_inputs, self.gesture_targets, label, num_set)
            all_inputs.extend(inputs)
            all_targets.extend(targets)

        #TODO: Make sets of inputs. (input_gesture1, input_aespa1), target
        # Shape: num_set*7 x 
        self.inputs = all_inputs
        self.targets = all_targets

        x_train, x_test, y_train, y_test = train_test_split(all_inputs, all_targets, test_size=0.2, random_state=77, stratify=all_targets)

        return MultimodalDataset(x_train, y_train), MultimodalDataset(x_test, y_test)

    def load_dataloader(self, train, test, batch_size=0):
        if batch_size == 0:
            batch_size = self.batch_size

        return DataLoader(train, num_workers=4, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True), \
                DataLoader(test, num_workers=4, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=True)


        
if __name__ == '__main__':
    DM = MultimodalDataManager(batch_size=4, aespa_dataset_path='./ppgs_sep')
    TRAIN_DATA, TEST_DATA = DM.load_dataset()