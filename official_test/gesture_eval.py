import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from models.ResNet3D import *
from models.gesture_dataset import *
import torch.nn as nn
import torch
from tqdm.notebook import tqdm
import numpy as np
import utilities.utils as utils


if __name__ == "__main__":
    # region Settings
    BATCH_SIZE = 4
    NUM_CLASSES = 7

    ''''''''''''''''''''''''''''''''''''
    '''          Need to tune        '''
    ''''''''''''''''''''''''''''''''''''
    LEARNING_RATE = 0.000001  # L1
    # LEARNING_RATE  = 0.00005 # L1
    L2WEIGHT_DECAY = 0.000001  # L2
    ''''''''''''''''''''''''''''''''''''
    GPU_NUM = 0

    IS_CUDA = torch.cuda.is_available()
    DEVICE = torch.device('cuda:' + str(GPU_NUM) if IS_CUDA else 'cpu')
    # endregion
    

    # region TRAINING
    DM = GestureDataManager(BATCH_SIZE)
    TRAIN_GESTURE_DATA, TEST_GESTURE_DATA = DM.Load_Dataset()
    TRAIN_LOADER, TEST_LOADER = DM.Load_DataLoader(TRAIN_GESTURE_DATA, TEST_GESTURE_DATA)
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    MODEL = generate_model(18, n_classes=7)

    LOAD_MODEL = True
    if LOAD_MODEL:
        MODEL.load_state_dict(torch.load("official_test/gesture_60.2.pth"))

    MODEL.to(DEVICE)

    total_loss = []
    total_acc = []

    MODEL.eval()
    with torch.no_grad():
        with tqdm(TEST_LOADER, unit='batch') as test_epoch:
            for val_inputs, val_targets in test_epoch:
                val_inputs = val_inputs.to(DEVICE)
                val_targets = val_targets.to(DEVICE)
                val_outputs = MODEL(val_inputs)

                val_loss = criterion(val_outputs, val_targets)
                val_acc = utils.calculate_accuracy(
                    val_outputs, val_targets)

                total_loss.append(val_loss.cpu().detach().numpy())
                total_acc.append(val_acc)
                test_epoch.set_description(f'Evaluating...->')

    total_loss_mean = np.mean(total_loss)
    total_acc_mean = np.mean(total_acc)

    print(f'loss {total_loss_mean:.3f} | Acc {total_acc_mean:.3f}')
