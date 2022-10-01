import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import utils
import numpy as np
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models.gesture_dataset import *
from models.ResNet3D import *

import sklearn.metrics as metrics

from torch.utils.tensorboard import SummaryWriter

def sf(float_input):
    scientific_output = np.format_float_scientific(float_input, trim='-', precision=0, exp_digits=2)
    return scientific_output

# region 3D ResNet
def get_inplanes():
    return [64, 128, 256, 512]


def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model

if __name__ == "__main__":
    # region Settings
    BATCH_SIZE  = 4
    NUM_CLASSES = 7

    ''''''''''''''''''''''''''''''''''''
    '''          Need to tune        '''
    ''''''''''''''''''''''''''''''''''''
    MODEL_DEPTH    = 10

    # LEARNING_RATE  = 0.000001 # L1
    LEARNING_RATE  = 0.00005 # L1
    L2WEIGHT_DECAY = 0.000001  # L2
    ''''''''''''''''''''''''''''''''''''

    EPOCHS        = 300
    GPU_NUM       = 0

    IS_CUDA = torch.cuda.is_available()
    DEVICE  = torch.device('cuda:' + str(GPU_NUM) if IS_CUDA else 'cpu')
    # endregion

    # region EVALUATION
    # evaluation
    DM = GestureDataManager(BATCH_SIZE)
    TRAIN_GESTURE_DATA, TEST_GESTURE_DATA = DM.Load_Dataset()
    _, TEST_LOADER     = DM.Load_DataLoader(TRAIN_GESTURE_DATA, TEST_GESTURE_DATA)
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    MODEL = generate_model(18, n_classes=7)

    LOAD_MODEL = True
    if LOAD_MODEL:
        MODEL.load_state_dict(torch.load("savepoints/savepoint_back_50_95.5.pth"))

    MODEL.to(DEVICE)
    MODEL.eval()
    total_loss = []
    total_acc = []
    Result_pred = []
    Result_anno = []
    with torch.no_grad():
        with tqdm(TEST_LOADER, unit='batch') as test_epoch:
            for val_inputs, val_targets in test_epoch:
                val_inputs = val_inputs.to(DEVICE)
                val_targets = val_targets.to(DEVICE)
                val_outputs = MODEL(val_inputs)

                val_loss = criterion(val_outputs, val_targets)
                val_acc = utils.calculate_accuracy(val_outputs, val_targets)          
                
                Result_pred.append(val_outputs.cpu().detach().numpy())
                Result_anno.append(val_targets.cpu().detach().numpy())
                total_loss.append(val_loss.cpu().detach().numpy())
                total_acc.append(val_acc)
                test_epoch.set_description(f'Evaluating...->')

    total_loss_mean = np.mean(total_loss)
    total_acc_mean = np.mean(total_acc)
    print(f'loss {total_loss_mean:.3f} | Acc {total_acc_mean:.3f}\n')
    Result_pred = np.array(Result_pred).reshape(-1, NUM_CLASSES)
    Result_pred = [np.argmax(pred) for pred in Result_pred]
    Result_pred_np = np.array(Result_pred).reshape(-1, 1).squeeze()
    Result_anno_np = np.array(Result_anno).reshape(-1, 1).squeeze()
    ACC_TEST = metrics.accuracy_score(Result_anno_np, Result_pred_np)
    print('Accuracy: ', ACC_TEST)
    conf_mat = metrics.confusion_matrix(Result_anno_np, Result_pred_np)
    print(metrics.classification_report(Result_anno_np, Result_pred_np))
    print(conf_mat)
    utils.plot_confusion_matrix(conf_mat, to_show=True)
    # endregion