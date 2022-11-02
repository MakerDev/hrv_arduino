import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from torch.utils.tensorboard import SummaryWriter
from models.ResNet3D import *
from models.gesture_dataset import *
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
from tqdm.notebook import tqdm
import numpy as np
import utilities.utils as utils
import sklearn.metrics as metrics

def print_report_and_confusion_matrix(result_pred, result_anno, num_classes):
    result_pred = np.array(result_pred).reshape(-1, num_classes)
    result_pred = [np.argmax(pred) for pred in result_pred]
    result_pred_np = np.array(result_pred).reshape(-1, 1).squeeze()
    result_anno_np = np.array(result_anno).reshape(-1, 1).squeeze()

    conf_mat = metrics.confusion_matrix(result_anno_np, result_pred_np)
    print(metrics.classification_report(result_anno_np, result_pred_np, zero_division=0))
    print(conf_mat)

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

    EPOCHS = 300
    GPU_NUM = 0

    IS_CUDA = torch.cuda.is_available()
    DEVICE = torch.device('cuda:' + str(GPU_NUM) if IS_CUDA else 'cpu')
    # endregion

    os.makedirs('tb_logs/gesture_model', exist_ok=True)

    writer = SummaryWriter('tb_logs/gesture_model')

    MODEL = generate_model(18, n_classes=7)

    LOAD_MODEL = False
    if LOAD_MODEL:
        MODEL.load_state_dict(torch.load("savepoints/gesture_savepoints/savepoint_150.pth"))

    MODEL.to(DEVICE)

    # region TRAINING
    DM = GestureDataManager(BATCH_SIZE)
    TRAIN_GESTURE_DATA, TEST_GESTURE_DATA = DM.Load_Dataset()
    TRAIN_LOADER, TEST_LOADER = DM.Load_DataLoader(TRAIN_GESTURE_DATA, TEST_GESTURE_DATA)
    
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = optim.Adam(MODEL.parameters(), lr=LEARNING_RATE, weight_decay=L2WEIGHT_DECAY)

    best_acc = 0 
    for epoch in range(0, EPOCHS+1):
        MODEL.train()
        total_loss = []
        total_acc = []
        with tqdm(TRAIN_LOADER, unit='batch') as train_epoch:
            for i, (inputs, targets) in enumerate(train_epoch):
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)
                outputs = MODEL(inputs)

                loss = criterion(outputs, targets)
                acc = utils.calculate_accuracy(outputs, targets)
                total_loss.append(loss.cpu().detach().numpy())
                total_acc.append(acc)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_epoch.set_description(f'Training epoch {epoch}->')
        total_loss_mean = np.mean(total_loss)
        total_acc_mean = np.mean(total_acc)
        writer.add_scalar("Loss/train", np.mean(total_loss_mean), epoch)
        writer.add_scalar("Acc/train", np.mean(total_acc_mean), epoch)
        print(f'train loss {total_loss_mean:.3f} | train acc {total_acc_mean:.3f}')

        total_loss = []
        total_acc = []
        result_pred = []
        result_anno = []

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
                    result_pred.append(val_outputs.cpu().detach().numpy())
                    result_anno.append(val_targets.cpu().detach().numpy())

                    total_loss.append(val_loss.cpu().detach().numpy())
                    total_acc.append(val_acc)
                    test_epoch.set_description(f'Evaluating...->')

        total_loss_mean = np.mean(total_loss)
        total_acc_mean = np.mean(total_acc)
        writer.add_scalar("Loss/test", total_loss_mean, epoch)
        writer.add_scalar("Acc/test", total_acc_mean, epoch)
        writer.flush()

        print(f'Epoch {epoch}')
        print(f'loss {total_loss_mean:.3f} | Acc {total_acc_mean:.3f}')
        print_report_and_confusion_matrix(result_pred, result_anno, 7)
        best_acc = max(total_acc_mean, best_acc)
        print(f'Best so far: {best_acc*100:.2f}\n')

        if total_acc_mean >= 0.6:
            torch.save(MODEL.state_dict(), f"official_test/gesture_{total_acc_mean*100:.1f}.pth")

        if epoch in [10, 20, 30, 50, 70, 85, 100, 120, 150, 170, 200, 250, 500] or (epoch >= 100 and total_acc_mean > best_acc):
            torch.save(MODEL.state_dict(), f"savepoints/gesture_savepoints/gesture_{epoch}_{total_acc_mean*100:.1f}.pth")

    writer.close()
