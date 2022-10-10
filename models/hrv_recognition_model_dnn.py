
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import utils
from models.vreed_dataset import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import sklearn.metrics as metrics
from models.Conv1DNet import Conv1dNetwork

# TODO : LSTM없는 그냥 Conv1D만 쓰고 마지막에 concat후 dense 레이어로 가는 모델 만들기.

class Conv1d_LSTM(nn.Module):
    def __init__(self, out_channel=5, in_channel=1):
        super(Conv1d_LSTM, self).__init__()
        self.conv1d_1 = nn.Conv1d(in_channels=in_channel,
                                out_channels=16,
                                kernel_size=64,
                                stride=16,
                                padding=1)
        self.conv1d_2 = nn.Conv1d(in_channels=16,
                                out_channels=32,
                                kernel_size=64,
                                stride=16,
                                padding=1)

        self.lstm = nn.LSTM(input_size=32,
                            hidden_size=50,
                            num_layers=1,
                            bias=True,
                            bidirectional=False,
                            batch_first=True)
        
        self.dropout = nn.Dropout(0.5)
    
        self.fc_layer1 = nn.Linear(50, 32)
        self.fc_layer2 = nn.Linear(32, out_channel)

    def forward(self, x):
	    # Raw x shape : (B, S, F) => (B, 10, 3)        
        # B: Batch, S: Sequence Length, F: Features.
        # In hrv case, feature channel is 1
        # Shape : (B, F, S) => (B, 3, 10)
        x = x.transpose(1, 2)
        # Shape : (B, F, S) == (B, C, S) // C = channel => (B, 16, 10)
        x = self.conv1d_1(x)
        # Shape : (B, C, S) => (B, 32, 10)
        x = self.conv1d_2(x)
        # Shape : (B, S, C) == (B, S, F) => (B, 10, 32)
        x = x.transpose(1, 2)
        
        self.lstm.flatten_parameters()
        # Shape : (B, S, H) // H = hidden_size => (B, 10, 50)
        _, (hidden, _) = self.lstm(x)
        # Shape : (B, H) // -1 means the last sequence => (B, 50)
        x = hidden[-1]
        
        # Shape : (B, H) => (B, 50)
        # x = self.dropout(x)
        # Shape : (B, 32)
        x = F.relu(self.fc_layer1(x))
        # Shape : (B, O) // O = output => (B, 1)
        x = F.relu(self.fc_layer2(x))

        return x

# class Conv1dNetwork(nn.Module):
#     def __init__(self, out_channel=5, in_channel=1):
#         super(Conv1dNetwork, self).__init__()
#         self.conv1d_1 = nn.Conv1d(in_channels=in_channel,
#                                 out_channels=16,
#                                 kernel_size=64,
#                                 stride=16,
#                                 padding=1)
#         self.conv1d_2 = nn.Conv1d(in_channels=16,
#                                 out_channels=32,
#                                 kernel_size=64,
#                                 stride=16,
#                                 padding=1)
        
#         self.pool = nn.MaxPool1d(4)
#         self.fc_layer1 = nn.Linear(2688, 1024)
#         self.fc_layer2 = nn.Linear(1024, 512)
#         self.fc_layer3 = nn.Linear(512, out_channel)

#     def forward(self, x):
# 	    # Raw x shape : (B, S, F) => (B, 10, 3)        
#         # B: Batch, S: Sequence Length, F: Features.
#         # In hrv case, feature channel is 1
#         # Shape : (B, F, S) => (B, 3, 10)
#         x = x.transpose(1, 2)
#         # Shape : (B, F, S) == (B, C, S) // C = channel => (B, 16, 10)
#         x = self.pool(self.conv1d_1(x))
#         # Shape : (B, C, S) => (B, 32, 10)
#         x = self.pool(self.conv1d_2(x))
#         # Shape : (B, S, C) == (B, S, F) => (B, 10, 32)
#         # x = x.transpose(1, 2)
#         x = torch.flatten(x, 1)
                
#         # Shape : (B, H) => (B, 50)
#         # x = self.dropout(x)
#         # Shape : (B, 32)
#         x = F.relu(self.fc_layer1(x))
#         # Shape : (B, O) // O = output => (B, 1)
#         x = F.relu(self.fc_layer2(x))
#         x = F.relu(self.fc_layer3(x))

#         return x



if __name__ == "__main__":
    # region Settings
    BATCH_SIZE  = 8
    NUM_CLASSES = 5

    ''''''''''''''''''''''''''''''''''''
    '''          Need to tune        '''
    ''''''''''''''''''''''''''''''''''''
    LEARNING_RATE  = 0.00005 # L1
    # LEARNING_RATE  = 0.00005 # L1
    L2WEIGHT_DECAY = 0.00005  # L2
    ''''''''''''''''''''''''''''''''''''

    EPOCHS        = 300
    GPU_NUM       = 0
    IS_CUDA = torch.cuda.is_available()
    DEVICE  = torch.device('cuda:' + str(GPU_NUM) if IS_CUDA else 'cpu')
    # endregion

    writer = SummaryWriter('tb_logs/hrv_vreed')

    # region TRAINING
    DM = HRVDataManager("./vreed_dataset", BATCH_SIZE)
    TRAIN_ECG_DATA, TEST_ECG_DATA = DM.load_dataset(as_hrv=True, interpolation=True, target_seq_len=35000)
    TRAIN_LOADER, TEST_LOADER     = DM.load_dataloader(TRAIN_ECG_DATA, TEST_ECG_DATA)
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    MODEL = Conv1d_LSTM(out_channel=NUM_CLASSES)
    # MODEL = Conv1dNetwork(out_channel=NUM_CLASSES, seq_len=350000, fc_size=512, kernel_size=64)

    LOAD_MODEL = False
    if LOAD_MODEL:
        MODEL.load_state_dict(torch.load("savepoints/savepoint_back_50_84.2.pth"))
        
    MODEL.to(DEVICE)

    optimizer = optim.Adam(MODEL.parameters(), lr = LEARNING_RATE)
    
    best_acc = 0.0
    for epoch in range(1, EPOCHS+1):
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

        train_loss = np.mean(total_loss)
        train_acc = np.mean(total_acc)
        writer.add_scalar('training loss', train_loss, epoch)
        writer.add_scalar('training accuracy', train_acc, epoch)

        print(f'Train loss {train_loss:.3f} | Train Acc {train_acc:.3f}')

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
                    val_acc = utils.calculate_accuracy(val_outputs, val_targets)          
                    result_pred.append(val_outputs.cpu().detach().numpy())
                    result_anno.append(val_targets.cpu().detach().numpy())

                    total_loss.append(val_loss.cpu().detach().numpy())
                    total_acc.append(val_acc)
                    test_epoch.set_description(f'Evaluating...->')

        total_loss_mean = np.mean(total_loss)
        total_acc_mean = np.mean(total_acc)
        writer.add_scalar('validation loss', total_loss_mean, epoch)
        writer.add_scalar('validation accuracy', total_acc_mean, epoch)
        result_pred = np.array(result_pred).reshape(-1, NUM_CLASSES)
        result_pred = [np.argmax(pred) for pred in result_pred]
        result_pred_np = np.array(result_pred).reshape(-1, 1).squeeze()
        result_anno_np = np.array(result_anno).reshape(-1, 1).squeeze()

        print(f'loss {total_loss_mean:.3f} | Acc {total_acc_mean:.3f}')
        conf_mat = metrics.confusion_matrix(result_anno_np, result_pred_np)
        print(metrics.classification_report(result_anno_np, result_pred_np))
        print(conf_mat)

        if epoch in [50, 70, 85, 100, 120, 150, 170, 200, 250, 500]:
            torch.save(MODEL.state_dict(), f"savepoints/vreed_hrv/savepoint_{epoch}_{total_acc_mean*100:.1f}.pth")
        
        if epoch >= 100 and total_acc_mean > best_acc:
            torch.save(MODEL.state_dict(), f"savepoints/vreed_hrv/savepoint_{epoch}_{total_acc_mean*100:.1f}.pth")
        best_acc = max(total_acc_mean, best_acc)

        print(f'Best so far: {best_acc*100:.1f}%\n')
    
    writer.close()
    torch.save(MODEL.state_dict(), f"savepoints/vreed_hrv/savepoint_{epoch}.pth")