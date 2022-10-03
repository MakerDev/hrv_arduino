
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import utils
from models.aespa_dataset import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import sklearn.metrics as metrics


class Conv1dNetwork(nn.Module):
    def __init__(self, out_channel=7, in_channel=1):
        super(Conv1dNetwork, self).__init__()
        self.conv1d_1 = nn.Conv1d(in_channels=in_channel,
                                out_channels=16,
                                kernel_size=32,
                                stride=4,
                                padding=1)

        self.conv1d_2 = nn.Conv1d(in_channels=16,
                                out_channels=32,
                                kernel_size=32,
                                stride=4,
                                padding=1)
        self.pool = nn.MaxPool1d(4)
        self.dropout = nn.Dropout(0.5)
        self.conv_net = nn.Sequential(
            self.conv1d_1,
            self.pool,
            self.conv1d_2,
            self.pool
        )

        flat_size = self.get_flat_size((1, 60000), self.conv_net)

        self.fc_layer1 = nn.Linear(flat_size, 1024)
        self.fc_layer2 = nn.Linear(1024, 512)
        self.fc_layer3 = nn.Linear(512, out_channel)
        self.fc_net = nn.Sequential(
            self.fc_layer1,
            nn.ReLU(),
            self.dropout,
            self.fc_layer2,
            nn.ReLU(),
            self.dropout,
            self.fc_layer3,
            nn.ReLU()
        )
    
    def get_flat_size(self, input_shape, conv_net):
        f = conv_net(torch.Tensor((torch.ones(1, *input_shape))))
        return torch.flatten(f).shape[0]


    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv_net(x)
        # x = x.transpose(1, 2)
        x = torch.flatten(x, 1)
                
        # x = self.dropout(x)
        x = self.fc_net(x)

        return x



if __name__ == "__main__":
    # region Settings
    BATCH_SIZE  = 4
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
    MODEL_NAME    = "ASEPA_PPG_SAM_TWO_DROPOUT"
    
    TB_LOG_DIR = f"tb_logs/{MODEL_NAME.lower()}"
    
    if not os.path.exists(TB_LOG_DIR):
        os.mkdir(TB_LOG_DIR)

    writer = SummaryWriter(TB_LOG_DIR, comment=f'K_32_{MODEL_NAME}', filename_suffix=MODEL_NAME)

    # region TRAINING
    DM = AESPADataManager("./ppgs_sep", BATCH_SIZE)
    TRAIN_PPG_DATA, TEST_PPG_DATA = DM.load_dataset(as_hrv=True, as_sam=True)
    TRAIN_LOADER, TEST_LOADER     = DM.load_dataloader(TRAIN_PPG_DATA, TEST_PPG_DATA)
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    # MODEL = Conv1d_LSTM(out_channel=NUM_CLASSES)
    MODEL = Conv1dNetwork(out_channel=NUM_CLASSES)

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
        print(metrics.classification_report(result_anno_np, result_pred_np, zero_division=0))
        print(conf_mat)

        if epoch in [10, 20, 30, 50, 70, 85, 100, 120, 150, 170, 200, 250, 500]:
            torch.save(MODEL.state_dict(), f"savepoints/aespa_hrv/savepoint_{epoch}_{total_acc_mean*100:.1f}_{MODEL_NAME}.pth")
        
        if epoch >= 100 and total_acc_mean > best_acc:
            torch.save(MODEL.state_dict(), f"savepoints/aespa_hrv/savepoint_{epoch}_{total_acc_mean*100:.1f}_{MODEL_NAME}.pth")
        best_acc = max(total_acc_mean, best_acc)

        print(f'Best so far: {best_acc*100:.1f}%\n')
    
    writer.close()
    torch.save(MODEL.state_dict(), f"savepoints/aespa_hrv/savepoint_{epoch}_{MODEL_NAME}.pth")