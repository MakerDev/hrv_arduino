import os 
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from ResNet3D import ResNet, generate_model
from Conv1DNet import Conv1dNetwork
from gesture_dataset import GestureDataset
from aespa_dataset import AESPADataset
from models.MultimodalDataset import MultimodalDataManager
import argparse
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.optim as optim
import utilities.utils as utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.MultimodelEmbeddingNet import *
import sklearn.metrics as metrics

#TODO: Move this to utils
def print_report_and_confusion_matrix(result_pred, result_anno, num_classes):
    result_pred = np.array(result_pred).reshape(-1, num_classes)
    result_pred = [np.argmax(pred) for pred in result_pred]
    result_pred_np = np.array(result_pred).reshape(-1, 1).squeeze()
    result_anno_np = np.array(result_anno).reshape(-1, 1).squeeze()

    conf_mat = metrics.confusion_matrix(result_anno_np, result_pred_np)
    print(metrics.classification_report(result_anno_np, result_pred_np, zero_division=0))
    print(conf_mat)

def train(model_name, model, train_loader, test_loader, num_classes, savepoint_dir, epoch=200, device='cuda:0', top_k=2):
    # region Settings
    ''''''''''''''''''''''''''''''''''''
    '''          Need to tune        '''
    ''''''''''''''''''''''''''''''''''''
    LEARNING_RATE = 0.00005  # L1
    # LEARNING_RATE  = 0.00005 # L1
    L2WEIGHT_DECAY = 0.00005  # L2
    ''''''''''''''''''''''''''''''''''''
    # endregion

    TB_LOG_DIR = f"tb_logs/{model_name.lower()}"
    os.makedirs(TB_LOG_DIR, exist_ok=True)

    writer = SummaryWriter(TB_LOG_DIR, comment=f'{model_name}', filename_suffix=model_name)
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    # region TRAINING
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2WEIGHT_DECAY)

    best_acc = 0.0    

    for epoch in range(1, epoch+1):
        model.train()
        total_loss = []
        total_acc = []
        total_top_k_acc = []

        with tqdm(train_loader, unit='batch') as train_epoch:
            for i, (inputs, targets) in enumerate(train_epoch):
                input_conv, input_resnet = inputs[:, 0].to(device), inputs[:, 1].to(device)
                targets = targets.to(device)
                outputs = model(input_resnet, input_conv)

                loss = criterion(outputs, targets)
                acc = utils.calculate_accuracy(outputs, targets)
                total_loss.append(loss.cpu().detach().numpy())
                total_acc.append(acc)
                total_top_k_acc.append(utils.calculate_top_k_accuracy(outputs, targets, k=top_k))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_epoch.set_description(f'Training epoch {epoch}->')

        train_loss = np.mean(total_loss)
        train_acc = np.mean(total_acc)
        top_k_acc = np.mean(total_top_k_acc)
        writer.add_scalar('training loss', train_loss, epoch)
        writer.add_scalar('training accuracy', train_acc, epoch)
        writer.add_scalar(f'trainig top {top_k} acc', top_k_acc, epoch)

        print(f'Train loss {train_loss:.3f} | Train Acc {train_acc:.3f} | Top-{top_k} Acc {top_k_acc:.3f}')

        total_loss = []
        total_acc = []
        total_top_k_acc = []
        result_pred = []
        result_anno = []

        model.eval()
        with torch.no_grad():
            with tqdm(test_loader, unit='batch') as test_epoch:
                for val_inputs, val_targets in test_epoch:                    
                    val_targets = val_targets.to(device)
                    val_outputs = model(val_inputs)
                    input_conv, input_resnet = val_inputs[:, 0].to(device), val_inputs[:, 1].to(device)
                    val_targets = val_targets.to(device)
                    outputs = model(input_resnet, input_conv)

                    val_loss = criterion(val_outputs, val_targets)
                    val_acc = utils.calculate_accuracy(val_outputs, val_targets)
                    val_top_k_acc = utils.calculate_top_k_accuracy(val_outputs, val_targets, k=top_k)
                    result_pred.append(val_outputs.cpu().detach().numpy())
                    result_anno.append(val_targets.cpu().detach().numpy())

                    total_loss.append(val_loss.cpu().detach().numpy())
                    total_acc.append(val_acc)
                    total_top_k_acc.append(val_top_k_acc)
                    test_epoch.set_description(f'Evaluating...->')

        total_loss_mean = np.mean(total_loss)
        total_acc_mean = np.mean(total_acc)
        total_top_k_acc = np.mean(total_top_k_acc)
        writer.add_scalar('validation loss', total_loss_mean, epoch)
        writer.add_scalar('validation accuracy', total_acc_mean, epoch)
        writer.add_scalar(f'validation top {top_k} acc', top_k_acc, epoch)

        print(f'loss {total_loss_mean:.3f} | Acc {total_acc_mean:.3f} | Top-{top_k} Acc {total_top_k_acc:.3f}')
        print_report_and_confusion_matrix(result_pred, result_anno, num_classes)

        if epoch in [10, 20, 30, 50, 70, 85, 100, 120, 150, 170, 200, 250, 500]:
            torch.save(model.state_dict(), os.path.join(savepoint_dir, f"{model_name}_{epoch}_{total_acc_mean*100:.1f}.pth"))
        elif epoch >= 50 and total_acc_mean > best_acc:
            torch.save(model.state_dict(), os.path.join(savepoint_dir, f"{model_name}_{epoch}_{total_acc_mean*100:.1f}.pth"))
        
        best_acc = max(total_acc_mean, best_acc)
        print(f'Best so far: {best_acc*100:.1f}%\n')

    writer.close()
    torch.save(model.state_dict(), os.path.join(savepoint_dir, f"{model_name}_{epoch}_{total_acc_mean*100:.1f}.pth"))

if __name__ == '__main__':
    GPU_NUM = 0
    IS_CUDA = torch.cuda.is_available()
    DEVICE = torch.device('cuda:' + str(GPU_NUM) if IS_CUDA else 'cpu')
    parser = argparse.ArgumentParser(description='Multimodal arg parser')

    # 입력받을 인자값 설정 (default 값 설정가능)
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--fc_size', type=int, default=512)
    parser.add_argument('--top_k', type=int, default=2)

    args = parser.parse_args()
    COMMENT    = ''
    MODEL_NAME = f"Multimodal_FC_{args.fc_size}" + COMMENT

    print(f'{MODEL_NAME} training start')

    DM = MultimodalDataManager(batch_size=args.batch_size, aespa_dataset_path="./ppgs_sep")
    TRAIN_DATA, TEST_DATA = DM.load_dataset()
    TRAIN_LOADER, TEST_LOADER = DM.load_dataloader(TRAIN_DATA, TEST_DATA)

    #TODO: 더 좋게 학습한 모델로 교체
    convnet = Conv1dNetwork(out_channel=7, seq_len=60000, kernel_size=64, fc_size=512)
    convnet_savepoint = './savepoitns/aespa_ppg/savepoint_221_79.7_PPG_SAM_TWO_DROPOUTS_K_64_L_512.pth'
    convnet.load_state_dict(torch.load(convnet_savepoint))

    resnet_model = generate_model(18, n_classes=7)
    resnet_model.load_state_dict(torch.load("savepoints/gesture_savepoints/savepoint_back_50_84.2.pth"))

    model = MultimodalEmbeddingNet(fc_size=args.fc_size)

    savepoint_dir = f'savepoints/{MODEL_NAME}'
    os.makedirs(savepoint_dir, exist_ok=True)
    train(model_name=MODEL_NAME, model=model, train_loader=TRAIN_LOADER, test_loader=TEST_LOADER, 
        num_classes=7, savepoint_dir=savepoint_dir, epoch=args.epoch, device=DEVICE)
