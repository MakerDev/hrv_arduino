import os 
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from models.ResNet3D import ResNet, generate_model
from models.Conv1DNet import Conv1dNetwork
from models.gesture_dataset import GestureDataset
from models.aespa_dataset import AESPADataset
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


def eval(model, test_loader, device='cuda:0', top_k=2, num_classes=7):
    # region TRAINING
    model.to(device)

    total_acc = []
    total_top_k_acc = []
    result_pred = []
    result_anno = []

    model.eval()
    with torch.no_grad():
        with tqdm(test_loader, unit='batch') as test_epoch:
            for val_inputs, val_targets in test_epoch:                    
                input_conv, input_resnet = val_inputs[0].to(device), val_inputs[1].to(device)
                val_targets = val_targets.to(device)
                val_outputs = model(input_resnet, input_conv)

                val_acc = utils.calculate_accuracy(val_outputs, val_targets)
                val_top_k_acc = utils.calculate_top_k_accuracy(val_outputs, val_targets, k=top_k)
                result_pred.append(val_outputs.cpu().detach().numpy())
                result_anno.append(val_targets.cpu().detach().numpy())

                total_acc.append(val_acc)
                total_top_k_acc.append(val_top_k_acc)
                test_epoch.set_description(f'Evaluating...->')

    total_acc_mean = np.mean(total_acc)

    print(f'Acc {total_acc_mean:.3f}')
    utils.print_report_and_confusion_matrix(result_pred, result_anno, num_classes)

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

    #TODO: 더 좋게 학습한 모델로 교체
    convnet = Conv1dNetwork(out_channel=7, seq_len=60000, kernel_size=64, fc_size=512)
    convnet_savepoint = './savepoints/PPG_KEYWORD__K_64_SV_False_L_512_TS_60000_TOPK_2_FOR_MM/PPG_KEYWORD__K_64_SV_False_L_512_TS_60000_TOPK_2_FOR_MM_20_61.8.pth'
    convnet.load_state_dict(torch.load(convnet_savepoint))

    resnet_model = generate_model(18, n_classes=7)
    resnet_model.load_state_dict(torch.load("savepoints/gesture_savepoints/gesture_200_72.5.pth"))

    model = MultimodalEmbeddingNet(resnet3d=resnet_model, conv1dnet=convnet, fc_size=args.fc_size)

    DM = MultimodalDataManager(batch_size=args.batch_size, aespa_dataset_path="./ppgs_sep")
    TRAIN_DATA, TEST_DATA = DM.load_dataset()
    TRAIN_LOADER, TEST_LOADER = DM.load_dataloader(TRAIN_DATA, TEST_DATA)

    eval(model=model, test_loader=TEST_LOADER, device=DEVICE)
