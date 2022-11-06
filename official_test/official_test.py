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

def eval_cross_modality_model(hrv_model, gesture_model, test_loader, num_classes, device='cuda:0'):
    hrv_model.to(device)
    hrv_model.eval()
    gesture_model.to(device)
    gesture_model.eval()

    total_acc = []
    result_pred = []
    result_anno = []

    with tqdm(test_loader, unit='batch') as test_epoch:
        for i, (val_inputs, val_targets) in enumerate(test_epoch):
            input_conv, input_resnet = val_inputs[0].to(device), val_inputs[1].to(device)
            val_targets = val_targets.to(device)
            val_outputs_hrv = hrv_model(input_conv)
            val_outputs_gesture = gesture_model(input_resnet)
            
            # Confidence 중 가장 큰 값들만 추린다. 이러면 두 model의 output 중 더 큰 confidence로 추론한 결과가
            # 최종 결과가 된다.
            val_maximum = torch.Tensor(np.maximum(val_outputs_hrv.cpu().detach().numpy(), val_outputs_gesture.cpu().detach().numpy()))
            val_targets = val_targets.cpu().detach()
            val_acc = utils.calculate_accuracy(val_maximum, val_targets)
            result_pred.append(val_maximum.cpu().detach().numpy())
            result_anno.append(val_targets.cpu().detach().numpy())

            total_acc.append(val_acc)
            test_epoch.set_description(f'Evaluating...->')

    total_acc_mean = np.mean(total_acc)
    print(f'Acc {total_acc_mean:.3f}')
    utils.print_report_and_confusion_matrix(result_pred, result_anno, num_classes)


def eval_single_model(model, test_loader, num_classes, device='cuda:0'):
    # region TRAINING
    criterion = nn.CrossEntropyLoss().to(device)

    model.to(device)
    model.eval()

    total_loss = []
    total_acc = []
    result_pred = []
    result_anno = []

    # with torch.no_grad():
    # heatmap 계산을 위해서 no_grad는 끈다.
    with tqdm(test_loader, unit='batch') as test_epoch:
        for i, (val_inputs, val_targets) in enumerate(test_epoch):
            val_inputs = val_inputs.to(device)
            val_targets = val_targets.to(device)
            val_outputs = model(val_inputs)
            
            val_loss = criterion(val_outputs, val_targets)
            val_acc = utils.calculate_accuracy(val_outputs, val_targets)
            result_pred.append(val_outputs.cpu().detach().numpy())
            result_anno.append(val_targets.cpu().detach().numpy())

            total_loss.append(val_loss.cpu().detach().numpy())
            total_acc.append(val_acc)
            test_epoch.set_description(f'Evaluating...->')

    total_loss_mean = np.mean(total_loss)
    total_acc_mean = np.mean(total_acc)
    print(f'loss {total_loss_mean:.3f} | Acc {total_acc_mean:.3f}')
    utils.print_report_and_confusion_matrix(result_pred, result_anno, num_classes)

def eval_multimodal(model, test_loader, device='cuda:0', top_k=2, num_classes=7):
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

    convnet_model = Conv1dNetwork(out_channel=7, seq_len=60000, kernel_size=64, fc_size=512)
    convnet_savepoint = './official_test/PPG_KEYWORD__K_64_SV_False_L_512_TS_60000_TOPK_2_FOR_MM_200_76.5.pth'
    convnet_model.load_state_dict(torch.load(convnet_savepoint))

    resnet_model = generate_model(18, n_classes=7)
    resnet_model.load_state_dict(torch.load("official_test/gesture_60.2.pth"))


    DM = MultimodalDataManager(batch_size=args.batch_size, aespa_dataset_path="./ppgs_sep")
    TRAIN_DATA, TEST_DATA = DM.load_dataset()
    TRAIN_LOADER, MULTIMODAL_TEST_LOADER = DM.load_dataloader(TRAIN_DATA, TEST_DATA)

    TRAIN_AESPA, TEST_AESPA = DM.aespa_train_dataset, DM.aespa_test_dataset
    AESPA_TRAIN_LOADER, AESPA_TEST_LOADER = DM.load_dataloader(TRAIN_AESPA, TEST_AESPA)

    TRAIN_GESTRUE, TEST_GESTURE = DM.gesture_train_dataset, DM.gesture_test_dataset
    GESTURE_TRAIN_LOADER, GESTURE_TEST_LOADER = DM.load_dataloader(TRAIN_GESTRUE, TEST_GESTURE)

    #TODO: 좀 멋있게 꾸미기.
    print('Start evaluating HRV emotion classification.')
    eval_single_model(convnet_model, AESPA_TEST_LOADER, num_classes=7, device=DEVICE)
    print('Start evaluating Gesture emotion classification.')
    eval_single_model(resnet_model, GESTURE_TEST_LOADER, num_classes=7, device=DEVICE)

    # Reload models
    convnet_model = Conv1dNetwork(out_channel=7, seq_len=60000, kernel_size=64, fc_size=512)
    convnet_savepoint = './savepoints/PPG_KEYWORD__K_64_SV_False_L_512_TS_60000_TOPK_2_FOR_MM/PPG_KEYWORD__K_64_SV_False_L_512_TS_60000_TOPK_2_FOR_MM_20_61.8.pth'
    convnet_model.load_state_dict(torch.load(convnet_savepoint))

    resnet_model = generate_model(18, n_classes=7)
    resnet_model.load_state_dict(torch.load("savepoints/gesture_savepoints/gesture_200_72.5.pth"))

    model = MultimodalEmbeddingNet(resnet3d=resnet_model, conv1dnet=convnet_model, fc_size=args.fc_size)
    model.load_state_dict(torch.load('official_test/Multimodal_FC_512_2_61.9.pth'))

    print('Start evaluating Multi modality model on HRV and gesture')
    eval_multimodal(model=model, test_loader=MULTIMODAL_TEST_LOADER, device=DEVICE)

    print('Start evaluating cross modality model on HRV and gesture')
    eval_cross_modality_model(hrv_model=convnet_model, gesture_model=resnet_model, test_loader=MULTIMODAL_TEST_LOADER, num_classes=7, device=DEVICE)
