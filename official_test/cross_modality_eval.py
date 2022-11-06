import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import argparse
from tqdm import tqdm
import torch
import utilities.utils as utils
import numpy as np
from models.ResNet3D import *
from models.gesture_dataset import *
from models.aespa_dataset import *
from models.Conv1DNet import Conv1dNetwork
from models.MultimodalDataset import MultimodalDataManager


def eval(hrv_model, gesture_model, test_loader, num_classes, device='cuda:0'):
    hrv_model.to(device)
    hrv_model.eval()
    gesture_model.to(device)
    gesture_model.eval()

    total_loss = []
    total_acc = []
    result_pred = []
    result_anno = []

    # with torch.no_grad():
    # heatmap 계산을 위해서 no_grad는 끈다.
    with tqdm(test_loader, unit='batch') as test_epoch:
        for i, (val_inputs, val_targets) in enumerate(test_epoch):
            input_conv, input_resnet = val_inputs[0].to(device), val_inputs[1].to(device)
            val_targets = val_targets.to(device)
            val_outputs_hrv = hrv_model(input_conv)
            val_outputs_gesture = gesture_model(input_resnet)
            
            # Confidence 중 가장 큰 값들만 추린다. 이러면 두 model의 output 중 더 큰 confidence로 추론한 결과가
            # 최종 결과가 된다.
            val_maximum = np.maximum(val_outputs_hrv, val_outputs_gesture)

            val_acc = utils.calculate_accuracy(val_maximum, val_targets)
            result_pred.append(val_maximum.cpu().detach().numpy())
            result_anno.append(val_targets.cpu().detach().numpy())

            total_acc.append(val_acc)
            test_epoch.set_description(f'Evaluating...->')

    total_loss_mean = np.mean(total_loss)
    total_acc_mean = np.mean(total_acc)
    print(f'loss {total_loss_mean:.3f} | Acc {total_acc_mean:.3f}')
    utils.print_report_and_confusion_matrix(result_pred, result_anno, num_classes)


if __name__ == "__main__":
    GPU_NUM = 0
    IS_CUDA = torch.cuda.is_available()
    DEVICE = torch.device('cuda:' + str(GPU_NUM) if IS_CUDA else 'cpu')
    parser = argparse.ArgumentParser(description='Multimodal arg parser')

    # 입력받을 인자값 설정 (default 값 설정가능)
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--top_k', type=int, default=2)

    args = parser.parse_args()

    #TODO: 더 좋게 학습한 모델로 교체
    convnet = Conv1dNetwork(out_channel=7, seq_len=60000, kernel_size=64, fc_size=512)
    convnet_savepoint = 'official_test/PPG_KEYWORD__K_64_SV_False_L_512_TS_60000_TOPK_2_FOR_MM_200_76.5.pth'
    convnet.load_state_dict(torch.load(convnet_savepoint))

    resnet_model = generate_model(18, n_classes=7)
    resnet_model.load_state_dict(torch.load("official_test/gesture_60.2.pth"))

    DM = MultimodalDataManager(batch_size=args.batch_size, aespa_dataset_path="./ppgs_sep")
    TRAIN_DATA, TEST_DATA = DM.load_dataset()
    TRAIN_LOADER, TEST_LOADER = DM.load_dataloader(TRAIN_DATA, TEST_DATA)

    eval(hrv_model=convnet, gesture_model=resnet_model, test_loader=TEST_LOADER, num_classes=7, device=DEVICE)
