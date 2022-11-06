import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import argparse
from tqdm import tqdm
import torch.nn as nn
import torch
import utilities.utils as utils
import numpy as np

from models.aespa_dataset import *
from models.Conv1DNet import Conv1dNetwork


def eval(model, test_loader, num_classes, device='cuda:0'):
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


if __name__ == "__main__":
    # 인자값을 받을 수 있는 인스턴스 생성
    parser = argparse.ArgumentParser(description='AESPA arg parser')

    # 입력받을 인자값 설정 (default 값 설정가능)
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--as_hrv', type=bool, default=False)
    parser.add_argument('--as_sam', type=bool, default=False)
    parser.add_argument('--interpolation', type=bool, default=False)
    parser.add_argument('--use_survey', type=bool, default=False)
    parser.add_argument('--kernel_size', type=int, default=64)
    parser.add_argument('--fc_size', type=int, default=512)
    parser.add_argument('--target_seq_len', type=int, default=60000)
    parser.add_argument('--top_k', type=int, default=2)

    args = parser.parse_args()

    # TODO: 쉘로 파라미터 다 받아서 실행 할 수 있도록 구성하기.
    GPU_NUM = 0
    IS_CUDA = torch.cuda.is_available()
    DEVICE = torch.device('cuda:' + str(GPU_NUM) if IS_CUDA else 'cpu')

    COMMENT    = '_FOR_MM'
    MODEL_NAME = f"{'HRV' if args.as_hrv else 'PPG'}_" + \
                 f"{'SAM' if args.as_sam else 'KEYWORD'}_" + \
                 f"{'INTERP' if args.interpolation else ''}_" + \
                 f"K_{args.kernel_size}_" + \
                 f"SV_{args.use_survey}_" + \
                 f"L_{args.fc_size}_" + \
                 f"TS_{args.target_seq_len}_" + \
                 f"TOPK_{args.top_k}" + COMMENT

    print(f'{MODEL_NAME} test start')

    DM = AESPADataManager("./ppgs_sep", batch_size=args.batch_size)
    TRAIN_DATA, TEST_DATA = DM.load_dataset(
        as_hrv=args.as_hrv,
        as_sam=args.as_sam,
        interpolation=args.interpolation,
        target_seq_len=args.target_seq_len,
        use_survey=args.use_survey)
    TRAIN_LOADER, TEST_LOADER = DM.load_dataloader(TRAIN_DATA, TEST_DATA)

    if args.as_sam:
        num_classes = 5
    else:
        num_classes = 7

    model = Conv1dNetwork(out_channel=num_classes, seq_len=args.target_seq_len, 
                            kernel_size=args.kernel_size, fc_size=args.fc_size)

    _, TRAIN_LOADER = DM.load_dataloader(TRAIN_DATA, TEST_DATA, batch_size=4)
    #TODO : Load pretrained model
    pretrained_model_path = r'official_test\\PPG_KEYWORD__K_64_SV_False_L_512_TS_60000_TOPK_2_FOR_MM_200_76.5.pth'
    model.load_state_dict(torch.load(pretrained_model_path))

    eval(model=model, test_loader=TEST_LOADER, num_classes=num_classes, device=DEVICE)
