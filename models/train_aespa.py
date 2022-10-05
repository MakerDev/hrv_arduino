import argparse
import matplotlib.pyplot as plt
from torch import autograd
import sklearn.metrics as metrics
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
from models.aespa_dataset import *
import utils
import numpy as np
import sys
import os
from models.conv1d_net import Conv1dNetwork

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

'''
https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82
https://coderzcolumn.com/tutorials/artificial-intelligence/pytorch-grad-cam
'''
class Conv1DNetForGradCAM(nn.Module):
    def __init__(self, pretrained_model_path, n_classes=7):
        super(Conv1DNetForGradCAM, self).__init__()

        self.conv1dnet = Conv1dNetwork(out_channel=n_classes)
        self.conv1dnet.load_state_dict(torch.load(pretrained_model_path))
        self.conv_net = self.conv1dnet.conv_net[:3]
        self.classifier = self.conv1dnet.fc_net
        self.conv_layer_output = None

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv_net(x)
        self.conv_layer_output = x

        x = self.conv1dnet.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


def get_cam_heatmap(pred, model: Conv1DNetForGradCAM):
    grads = autograd.grad(
        pred[:, pred.argmax().item()], model.conv_layer_output)
    pooled_grads = grads[0].mean((0, 2))
    conv_output = model.conv_layer_output.squeeze()
    conv_output = F.relu(conv_output)
    for i in range(len(pooled_grads)):
        conv_output[i, :] *= pooled_grads[i]

    heatmap = conv_output.mean(dim=0).squeeze()
    # Normalize heatmap
    #heatmap = F.relu(heatmap) / torch.max(heatmap)
    heatmap = heatmap / torch.max(heatmap)
    heatmap = heatmap.detach()

    return heatmap


def save_cam_heatmap(filename, pred, model, x):
    heatmap = get_cam_heatmap(pred, model)
    x = x.cpu().detach()
    heatmap = heatmap.cpu().detach()
    plt.figure(figsize=(28, 4))
    plt.imshow(np.expand_dims(heatmap, axis=0), cmap='Reds', aspect="auto",
               interpolation='nearest', extent=[0, 60000, x.min(), x.max()], alpha=0.5)
    plt.plot(x.squeeze(), 'k')
    plt.colorbar()
    plt.savefig(filename)


def print_report_and_confusion_matrix(result_pred, result_anno, num_classes):
    result_pred = np.array(result_pred).reshape(-1, num_classes)
    result_pred = [np.argmax(pred) for pred in result_pred]
    result_pred_np = np.array(result_pred).reshape(-1, 1).squeeze()
    result_anno_np = np.array(result_anno).reshape(-1, 1).squeeze()

    conf_mat = metrics.confusion_matrix(result_anno_np, result_pred_np)
    print(metrics.classification_report(result_anno_np, result_pred_np, zero_division=0))
    print(conf_mat)


def train(model_name, model, train_loader, test_loader, num_classes, savepoint_dir, epoch=200, device='cuda:0'):
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
        with tqdm(train_loader, unit='batch') as train_epoch:
            for i, (inputs, targets) in enumerate(train_epoch):
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)

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

        model.eval()
        with torch.no_grad():
            with tqdm(test_loader, unit='batch') as test_epoch:
                for val_inputs, val_targets in test_epoch:
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

        writer.add_scalar('validation loss', total_loss_mean, epoch)
        writer.add_scalar('validation accuracy', total_acc_mean, epoch)

        print(f'loss {total_loss_mean:.3f} | Acc {total_acc_mean:.3f}')
        print_report_and_confusion_matrix(result_pred, result_anno, num_classes)

        if epoch in [10, 20, 30, 50, 70, 85, 100, 120, 150, 170, 200, 250, 500]:
            torch.save(model.state_dict(), os.path.join(savepoint_dir, f"{model_name}_{epoch}_{total_acc_mean*100:.1f}.pth"))
        elif epoch >= 100 and total_acc_mean > best_acc:
            torch.save(model.state_dict(), os.path.join(savepoint_dir, f"{model_name}_{epoch}_{total_acc_mean*100:.1f}.pth"))
        
        best_acc = max(total_acc_mean, best_acc)
        print(f'Best so far: {best_acc*100:.1f}%\n')

    writer.close()
    torch.save(model.state_dict(), os.path.join(savepoint_dir, f"{model_name}_{epoch}_{total_acc_mean*100:.1f}.pth"))


def eval(model, test_loader, num_classes, cam_save_dir=None, device='cuda:0'):
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
            
            if cam_save_dir != None:
                filepath = os.path.join(cam_save_dir, f'{i}_pred_{val_outputs.argmax().item()}_label_{val_targets.item()}.png')
                save_cam_heatmap(filepath, val_outputs, model, val_inputs)

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
    print_report_and_confusion_matrix(result_pred, result_anno, num_classes)


if __name__ == "__main__":
    # 인자값을 받을 수 있는 인스턴스 생성
    parser = argparse.ArgumentParser(description='Argparse Tutorial')

    # 입력받을 인자값 설정 (default 값 설정가능)
    parser.add_argument('--epoch', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_classes', type=int, default=7)
    parser.add_argument('--as_hrv', type=bool, default=True)
    parser.add_argument('--as_sam', type=bool, default=False)
    parser.add_argument('--interpolation', type=bool, default=True)
    parser.add_argument('--kernel_size', type=int, default=64)
    parser.add_argument('--fc_size', type=int, default=256)
    parser.add_argument('--target_seq_len', type=int, default=40000)

    args = parser.parse_args()

    # TODO: 쉘로 파라미터 다 받아서 실행 할 수 있도록 구성하기.
    GPU_NUM = 0
    IS_CUDA = torch.cuda.is_available()
    DEVICE = torch.device('cuda:' + str(GPU_NUM) if IS_CUDA else 'cpu')

    MODEL_NAME = f"{'HRV' if args.as_hrv else 'PPG'}_ \
                    {'KEYWORD' if args.as_sam else 'SAM'}_ \
                    {'INTERP' if args.interpolation else ''}_ \
                    K_{args.kernel_size}_\
                    L_{args.fc_size}_ \
                    TS_{args.target_seq_len}"

    DM = AESPADataManager("./ppgs_sep", batch_size=args.batch_size)
    TRAIN_DATA, TEST_DATA = DM.load_dataset(
        as_hrv=args.as_hrv,
        as_sam=args.as_sam,
        interpolation=args.interpolation,
        target_seq_len=args.target_seq_len)
    TRAIN_LOADER, TEST_LOADER = DM.load_dataloader(TRAIN_DATA, TEST_DATA)

    model = Conv1dNetwork(out_channel=args.num_classes, seq_len=args.target_seq_len, 
                            kernel_size=args.kernel_size, fc_size=args.fc_size)

    savepoint_dir = f'savepoints/{MODEL_NAME}'
    os.makedirs(savepoint_dir, exist_ok=True)

    train(model_name=MODEL_NAME, model=model, train_loader=TRAIN_LOADER, test_loader=TEST_LOADER, 
        num_classes=args.num_classes, savepoint_dir=f'savepoints', epoch=args.epoch, device=DEVICE)

    # CAM 때문에 test는 배치 1로함.
    _, TRAIN_LOADER = DM.load_dataloader(TRAIN_DATA, TEST_DATA, batch_size=1)

    #TODO: 가장 잘 된 모델 찾기
    for pth in os.listdir(savepoint_dir):
        if f'{args.epoch}' in pth:
            pretrained_model_path = os.path.join(savepoint_dir, pth)
            break

    model_cam = Conv1DNetForGradCAM(pretrained_model_path=pretrained_model_path, n_classes=args.num_classes)
    cam_save_dir = os.path.join('hrv_pictures', MODEL_NAME)
    os.makedirs(cam_save_dir, exist_ok=True)

    eval(model=model_cam, test_loader=TEST_LOADER, num_classes=args.num_classes, cam_save_dir=cam_save_dir, device=DEVICE)
