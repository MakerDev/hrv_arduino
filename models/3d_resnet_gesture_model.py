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



def sf(float_input):
    scientific_output = np.format_float_scientific(float_input, trim='-', precision=0, exp_digits=2)
    return scientific_output

# region Settings
BATCH_SIZE  = 4
NUM_CLASSES = 7

''''''''''''''''''''''''''''''''''''
'''          Need to tune        '''
''''''''''''''''''''''''''''''''''''
MODEL_DEPTH    = 10
INPUT_RATIO    = 0.25 # Multiply 0.25 to both widht and height.

LEARNING_RATE  = 0.000001 # L1
L2WEIGHT_DECAY = 0.000001  # L2
DATASET_NAME   = 'front'
''''''''''''''''''''''''''''''''''''

SAMPLING_RATE = 180
NUM_VARIABLES = 5
EPOCHS        = 100
GPU_NUM       = 0

IS_CUDA = torch.cuda.is_available()
#DEVICE  = 'cpu'
DEVICE  = torch.device('cuda:' + str(GPU_NUM) if IS_CUDA else 'cpu')

ModelName     = "3D_RESNET_V1"
SaveModelName = ModelName + "_B" + str(BATCH_SIZE) + "_T" + str(MODEL_DEPTH) + "_IN" + str(INPUT_RATIO) + \
                "_L1_" + str(sf(LEARNING_RATE)) +"_L2_" + str(sf(L2WEIGHT_DECAY)) + "_" + DATASET_NAME

DATASET_DIR = "./Datasets_front"
print("MODEL NAME: ", SaveModelName)
# endregion

# region 3D ResNet
def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=1,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=7):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
# endregion

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
    DM = GestureDataManager(BATCH_SIZE)
    TRAIN_GESTURE_DATA, TEST_GESTURE_DATA = DM.Load_Dataset()
    TRAIN_LOADER, TEST_LOADER     = DM.Load_DataLoader(TRAIN_GESTURE_DATA, TEST_GESTURE_DATA)
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    MODEL = generate_model(18, n_classes=7)

    LOAD_MODEL = True
    if LOAD_MODEL:
        MODEL.load_state_dict(torch.load("savepoints/savepoint_100.pth"))
        
    MODEL.to(DEVICE)
        
    optimizer = optim.Adam(MODEL.parameters(), lr = LEARNING_RATE, weight_decay = L2WEIGHT_DECAY)

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

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_epoch.set_description(f'Training epoch {epoch}->')            

        MODEL.eval()

        with torch.no_grad():
            with tqdm(TEST_LOADER, unit='batch') as test_epoch:
                for val_inputs, val_targets in test_epoch:
                    val_inputs = val_inputs.to(DEVICE)
                    val_targets = val_targets.to(DEVICE)
                    val_outputs = MODEL(val_inputs)

                    val_loss = criterion(val_outputs, val_targets)
                    val_acc = utils.calculate_accuracy(val_outputs, val_targets)          
                    
                    total_loss.append(val_loss.cpu().detach().numpy())
                    total_acc.append(val_acc)
                    test_epoch.set_description(f'Evaluating...->')

        total_loss_mean = np.mean(total_loss)
        total_acc_mean = np.mean(total_acc)
        print(f'loss {total_loss_mean:.3f} | Acc {total_acc_mean:.3f}\n')
            
        if epoch in [10, 20, 50, 100, 200, 250, 500]:
            torch.save(MODEL.state_dict(), f"savepoints/savepoint_{epoch}.pth")
        
    torch.save(MODEL.state_dict(), f"savepoints/savepoint_{epoch}.pth")

    # evaluation
    # DM = GestureDataManager(DATASET_DIR, BATCH_SIZE)
    # TRAIN_GESTURE_DATA, TEST_GESTURE_DATA = DM.Load_Dataset()
    # TRAIN_LOADER, TEST_LOADER     = DM.Load_DataLoader(TRAIN_GESTURE_DATA, TEST_GESTURE_DATA)
    # criterion = nn.CrossEntropyLoss().to(DEVICE)

    # MODEL = generate_model(34, n_classes=4)

    # LOAD_MODEL = True
    # if LOAD_MODEL:
    #     MODEL.load_state_dict(torch.load("savepoints/savepoint_1000.pth"))

    # MODEL.to(DEVICE)
    # MODEL.eval()

    # with torch.no_grad():
    #     with tqdm(TEST_LOADER, unit='batch') as test_epoch:
    #         for val_inputs, val_targets in test_epoch:
    #             val_inputs = val_inputs.to(DEVICE)
    #             val_targets = val_targets.to(DEVICE)
    #             val_outputs = MODEL(val_inputs)

    #             val_loss = criterion(val_outputs, val_targets)
    #             val_acc = utils.calculate_accuracy(val_outputs, val_targets)          
                
    #             total_loss.append(val_loss.cpu().detach().numpy())
    #             total_acc.append(val_acc)
    #             test_epoch.set_description(f'Evaluating...->')

    # total_loss_mean = np.mean(total_loss)
    # total_acc_mean = np.mean(total_acc)
    # print(f'loss {total_loss_mean:.3f} | Acc {total_acc_mean:.3f}\n')
    
    # Result_pred_np = np.array(Result_pred).reshape(-1, 1)[:REAL_SAMPLE_LEN]
    # Result_anno_np = np.array(Result_anno).reshape(-1, 1)[:REAL_SAMPLE_LEN]
    # print('--------------------------------------------------------------')
    # ACC_TEST = metrics.accuracy_score(Result_anno_np, Result_pred_np)
    # print('Accuracy: ', ACC_TEST)
    # conf_mat = metrics.confusion_matrix(Result_anno_np, Result_pred_np)
    # print(metrics.classification_report(Result_anno_np, Result_pred_np))
    # print(conf_mat)