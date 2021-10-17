import numpy as np
from torch import nn
import torch
from torchvision import models, transforms, datasets
import torch.nn.functional as F
import pretrainedmodels

from config import pretrained_model
from models import resnet
from models import pytorch_i3d

import pdb

class MainModel(nn.Module):
    def __init__(self, config):
        super(MainModel, self).__init__()
        self.use_dcl = config.use_dcl
        self.num_classes = config.numcls
        self.backbone_arch = config.backbone
        self.use_Asoftmax = config.use_Asoftmax
        print(self.backbone_arch)
        print(dir(models))

        if self.backbone_arch in dir(models):
            # self.model = getattr(models, self.backbone_arch)()
            # self.model = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], resnet.get_inplanes()) # [3,4,6,4]是resnet50网络结构中每一层的block数量, get_inplanes是特征通道的数量，也就是卷积核的数量
            # model_dict = self.model.state_dict()
            # print(self.model.name_parameters())
            self.model = pytorch_i3d.InceptionI3d(400,in_channels=3)
            # self.model.replace_logits(51)

            if self.backbone_arch in pretrained_model:
                # pretrain = torch.load(pretrained_model[self.backbone_arch])
                # pretrain = {k: v for k, v in pretrain['state_dict'].items() if k in self.model.state_dict() and k[:6] == 'layer4' or k[:2] == 'fc'}
                # model_dict.update(pretrain)
                # self.model.load_state_dict(model_dict)
                # pretrain = torch.load(pretrained_model[self.backbone_arch])
                # self.model.load_state_dict(pretrain['state_dict'])
                self.model.load_state_dict(torch.load('./models/pretrained/rgb_imagenet.pt'))

        else:
            if self.backbone_arch in pretrained_model:
                self.model = pretrainedmodels.__dict__[self.backbone_arch](num_classes=51, pretrained=None)
            else:
                self.model = pretrainedmodels.__dict__[self.backbone_arch](num_classes=101)

        if self.backbone_arch == 'resnet50' or self.backbone_arch == 'se_resnet50':
            self.model = nn.Sequential(*list(self.model.children())[3:])
        if self.backbone_arch == 'senet154':
            self.model = nn.Sequential(*list(self.model.children())[:-3])
        if self.backbone_arch == 'se_resnext101_32x4d':
            self.model = nn.Sequential(*list(self.model.children())[:-2])
        if self.backbone_arch == 'se_resnet101':
            self.model = nn.Sequential(*list(self.model.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool3d(output_size=1)
        # self.fc = nn.Linear(2048, 2048, bias=False)
        self.classifier = nn.Linear(1024, self.num_classes, bias=False)

        if self.use_dcl:
            if config.cls_2:
                self.classifier_swap = nn.Linear(1024, 2, bias=False)
            if config.cls_2xmul:
                self.classifier_swap = nn.Linear(1024, 2*self.num_classes, bias=False)
            self.Convmask = nn.Conv3d(1024, 1, 1, stride=1, padding=0, bias=True)
            self.avgpool3 = nn.AvgPool3d((1,3,3), stride=1)
            self.maxpool3 = nn.MaxPool3d((1,5,5), stride=2)

        if self.use_Asoftmax:
            self.Aclassifier = AngleLinear(2048, self.num_classes, bias=False)

    def forward(self, x, last_cont=None):
        x = self.model(x)
        if self.use_dcl:
            mask = self.Convmask(x)
            mask = self.maxpool3(mask)
            # mask = self.avgpool3(mask)
            # mask = torch.tanh(mask)
            mask = torch.sigmoid(mask)
            # mask = torch.relu(mask)    #还不到10步很快就变成0.0000
            mask = mask.view(mask.size(0), -1)
            # mask = F.softmax(mask,dim=1)    #通过softmax做一个归一化


        x = self.avgpool(x)
        x = x.view(x.size(0), -1) # view相当于reshape,（行，列),负数表示不确定
        out = []
        out.append(self.classifier(x))
        # out = x

        if self.use_dcl:
            out.append(self.classifier_swap(x))
            out.append(mask)
            # out.append(x)

        if self.use_Asoftmax:
            if last_cont is None:
                x_size = x.size(0)
                out.append(self.Aclassifier(x[0:x_size:2]))
            else:
                last_x = self.model(last_cont)
                last_x = self.avgpool(last_x)
                last_x = last_x.view(last_x.size(0), -1)
                out.append(self.Aclassifier(last_x))

        return out
