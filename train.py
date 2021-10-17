#coding=utf-8
import os
import datetime
import argparse
import logging
import pandas as pd

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.utils.data as torchdata
from torchvision import datasets, models
import torch.optim as optim
from torch.optim import lr_scheduler #根据epoch训练次数来调整学习率（learning rate）的方法
import torch.backends.cudnn as cudnn #那么cuDNN使用的非确定性算法就会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题
#如果网络的输入数据维度或类型上变化不大，设置  torch.backends.cudnn.benchmark = true  可以增加运行效率；
#如果网络的输入数据在每次 iteration 都变化的话，会导致 cnDNN 每次都会去寻找一遍最优配置，这样反而会降低运行效率。
from transforms import transforms
from utils.train_model import train
from models.LoadModel import MainModel
from config import LoadConfig, load_data_transformers
from utils.dataset_DCL import collate_fn4train, collate_fn4val, collate_fn4test, collate_fn4backbone, dataset

import pdb

os.environ['CUDA_DEVICE_ORDRE'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# parameters setting
def parse_args():
    parser = argparse.ArgumentParser(description='dcl parameters')
    parser.add_argument('--data', dest='dataset',
                        default='HMDB51', type=str)

    # parser.add_argument('--save', dest='resume',
    #                    default='./net_model/_11810_HMDB51/weights_59_4620_0.7428_0.8849.pth',type=str)
    parser.add_argument('--save', dest='resume',
                         default=None, type=str)
    parser.add_argument('--backbone', dest='backbone',
                        default='resnet50', type=str)
    parser.add_argument('--auto_resume', dest='auto_resume',
                        action='store_true')
    parser.add_argument('--epoch', dest='epoch',
                        default=150, type=int)
    parser.add_argument('--tb', dest='train_batch',
                        default=6, type=int)
    parser.add_argument('--vb', dest='val_batch',
                        default=6, type=int)
    parser.add_argument('--sp', dest='save_point',
                        default=100000, type=int)
    parser.add_argument('--cp', dest='check_point',
                        default=100000, type=int)
    parser.add_argument('--lr', dest='base_lr',
                        default=0.001, type=float)
    parser.add_argument('--lr_step', dest='decay_step',
                        default=30, type=int)
    parser.add_argument('--cls_lr_ratio', dest='cls_lr_ratio',
                        default=10.0, type=float)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        default=1,  type=int)
    parser.add_argument('--tnw', dest='train_num_workers',
                        default=5, type=int)
    parser.add_argument('--vnw', dest='val_num_workers',
                        default=10, type=int)
    parser.add_argument('--detail', dest='discribe',
                        default='', type=str)
    parser.add_argument('--size', dest='resize_resolution',
                        default=512, type=int)
    parser.add_argument('--crop', dest='crop_resolution',
                        default=448, type=int)
    parser.add_argument('--cls_2', dest='cls_2',
                        action='store_true')# store_true只要运行时该变量有传参就将该变量设为True
    parser.add_argument('--cls_mul', dest='cls_mul',
                        action='store_true')
    parser.add_argument('--swap_num', default=[5, 5],
                    nargs=2, metavar=('swap1', 'swap2'),
                    type=int, help='specify a range')
    parser.add_argument('--clip_length', dest='clip_length',
                        default=16, type=int)
    args = parser.parse_args()
    return args

# 返回了一个路径，导入模型路径
def auto_load_resume(load_dir):
    folders = os.listdir(load_dir) #os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
    date_list = [int(x.split('_')[1].replace(' ',0)) for x in folders]
    choosed = folders[date_list.index(max(date_list))] # max返回给定参数的最大值,index()方法检测字符串中是否包含子字符串 str,返回起始位置
    weight_list = os.listdir(os.path.join(load_dir, choosed))
    acc_list = [x[:-4].split('_')[-1] if x[:7]=='weights' else 0 for x in weight_list]
    acc_list = [float(x) for x in acc_list]
    choosed_w = weight_list[acc_list.index(max(acc_list))]
    return os.path.join(load_dir, choosed, choosed_w)


if __name__ == '__main__':
    args = parse_args()
    print(args, flush=True) #使用flush=True之后，会在print结束之后，不管你有没有达到条件，立即将内存中的东西显示到屏幕上，清空缓存
    Config = LoadConfig(args, 'train') #train是模式选择，看得到何种数据训练验证测试，这个函数得到数据以及数据文件
    # Config.cls_2 = args.cls_2
    # Config.cls_2xmul = args.cls_mul
    assert Config.cls_2 ^ Config.cls_2xmul  # 断言，满足条件时执行，表达式条件为 false 的时候触发异常，^ 异或逻辑运算符
    # 异或，值不同为1，值相同为0
    transformers = load_data_transformers(args.clip_length,args.resize_resolution, args.crop_resolution, args.swap_num)#调整尺寸，图片切割打乱

    # inital dataloader
    train_set = dataset(Config = Config,\
                        anno = Config.train_anno,\
                        common_aug = transformers["common_aug"],\
                        swap = transformers["swap"],\
                        totensor = transformers["train_totensor"],\
                        train = True)

    trainval_set = dataset(Config = Config,\
                        anno = Config.train_anno,\
                        common_aug = transformers["None"],\
                        swap = transformers["None"],\
                        totensor = transformers["val_totensor"],\
                        train = False,
                        train_val = True)

    val_set = dataset(Config = Config,\
                      anno = Config.val_anno,\
                      common_aug = transformers["None"],\
                      swap = transformers["None"],\
                      totensor = transformers["test_totensor"],\
                      test=True)

    dataloader = {}
    dataloader['train'] = torch.utils.data.DataLoader(train_set,\
                                                batch_size=args.train_batch,\
                                                shuffle=True,\
                                                num_workers=args.train_num_workers,\
                                                collate_fn=collate_fn4train if not Config.use_backbone else collate_fn4backbone,
                                                drop_last=True if Config.use_backbone else False,
                                                pin_memory=True)
# num_works:使用多进程加载的的进程数，0代表不使用多进程
# collate_fn:如何将多个样本数据拼接成一个batch，也会将空对象过滤掉包括损坏的图片
# drop_last:会将多出来的不足一个batch数量的数据丢弃
# pin_memory: 是否将数据保存在pin memory区，pin memory中的数据转到gpu快一些
    setattr(dataloader['train'], 'total_item_len', len(train_set))#setattr(object, name, value)设置属性值

    dataloader['trainval'] = torch.utils.data.DataLoader(trainval_set,\
                                                batch_size=args.val_batch,\
                                                shuffle=False,\
                                                num_workers=args.val_num_workers,\
                                                collate_fn=collate_fn4val if not Config.use_backbone else collate_fn4backbone,
                                                drop_last=True if Config.use_backbone else False,
                                                pin_memory=True)

    setattr(dataloader['trainval'], 'total_item_len', len(trainval_set))
    setattr(dataloader['trainval'], 'num_cls', Config.numcls)

    dataloader['val'] = torch.utils.data.DataLoader(val_set,\
                                                batch_size=args.val_batch,\
                                                shuffle=False,\
                                                num_workers=args.val_num_workers,\
                                                collate_fn=collate_fn4test if not Config.use_backbone else collate_fn4backbone,
                                                drop_last=True if Config.use_backbone else False,
                                                pin_memory=True)

    setattr(dataloader['val'], 'total_item_len', len(val_set))
    setattr(dataloader['val'], 'num_cls', Config.numcls)


    cudnn.benchmark = True

    print('Choose model and train set', flush=True)
    model = MainModel(Config)

    # load model
    if (args.resume is None) and (not args.auto_resume):
        print('train from imagenet pretrained models ...', flush=True)
    else:
        if not args.resume is None:
            resume = args.resume
            print('load from pretrained checkpoint %s ...'% resume, flush=True)
        elif args.auto_resume:
            resume = auto_load_resume(Config.save_dir)
            print('load from %s ...'%resume, flush=True)
        else:
            raise Exception("no checkpoints to load")

        model_dict = model.state_dict()   # 名称加值，以字典的方式
        pretrained_dict = torch.load(resume,map_location={'cuda:1': 'cuda:0'})
        # print(pretrained_dict.items())
        # for i in model_dict.keys():
        #     if  i[:7] == 'model.7':
        #         print(i)
        # for k, v in pretrained_dict.items():
        #     print(k[7:14])

        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
        # print(pretrained_dict.keys())
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    print('Set cache dir', flush=True)
    time = datetime.datetime.now()
    filename = '%s_%d%d%d_%s'%(args.discribe, time.month, time.day, time.hour, Config.dataset)
    save_dir = os.path.join(Config.save_dir, filename)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.cuda() #将模型复制到GPU
    model = nn.DataParallel(model)  # 多GPU训练
    print(model)
    # optimizer prepare
    if Config.use_backbone:
        # for name, param in model.module.model.named_parameters():
        #     print(name,param.requires_grad)
        ignored_params1 = list(map(id, model.module.model[7].parameters()))  # layer4
        ignored_params2 = list(map(id, model.module.model[8].parameters()))  # pool
        # ignored_params4 = list(map(id, model.module.fc.parameters()))  # 额外加的fc
        ignored_params3 = list(map(id, model.module.classifier.parameters()))  # 新加的分类层
                # map()会根据提供的函数对指定序列做映射,python2返回列表，3返回迭代器
        ignored_params = ignored_params1 + ignored_params2 + ignored_params3
    else:
        ignored_params1 = list(map(id, model.module.classifier.parameters()))
        ignored_params2 = list(map(id, model.module.classifier_swap.parameters()))
        ignored_params3 = list(map(id, model.module.Convmask.parameters()))
        ignored_params4 = list(map(id, model.module.model[15].parameters()))
        ignored_params5 = list(map(id, model.module.model[14].parameters()))
        ignored_params6 = list(map(id, model.module.model[12].parameters()))
        ignored_params7 = list(map(id, model.module.model[11].parameters()))

        # ignored_params6 = list(map(id, model.module.avgpool3.parameters()))

        ignored_params = ignored_params1 + ignored_params2 + ignored_params3 + ignored_params4 + ignored_params5 + ignored_params6 + ignored_params7
    print('the num of new layers:', len(ignored_params), flush=True)
    base_params = filter(lambda p: id(p) not in ignored_params, model.module.parameters())   # 保持参数不变的部分
    finetune_params =  filter(lambda p: id(p) in ignored_params, model.module.parameters())  # 参数要进行微调的部分
    # filter() 函数用于过滤序列，过滤掉不符合条件的元素，返回由符合条件元素组成的新列表
    lr_ratio = args.cls_lr_ratio
    base_lr = args.base_lr
    if Config.use_backbone:
        optimizer = optim.SGD([{'params': base_params, 'lr': 0},
                               {'params': finetune_params, 'lr': base_lr}], lr = base_lr, momentum=0.9, weight_decay=1e-5)
    else:
        # optimizer = optim.SGD([{'params': base_params, 'lr': 0},
        #                        {'params': model.module.classifier.parameters(), 'lr': lr_ratio*base_lr},
        #                        {'params': model.module.classifier_swap.parameters()},
        #                        {'params': model.module.Convmask.parameters()},
        #                       ], lr = base_lr, momentum=0.9, weight_decay=1e-5)
        optimizer = optim.SGD([{'params': base_params, 'lr': base_lr},
                               {'params': finetune_params, 'lr': base_lr*lr_ratio}], lr=base_lr, momentum=0.9, weight_decay=1e-5)


    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=0.1)  # 学习率在间隔decay_step,变为gamma×lr

    # train entry
    train(Config,
          model,
          epoch_num=args.epoch,
          start_epoch=args.start_epoch,
          optimizer=optimizer,
          exp_lr_scheduler=exp_lr_scheduler,
          data_loader=dataloader,
          save_dir=save_dir,
          data_size=args.crop_resolution,
          savepoint=args.save_point,
          checkpoint=args.check_point)


