#得到数据以及训练验证测试的文件
#对图片内数据进行打乱

import os
import pandas as pd
import torch

from transforms import transforms
from utils.autoaugment import ImageNetPolicy

# pretrained model checkpoints
pretrained_model = {'resnet50' : './models/pretrained/rgb_imagenet.pt',}
# pretrained_model = {'resnet50' : './net_model/_5119_UCF101/weights_21_35447_0.6790_0.8242.pth',}


# transforms dict
def load_data_transformers(clip_length=16,resize_reso=512, crop_reso=448, swap_num=[5, 5]):
    center_resize = 600
    Normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])# 归一化。前面是mean，后面三个是std标准差
    # 数据减去均值，除以标准差
    data_transforms = {
       	'swap': transforms.Compose([
            transforms.Randomswap((swap_num[0], swap_num[1])),
        ]),
        'common_aug': transforms.Compose([
            transforms.Resize((resize_reso, resize_reso)),
            transforms.RandomRotation(degrees=15), # 按（-degrees，+degrees)的度数随机翻转
            transforms.RandomCrop((crop_reso,crop_reso)),
            transforms.RandomHorizontalFlip(),
        ]),
        'train_totensor': transforms.Compose([
            transforms.Resize((crop_reso, crop_reso)),
            # transforms.CenterCrop((crop_reso, crop_reso)),   #添加的
            # ImageNetPolicy(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'val_totensor': transforms.Compose([
            transforms.Resize((crop_reso, crop_reso)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'test_totensor': transforms.Compose([
            transforms.Resize((resize_reso, resize_reso)),
            transforms.CenterCrop((crop_reso, crop_reso)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'None': None,
    }
    return data_transforms


class LoadConfig(object): # 选择训练或者测试，读进相应数据集
    def __init__(self, args, version):
        if version == 'train':
            get_list = ['train', 'val']
        elif version == 'val':
            get_list = ['val']
        elif version == 'test':
            get_list = ['test']
        else:
            raise Exception("train/val/test ???\n")

        ###############################
        #### add dataset info here ####
        ###############################

        # put image data in $PATH/data
        # put annotation txt file in $PATH/anno

        if args.dataset == 'UCF101':
            self.dataset = args.dataset
            self.rawdata_root = '/media/alice/datafile/data/UCF-101_optical'
            self.anno_root = './list'
            self.numcls = 101
        elif args.dataset == 'HMDB51':
            self.dataset = args.dataset
            self.rawdata_root = '/home/lili/文档/hmdb_rgb+flow'
            self.anno_root = './list'
            self.numcls = 51
        elif args.dataset == 'STCAR':
            self.dataset = args.dataset
            self.rawdata_root = './dataset/st_car/data'
            self.anno_root = './dataset/st_car/anno'
            self.numcls = 196
        elif args.dataset == 'AIR':
            self.dataset = args.dataset
            self.rawdata_root = './dataset/aircraft/data'
            self.anno_root = './dataset/aircraft/anno'
            self.numcls = 100
        else:
            raise Exception('dataset not defined ???')

        # annotation file organized as :
        # path/image_name cls_num\n

        if 'train' in get_list:
             self.train_anno = pd.read_csv(os.path.join(self.anno_root, 'change_hmdb51_train_03.txt'),\
                                           sep=" ",\
                                           header=None,\
                                           names=['ImageName', 'start_frame','gap', 'label'])
            # sep指定分隔符，pd.read_csv将csv文件读入并转化为数据框形式

        if 'val' in get_list:
            self.val_anno = pd.read_csv(os.path.join(self.anno_root, 'change_hmdb51_test_03.txt'),\
                                           sep=" ",\
                                           header=None,\
                                           names=['ImageName', 'start_frame','gap', 'label'])

        if 'test' in get_list:
            self.test_anno = pd.read_csv(os.path.join(self.anno_root, 'change_hmdb51_test_03.txt'),\
                                           sep=" ",\
                                           header=None,\
                                           names=['ImageName', 'start_frame','gap', 'label'])

        self.swap_num = args.swap_num

        self.save_dir = './net_model'
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.backbone = args.backbone

        self.use_dcl = True
        self.use_backbone = False if self.use_dcl else True
        self.use_Asoftmax = False
        self.use_focal_loss = False
        self.use_fpn = False
        self.use_hier = False

        self.weighted_sample = False
        self.cls_2 = True
        self.cls_2xmul = False

        self.log_folder = './logs'
        if not os.path.exists(self.log_folder):
            os.mkdir(self.log_folder)




