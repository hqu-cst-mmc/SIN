# coding=utf8
from __future__ import division
import os
import torch
import torch.utils.data as data
import pandas
import random
import PIL.Image as Image
from PIL import ImageStat

import numpy as np
# from skimage import io

import pdb

clip_length = 16

def random_sample(img_names,start_frame, gap, labels):
    anno_dict = {}
    img_list = []
    start_list = []
    gap_list = []
    anno_list = []
    for img,start,gap_frame, anno in zip(img_names,start_frame, gap, labels):
        if not anno in anno_dict:
            anno_dict[anno] = [img + " " + str(start) + " " + str(gap_frame)]
        else:
            anno_dict[anno].append(img+ " " + str(start)+ " " + str(gap_frame))

    for anno in anno_dict.keys():
        anno_len = len(anno_dict[anno])
        fetch_keys = random.sample(list(range(anno_len)), anno_len//10)

        for x in fetch_keys:
            img_ = str(anno_dict[anno][x]).split(" ")
            img_list.append(img_[0])
            start_list.append(int(img_[1]))
            gap_list.append(int(img_[2]))
            anno_list.append(anno)
        # anno_list.extend([anno for x in fetch_keys])
    return img_list,start_list,gap_list, anno_list



class dataset(data.Dataset):
    def __init__(self, Config, anno, swap_size=[5,5], common_aug=None, swap=None, totensor=None, train=False, train_val=False, test=False):
        self.root_path = Config.rawdata_root
        self.numcls = Config.numcls
        self.dataset = Config.dataset
        self.use_cls_2 = Config.cls_2
        self.use_cls_mul = Config.cls_2xmul
        if isinstance(anno, pandas.core.frame.DataFrame):
            self.paths = anno['ImageName'].tolist()
            self.start_frame = anno['start_frame'].tolist()
            self.gap = anno['gap'].tolist()
            self.labels = anno['label'].tolist()
        elif isinstance(anno, dict):
            self.paths = anno['img_name']
            self.start_frame = anno['start_frame']
            self.gap = anno['gap'].tolist()
            self.labels = anno['label']

        if train_val:
            self.paths,self.start_frame,self.gap, self.labels = random_sample(self.paths,self.start_frame, self.gap, self.labels)

        self.common_aug = common_aug  # 随机裁剪和随机翻转
        self.swap = swap  # 交换图片内切分好的顺序，里面有随机调换，看要怎么改，还没改
        self.totensor = totensor
        self.cfg = Config
        self.train = train
        self.swap_size = swap_size
        self.test = test

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item): # 返回数据
        # print(item)
        path_change = self.paths[item].split("/")
        # video_path = os.path.join(self.root_path, self.paths[item])# video_path里面包含视频切分好的帧
        video_path = os.path.join(self.root_path, path_change[0])
        # print(video_path,item)
        video_img = self.get_frame_in_video(video_path, item) # 从video_path里面选取连续的16帧
        # print("end")
        # img = self.pil_loader(img_path)
        if self.test:
            video_img = self.totensor(video_img)
            label = self.labels[item]
            return video_img, label, self.paths[item]
        video_img_unswap = self.common_aug(video_img) if not self.common_aug is None else video_img

        video_unswap_list = self.crop_image(video_img_unswap, self.swap_size)  #将视频中的每一帧按同种格式切分成7*7（swap_size）

        swap_range = self.swap_size[0] * self.swap_size[1]
        # swap_law1 = []
        # for i in range(clip_length):
        swap_law1 = [(i-(swap_range//2))/swap_range for i in range(swap_range)]
            # // 得数取整
            # swap_law1.append(swap_law_1)

        if self.train:
            video_swap = self.swap(video_img_unswap)  # 这里swap包括了图像内切分7*7
            video_swap_list = self.crop_image(video_swap, self.swap_size)
            unswap_stats = []
            swap_stats = []
            for frame in range(clip_length):
                unswap_stats.append([sum(ImageStat.Stat(im).mean) for im in video_unswap_list[frame]])
                swap_stats.append([sum(ImageStat.Stat(im).mean) for im in video_swap_list[frame]])
            # unswap_stats = [sum(ImageStat.Stat(im).mean) for im in image_unswap_list]
            # swap_stats = [sum(ImageStat.Stat(im).mean) for im in image_swap_list]
            swap_law2 = []

            # for frame in range(clip_length):
            #     swap_law_2 = []
            mean_unw = []
            mean_w = []
            for i in range(swap_range):
                sum_unw = 0.0
                for j in range(clip_length):
                    w = unswap_stats[j][i]
                    sum_unw = sum_unw + w
                mean_unw.append(sum_unw / clip_length)

            for i in range(swap_range):
                sum_w = 0.0
                for j in range(clip_length):
                    w = swap_stats[j][i]
                    sum_w = sum_w + w
                mean_w.append(sum_w / clip_length)

            for swap_im in mean_w:
                distance = [abs(swap_im - unswap_im) for unswap_im in mean_unw]
                index = distance.index(min(distance))
                swap_law2.append((index-(swap_range//2))/swap_range)
            # swap_law2.append(swap_law_2)
            video_swap = self.totensor(video_swap)
            label = self.labels[item]
            if self.use_cls_mul:
                label_swap = label + self.numcls
            if self.use_cls_2:
                label_swap = -1
            video_img_unswap = self.totensor(video_img_unswap)
            # return video_img_unswap, video_swap, label, label_swap, swap_law2, self.paths[item]
            return video_img_unswap, video_swap, label, label_swap, swap_law1, swap_law2, self.paths[item]
        else:
            label = self.labels[item]
            swap_law2 = [(i-(swap_range//2))/swap_range for i in range(swap_range)]
            label_swap = label
            img_unswap = self.totensor(video_img_unswap)
            return img_unswap, label, label_swap, swap_law1, swap_law2, self.paths[item]

    def get_frame_in_video(self, video_path,item):
        # slash_rows = video_path.split('.')
        # dir_name = slash_rows[0]
        # video_jpgs_path = os.path.join(self.root_dir, dir_name)

        # get the random 16 frame
        # data = pd.read_csv(os.path.join(video_jpgs_path, 'n_frames'), delimiter=' ', header=None)
        #frame_count = open(video_path, "r")
        #frame_count = len(list(frame_count))  # 获得总的帧数
        # frame_count = data[0][0]
        video_x = []
        #image_start = random.randint(1, frame_count - clip_length - 1)
        image_id = self.start_frame[item]
        gap_frame = self.gap[item]
        for i in range(clip_length):
            # s = "%05d" % image_id
            # image_name = 'image_' + s + '.jpg'
            # image_name = "img_" + "{:05}.jpg".format(str(image_id + i))
            image_path = os.path.join(video_path, "img_" + "{:05}.jpg".format(image_id + i*gap_frame))
            # print(image_path)
            if os.path.exists(image_path):
                tmp_image = self.pil_loader(image_path)
            video_x.append(tmp_image)
        return video_x

    def pil_loader(self,imgpath):
        with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def crop_image(self, video, cropnum):
        width, high = video[0].size
        crop_x = [int((width / cropnum[0]) * i) for i in range(cropnum[0] + 1)]
        crop_y = [int((high / cropnum[1]) * i) for i in range(cropnum[1] + 1)]
        video_list = []
        for k, frame in enumerate(video):
            im_list = []
            for j in range(len(crop_y) - 1):
                for i in range(len(crop_x) - 1):
                    im_list.append(video[k].crop((crop_x[i], crop_y[j], min(crop_x[i + 1], width), min(crop_y[j + 1], high))))
            video_list.append(im_list)
        return video_list


    def get_weighted_sampler(self):
        img_nums = len(self.labels)
        weights = [self.labels.count(x) for x in range(self.numcls)]
        return torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples=img_nums)


def collate_fn4train(batch):
    imgs = []
    label = []
    label_swap = []
    law_swap = []
    img_name = []
    for sample in batch:
        imgs.append(sample[0])
        imgs.append(sample[1])
        label.append(sample[2])
        label.append(sample[2])
        if sample[3] == -1:
            label_swap.append(1)
            label_swap.append(0)
        else:
            label_swap.append(sample[2])
            label_swap.append(sample[3])
        law_swap.append(sample[4])
        law_swap.append(sample[5])
        img_name.append(sample[-1])
    return torch.stack(imgs, 0), label, label_swap, law_swap, img_name
# torch.stack()按维度堆叠
def collate_fn4val(batch):
    imgs = []
    label = []
    label_swap = []
    law_swap = []
    img_name = []
    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[1])
        if sample[3] == -1:
            label_swap.append(1)
        else:
            label_swap.append(sample[2])
        law_swap.append(sample[3])
        img_name.append(sample[-1])
    return torch.stack(imgs, 0), label, label_swap, law_swap, img_name

def collate_fn4backbone(batch):
    imgs = []
    label = []
    img_name = []
    for sample in batch:
        imgs.append(sample[0])
        if len(sample) == 7:
            label.append(sample[2])
        else:
            label.append(sample[1])
        img_name.append(sample[-1])
    return torch.stack(imgs, 0), label, img_name


def collate_fn4test(batch):
    imgs = []
    label = []
    img_name = []
    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[1])
        img_name.append(sample[-1])
    return torch.stack(imgs, 0), label, img_name
