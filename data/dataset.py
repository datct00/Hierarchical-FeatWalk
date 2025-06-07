# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import json
import numpy as np
import torchvision.transforms as transforms
import os

identity = lambda x: x


def get_grid_location(size, ratio, num_grid):
    '''

    :param size: size of the height/width
    :param ratio: generate grid size/ even divided grid size
    :param num_grid: number of grid
    :return: a list containing the coordinate of the grid
    '''
    raw_grid_size = int(size / num_grid)
    enlarged_grid_size = int(size / num_grid * ratio)

    center_location = raw_grid_size // 2

    location_list = []
    for i in range(num_grid):
        location_list.append((max(0, center_location - enlarged_grid_size // 2),
                              min(size, center_location + enlarged_grid_size // 2)))
        center_location = center_location + raw_grid_size

    return location_list


class SimpleDataset:
    def __init__(self, data_path, data_file_list, transform, target_transform=identity):
        label = []
        data = []
        k = 0
        data_dir_list = data_file_list.replace(" ","").split(',')
        for data_file in data_dir_list:
            img_dir = data_path + '/' + data_file
            for i in os.listdir(img_dir):
                file_dir = os.path.join(img_dir, i)
                for j in os.listdir(file_dir):
                    data.append(file_dir + '/' + j)
                    label.append(k)
                k += 1
        self.data = data
        self.label = label
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, i):
        image_path = os.path.join(self.data[i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.label[i] - min(self.label))
        return img, target

    def __len__(self):
        return len(self.label)


class SetDataset:
    def __init__(self, data_path, data_file_list, batch_size, transform,aug_num=0,args=None):
        label = []
        data = []
        k = 0
        data_dir_list = data_file_list.replace(" ","").split(',')
        for data_file in data_dir_list:
            img_dir = data_path + '/' + data_file
            for i in os.listdir(img_dir):
                file_dir = os.path.join(img_dir, i)
                for j in os.listdir(file_dir):
                    data.append(file_dir + '/' + j)
                    label.append(k)
                k += 1
        self.data = data
        self.label = label
        self.transform = transform
        self.cl_list = np.unique(self.label).tolist()
        self.args = args

        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        for x, y in zip(self.data, self.label):
            self.sub_meta[y].append(x)

        self.sub_dataloader = []

        sub_data_loader_params = dict(batch_size=batch_size, # 5 + 15: n_shot + n_queries
                                      shuffle=True,
                                      num_workers=0,  # use main thread only or may receive multiple batches
                                      pin_memory=False)
        self.cl_num = 0
        for cl in self.cl_list:
            if len(self.sub_meta[cl])>=25:
                self.cl_num += 1
                sub_dataset = SubDataset(self.sub_meta[cl], cl, transform=transform,aug_num=aug_num,args=self.args)
                self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params))


    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return self.cl_num


class SubDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity,aug_num=0,args=None):
        self.sub_meta = sub_meta
        self.cl = cl
        self.transform = transform
        self.target_transform = target_transform
        self.aug_num = aug_num
        self.grid = args.grid
        self.transform_grid = transforms.Compose([
            transforms.Resize([args.img_size,args.img_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.472, 0.453, 0.410], std=[0.277, 0.268, 0.285])
        ])

        self.transform_s = transforms.Compose([
            transforms.RandomResizedCrop(args.img_size, scale=(0.4, 0.4)),
            # transforms.RandomResizedCrop(args.img_size, scale=(args.local_scale, args.local_scale)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.472, 0.453, 0.410], std=[0.277, 0.268, 0.285])
        ])
        
        self.transform_s1 = transforms.Compose([
            transforms.RandomResizedCrop(args.img_size, scale=(0.2, 0.2)),
            # transforms.RandomResizedCrop(args.img_size, scale=(args.local_scale, args.local_scale)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.472, 0.453, 0.410], std=[0.277, 0.268, 0.285])
        ])

    def __getitem__(self, i):
        image_path = os.path.join(self.sub_meta[i])
        img = Image.open(image_path).convert('RGB')
        img_set = []
        img_w = self.transform(img)
        img_set.append(img_w.unsqueeze(0))
        if self.grid:
            for num_patch in self.grid:
                patches = self.get_pyramid(img, num_patch)
                # print(patches.shape)
                img_set.append(patches)
        else:
            for _ in range((self.aug_num-1)//2):
                img_s = self.transform_s(img)
                img_set.append(img_s.unsqueeze(0))
                
            for _ in range(self.aug_num - 1):
                img_s1 = self.transform_s1(img)
                img_set.append(img_s1.unsqueeze(0))
                
        # for item in img_set:
        #     print(item.shape)
        img = torch.cat(img_set, dim=0)
        target = self.target_transform(self.cl)
        return img, target

    def get_pyramid(self, img, num_patch):
        num_grid = num_patch
        grid_ratio = 1
        w, h = img.size
        grid_locations_w = get_grid_location(w, grid_ratio, num_grid)
        grid_locations_h = get_grid_location(h, grid_ratio, num_grid)

        patches_list = []
        for i in range(num_grid):
            for j in range(num_grid):
                patch_location_w = grid_locations_w[j]
                patch_location_h = grid_locations_h[i]
                left_up_corner_w = patch_location_w[0]
                left_up_corner_h = patch_location_h[0]
                right_down_cornet_w = patch_location_w[1]
                right_down_cornet_h = patch_location_h[1]
                patch = img.crop((left_up_corner_w, left_up_corner_h, right_down_cornet_w, right_down_cornet_h))
                patch = self.transform_grid(patch)
                patches_list.append(patch.unsqueeze(0))
        return torch.cat(patches_list,dim=0)

    def __len__(self):
        return len(self.sub_meta)


class SimpleDataset_JSON:
    def __init__(self, data_path, data_file, transform, target_transform=identity):
        data = data_path + '/' + data_file
        with open(data, 'r') as f:
            self.meta = json.load(f)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, i):
        image_path = os.path.join(self.meta['image_names'][i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.meta['image_labels'][i])
        return img, target

    def __len__(self):
        return len(self.meta['image_names'])


class SetDataset_JSON:
    def __init__(self, data_path, data_file, batch_size, transform,aug_num=0,args=None):
        data = data_path + '/' + data_file

        print(transform.__dict__)
        with open(data, 'r') as f:
            self.meta = json.load(f)

        self.cl_list = np.unique(self.meta['image_labels']).tolist()
        self.args = args

        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        for x, y in zip(self.meta['image_names'], self.meta['image_labels']):
            self.sub_meta[y].append(x)

        self.sub_dataloader = []

        # print(len(self.cl_list))
        sub_data_loader_params = dict(batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=0,  # use main thread only or may receive multiple batches
                                      pin_memory=False)
        for cl in self.cl_list:
            sub_dataset = SubDataset_JSON(self.sub_meta[cl], cl, transform=transform,aug_num=aug_num,args=self.args)
            self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params))
            

    def __getitem__(self, i):
        data = next(iter(self.sub_dataloader[i]))
        return data

    def __len__(self):
        return len(self.cl_list)


class SubDataset_JSON:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity,aug_num=0,args=None):
        self.sub_meta = sub_meta
        self.cl = cl
        self.transform = transform
        self.target_transform = target_transform
        self.grid = args.grid
        self.transform_grid = transforms.Compose([
            transforms.Resize([args.img_size, args.img_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.472, 0.453, 0.410], std=[0.277, 0.268, 0.285])
        ])

        self.transform_s = transforms.Compose([
            transforms.RandomResizedCrop(args.img_size, scale=(0.4, 0.4)),
            # transforms.RandomResizedCrop(args.img_size, scale=(args.local_scale, args.local_scale)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.472, 0.453, 0.410], std=[0.277, 0.268, 0.285])
        ])
        
        self.transform_s1 = transforms.Compose([
            transforms.RandomResizedCrop(args.img_size, scale=(0.2, 0.2)),
            # transforms.RandomResizedCrop(args.img_size, scale=(args.local_scale, args.local_scale)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.472, 0.453, 0.410], std=[0.277, 0.268, 0.285])
        ])
        # print(aug_num)
        self.aug_num =aug_num

    def __getitem__(self, i):
        # print( '%d -%d' %(self.cl,i))
        image_path = os.path.join(self.sub_meta[i])
        img = Image.open(image_path).convert('RGB')
        img_set = []
        img_w = self.transform(img)
        img_set.append(img_w.unsqueeze(0))
        if self.grid:
            for num_patch in self.grid:
                patches = self.get_pyramid(img, num_patch)
                img_set.append(patches)
        else:
            for _ in range((self.aug_num-1)//2):
                img_s = self.transform_s(img)
                img_set.append(img_s.unsqueeze(0))
                
            for _ in range(self.aug_num - 1):
                img_s1 = self.transform_s1(img)
                img_set.append(img_s1.unsqueeze(0))
        img = torch.cat(img_set,dim=0)
        target = self.target_transform(self.cl)
        return img, target

    def get_pyramid(self, img, num_patch):
        num_grid = num_patch
        grid_ratio = 1
        w, h = img.size
        grid_locations_w = get_grid_location(w, grid_ratio, num_grid)
        grid_locations_h = get_grid_location(h, grid_ratio, num_grid)

        patches_list = []
        for i in range(num_grid):
            for j in range(num_grid):
                patch_location_w = grid_locations_w[j]
                patch_location_h = grid_locations_h[i]
                left_up_corner_w = patch_location_w[0]
                left_up_corner_h = patch_location_h[0]
                right_down_cornet_w = patch_location_w[1]
                right_down_cornet_h = patch_location_h[1]
                patch = img.crop((left_up_corner_w, left_up_corner_h, right_down_cornet_w, right_down_cornet_h))
                patch = self.transform_grid(patch)
                patches_list.append(patch.unsqueeze(0))
        return torch.cat(patches_list, dim=0)


    def __len__(self):
        return len(self.sub_meta)


class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]


