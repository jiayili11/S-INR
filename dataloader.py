import random
import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from skimage.segmentation import slic, mark_boundaries
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from sklearn.cluster import KMeans
from scipy import interpolate
from torch.distributions import Normal
import os
from sklearn.neighbors import KNeighborsRegressor


# 处理单个RGB图像, 得到分块好的gt，完整的图像, 对应的坐标(int, float)
class data_load_img(Dataset):
    def __init__(self, filename_gt, pixel_set=2000, freq=5, compactness=20):  # pixel_set 预估的每个superpixel的像素点的个数
        super().__init__()
        mat = scipy.io.loadmat(filename_gt + '.mat')
        self.img = mat["Ohsi"]
        self.img_shape = self.img.shape
        n_segments = int(self.img_shape[0] * self.img_shape[1] / pixel_set)
        inpainted_data = np.zeros((self.img.shape[0], self.img.shape[1], 2 + self.img.shape[2]))
        inpainted_data[:, :, 2:] = self.img
        for ind in range(self.img_shape[0] * self.img_shape[1]):
            j = ind % self.img_shape[1]
            i = (ind // self.img_shape[1]) % self.img_shape[0]
            inpainted_data[i, j, 0] = 5 * i / 255.
            inpainted_data[i, j, 1] = 5 * j / 255.

        inpainted_data = inpainted_data.reshape(self.img.shape[0] * self.img.shape[1], 2 + self.img.shape[2])
        kmeans = KMeans(n_clusters=n_segments, init='k-means++')
        kmeans.fit(inpainted_data)
        labels = kmeans.labels_[:]
        labels = labels.reshape(self.img.shape[0], self.img.shape[1])
        self.labels = labels
        # self.segments = slic(self.img, n_segments=n_segments, compactness=compactness)
        self.segments = labels
        self.pixel_num = self.segments.max()
        # 下面是inr的做法，需要提取每个superpixel对应的坐标和RGB值
        self.count = torch.zeros((self.pixel_num, 1))  # 记录每个superpixel中像素点个数
        self.coords1 = list()  # 每个超像素中像素的整数坐标
        self.coords2 = list()  # 每个超像素中像素的浮点数坐标
        self.gt = list()  # 每个超像素的 RGB 值
        self.coordsf = list()
        for n in range(self.pixel_num):
            temp1 = list()
            temp2 = list()
            temp3 = list()
            temp4 = list()
            self.coords1.append(temp1)
            self.coords2.append(temp2)
            self.gt.append(temp3)
            self.coordsf.append(temp4)
        for ind in range(self.img_shape[0] * self.img_shape[1]):
            j = ind % self.img_shape[1]
            i = (ind // self.img_shape[1]) % self.img_shape[0]
            self.gt[self.segments[i, j] - 1].append(self.img[i, j, :])
            self.coords1[self.segments[i, j] - 1].append([i, j])
            self.coords2[self.segments[i, j] - 1].append([i / (self.img_shape[0] - 1), j / (self.img_shape[1] - 1)])
            self.count[self.segments[i, j] - 1, 0] += 1
        for n in range(self.pixel_num):
            self.gt[n] = torch.from_numpy(np.array(self.gt[n], dtype=np.float32)).cuda().float()
            self.coords1[n] = torch.from_numpy(np.array(self.coords1[n], dtype=np.int32)).cuda()
            self.coords2[n] = torch.from_numpy(np.array(self.coords2[n], dtype=np.float32)).cuda().float()
            self.count[n] = self.count[n].squeeze().cuda()
            coordf2 = (2 ** (freq / (freq - 1)) ** torch.linspace(0., freq - 1, steps=freq)).cuda() * torch.unsqueeze(
                self.coords2[n], -1)
            self.coordsf[n] = torch.cat((torch.cos(coordf2), torch.sin(coordf2)), dim=-1).reshape(coordf2.size(0),
                                                                                                  2 * coordf2.size(
                                                                                                      1) * coordf2.size(
                                                                                                      2)).cuda().float()
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {'gt': self.gt, 'img': self.img, 'coords_int': self.coords1, 'coords_float': self.coords2,
                'pixel_num': self.pixel_num, 'pixel_count': self.count, 'poen_coords': self.coordsf,
                "labels": self.labels}



# 函数:获取扩展后块坐标范围
def get_expanded_coords(coords1, expand_pixels, img_shape):
    x_min, y_min = min(coords1, key=lambda x: x[0])[0], min(coords1, key=lambda x: x[1])[1]
    x_max, y_max = max(coords1, key=lambda x: x[0])[0], max(coords1, key=lambda x: x[1])[1]

    x_min -= expand_pixels
    x_max += expand_pixels
    y_min -= expand_pixels
    y_max += expand_pixels
    x_min = max(x_min, 0)
    y_min = max(y_min, 0)
    x_max = min(x_max, img_shape[0] - 1)
    y_max = min(y_max, img_shape[1] - 1)

    return x_min, x_max, y_min, y_max

# 利用插值和slic算法
class data_load_img_completion_overlap_bound(Dataset):
    def __init__(self, filename_ob, filename_gt, pixel_set=1000, compactness=50):  # pixel_set 预估的每个superpixel的像素点的个数
        super().__init__()
        mat = scipy.io.loadmat(filename_gt + '.mat')
        self.img_gt = mat["Ohsi"]
        mat = scipy.io.loadmat(filename_ob + '.mat')
        self.img_ob = mat["Nhsi"]
        self.img_shape = self.img_gt.shape
        n_segments = int(self.img_shape[0] * self.img_shape[1] / pixel_set)

        missing_mask = (self.img_ob == 0).all(axis=-1)
        interpolated_image = (self.img_ob.copy() * 255).astype(np.uint8)
        mask = missing_mask.astype(np.uint8) * 255
        inpainted = cv2.inpaint(interpolated_image, mask, 3, cv2.INPAINT_TELEA)
        inpainted = inpainted / 255.
        segment = slic(inpainted, n_segments=n_segments, compactness=compactness)
        self.pixel_num = segment.max()  # superpixel个数
        # 下面是inr的做法，需要提取每个superpixel对应的坐标和RGB值
        self.count = torch.zeros((self.pixel_num, 1))  # 记录每个superpixel中像素点个数
        self.coords1 = list()  # 每个超像素中像素的整数坐标
        self.coords2 = list()  # 每个超像素中像素的浮点数坐标
        self.gt = list()  # 每个超像素的 RGB 值
        self.gt_mask = list()  # 每个超像素的 RGB 值的mask
        self.gt_ob = list()
        self.segments = list()
        SEG_SHAPE = (self.img_shape[0], self.img_shape[1])
        for n in range(self.pixel_num):
            temp1 = list()
            temp2 = list()
            temp3 = list()
            temp4 = list()
            temp5 = list()
            seg = np.zeros(SEG_SHAPE)
            self.coords1.append(temp1)
            self.coords2.append(temp2)
            self.gt.append(temp3)
            self.gt_mask.append(temp4)
            self.gt_ob.append(temp5)
            self.segments.append(seg)
        for ind in range(self.img_shape[0] * self.img_shape[1]):
            j = ind % self.img_shape[1]
            i = (ind // self.img_shape[1]) % self.img_shape[0]
            self.gt[segment[i, j] - 1].append(self.img_gt[i, j, :])
            pixel_value = self.img_ob[i, j, :]
            if np.all(pixel_value == np.zeros_like(pixel_value)):
                self.gt_mask[segment[i, j] - 1].append(list(np.zeros_like(pixel_value)))
            else:
                self.gt_mask[segment[i, j] - 1].append(np.ones_like(pixel_value))
            self.coords1[segment[i, j] - 1].append((i, j))
            self.coords2[segment[i, j] - 1].append((i / (self.img_shape[0] - 1), j / (self.img_shape[1] - 1)))
            self.count[segment[i, j] - 1, 0] += 1
            self.segments[segment[i, j] - 1][i, j] = 1

        expand_pixels = 2  # 外扩像素大小
        for n in range(self.pixel_num):
            mask = self.segments[n]
            mask = mask.astype(np.uint8)
            edges = cv2.Laplacian(mask, cv2.CV_8U)
            # 外扩2像素
            kernel = np.ones((3, 3), np.uint8)
            edges_expanded = cv2.dilate(edges, kernel, iterations=expand_pixels - 1)
            edges_expanded[mask == 1] = 0
            # 遍历新边界所有像素点
            indices = np.where(edges_expanded >= 1)
            coords_edges = list(zip(indices[0], indices[1]))  # 组合成坐标
            for x, y in coords_edges:
                pixel_value = self.img_ob[x, y, :]
                self.coords1[n].append((x, y))
                self.coords2[n].append((x / (self.img_shape[0] - 1), y / (self.img_shape[1] - 1)))
                self.count[n] += 1
                self.gt[n].append(self.img_gt[x, y, :])
                if np.all(pixel_value == np.zeros_like(pixel_value)):
                    self.gt_mask[n].append(list(np.zeros_like(pixel_value)))
                else:
                    self.gt_mask[n].append(np.ones_like(pixel_value))

        for n in range(self.pixel_num):
            self.gt[n] = torch.from_numpy(np.array(self.gt[n], dtype=np.float32)).cuda().float().squeeze()
            self.gt_mask[n] = torch.from_numpy(np.array(self.gt_mask[n], dtype=np.float32)).cuda().float().squeeze()
            self.gt_ob[n] = self.gt_mask[n] * self.gt[n]
            self.coords1[n] = torch.from_numpy(np.array(self.coords1[n], dtype=np.int32)).cuda().squeeze()
            self.coords2[n] = torch.from_numpy(np.array(self.coords2[n], dtype=np.float32)).cuda().float().squeeze()
            self.count[n] = self.count[n].cuda()

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {'gt': self.gt, 'img': self.img_gt, 'coords_int': self.coords1, 'coords_float': self.coords2,
                'pixel_num': self.pixel_num, 'pixel_count': self.count.squeeze(), 'gt_mask': self.gt_mask,
                'gt_ob': self.gt_ob, 'segments': self.segments, 'img_ob': self.img_ob}


class data_load_img_noise_overlap_bound(Dataset):
    def __init__(self, filename_ob, filename_gt, pixel_set=200, freq=None):  # pixel_set 预估的每个superpixel的像素点的个数
        super().__init__()
        mat = scipy.io.loadmat(filename_gt + '.mat')
        self.img_gt = mat["Ohsi"]
        mat = scipy.io.loadmat(filename_ob + '.mat')
        self.img_ob = mat["Nhsi"]
        self.img_shape = self.img_gt.shape
        n_segments = int(self.img_shape[0] * self.img_shape[1] / pixel_set)

        inpainted_data = np.zeros((self.img_ob.shape[0], self.img_ob.shape[1], self.img_shape[2]+2))
        inpainted_data[:, :, 2:] = self.img_ob
        for ind in range(self.img_shape[0] * self.img_shape[1]):
            j = ind % self.img_shape[1]
            i = (ind // self.img_shape[1]) % self.img_shape[0]
            inpainted_data[i, j, 0] = 10 * i / 255.
            inpainted_data[i, j, 1] = 10 * j / 255.
        inpainted_data = inpainted_data.reshape(self.img_ob.shape[0] * self.img_ob.shape[1], self.img_shape[2]+2)
        kmeans = KMeans(n_clusters=n_segments, init='k-means++')
        kmeans.fit(inpainted_data)  # 分类
        labels = kmeans.labels_[:]
        labels = labels.reshape(self.img_ob.shape[0], self.img_ob.shape[1])
        segment = labels
        self.pixel_num = segment.max()
        print("img_shape", self.img_shape, "n_segments", n_segments, "pixel_num", self.pixel_num)
        # 下面是inr的做法，需要提取每个superpixel对应的坐标和RGB值
        self.count = torch.zeros((self.pixel_num, 1))  # 记录每个superpixel中像素点个数
        self.coords1 = list()  # 每个超像素中像素的整数坐标
        self.coords2 = list()  # 每个超像素中像素的浮点数坐标
        self.gt = list()  # 每个超像素的 RGB 值
        self.gt_ob = list()
        self.segments = list()
        self.coordsf = list()
        SEG_SHAPE = (self.img_shape[0], self.img_shape[1])
        for n in range(self.pixel_num):
            temp1 = list()
            temp2 = list()
            temp3 = list()
            temp4 = list()
            temp5 = list()
            seg = np.zeros(SEG_SHAPE)
            self.coords1.append(temp1)
            self.coords2.append(temp2)
            self.gt.append(temp3)
            self.gt_ob.append(temp4)
            self.segments.append(seg)
            self.coordsf.append(temp5)
        for ind in range(self.img_shape[0] * self.img_shape[1]):
            j = ind % self.img_shape[1]
            i = (ind // self.img_shape[1]) % self.img_shape[0]
            self.gt[segment[i, j] - 1].append(self.img_gt[i, j, :])
            self.gt_ob[segment[i, j] - 1].append(self.img_ob[i, j, :])
            self.coords1[segment[i, j] - 1].append((i, j))
            self.coords2[segment[i, j] - 1].append((i / (self.img_shape[0] - 1), j / (self.img_shape[1] - 1)))
            self.count[segment[i, j] - 1, 0] += 1
            self.segments[segment[i, j] - 1][i, j] = 1
        expand_pixels = 2  # 外扩像素大小
        for n in range(self.pixel_num):
            mask = self.segments[n]
            mask = mask.astype(np.uint8)
            edges = cv2.Laplacian(mask, cv2.CV_8U)
            # 外扩2像素
            kernel = np.ones((3, 3), np.uint8)
            edges_expanded = cv2.dilate(edges, kernel, iterations=expand_pixels - 1)
            edges_expanded[mask == 1] = 0
            # 遍历新边界所有像素点
            indices = np.where(edges_expanded >= 1)
            coords_edges = list(zip(indices[0], indices[1]))  # 组合成坐标
            for x, y in coords_edges:
                self.coords1[n].append((x, y))
                self.coords2[n].append((x / (self.img_shape[0] - 1), y / (self.img_shape[1] - 1)))
                self.count[n] += 1
                self.gt[n].append(self.img_gt[x, y, :])
                self.gt_ob[n].append(self.img_ob[x, y, :])

            self.gt[n] = torch.from_numpy(np.array(self.gt[n], dtype=np.float32)).squeeze().cuda()
            self.gt_ob[n] = torch.from_numpy(np.array(self.gt_ob[n], dtype=np.float32)).squeeze().cuda()
            self.coords1[n] = torch.from_numpy(np.array(self.coords1[n], dtype=np.int32)).squeeze().cuda()
            self.coords2[n] = torch.from_numpy(np.array(self.coords2[n], dtype=np.float32)).squeeze().cuda()
            self.count[n] = self.count[n].cuda()
            coordf2 = (2 ** (freq / (freq - 1)) ** torch.linspace(0., freq - 1, steps=freq)).cuda() * torch.unsqueeze(
                self.coords2[n], -1)
            self.coordsf[n] = torch.cat((torch.cos(coordf2), torch.sin(coordf2)), dim=-1).reshape(coordf2.size(0),
                                                                                                  2 * coordf2.size(
                                                                                                      1) * coordf2.size(
                                                                                                      2))
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {'gt': self.gt, 'img': self.img_gt, 'coords_int': self.coords1, 'coords_float': self.coords2,
                'pixel_num': self.pixel_num, 'pixel_count': self.count.squeeze(), 'gt_ob': self.gt_ob,
                'segments': self.segments, 'img_ob': self.img_ob}


class data_load_img_completion_overlap_bound_kmeans(Dataset):
    def __init__(self, filename_ob, filename_gt, filename_mask, pixel_set=1000, freq=4):  # pixel_set 预估的每个superpixel的像素点的个数
        super().__init__()
        mat = scipy.io.loadmat(filename_gt + '.mat')
        self.img_gt = mat["Ohsi"]
        mat = scipy.io.loadmat(filename_ob + '.mat')
        self.img_ob = mat["Nhsi"]
        self.img_shape = self.img_gt.shape
        n_segments = int(self.img_shape[0] * self.img_shape[1] / pixel_set)
        mat = scipy.io.loadmat(filename_mask + '.mat')
        self.img_mask = mat["Mask"]
        # 分别对选定的通道进行修复 按着间距选
        inpainted_channels = []
        for channel in [10, 20, 30]:
            knr = KNeighborsRegressor(n_neighbors=3)
            channel_input = self.img_ob[:, :, channel]
            missing_mask = (self.img_ob[:, :, channel] == 0)
            coords = np.argwhere(~missing_mask)
            values = channel_input[~missing_mask]
            knr.fit(coords, values)
            predicted_missing = knr.predict(np.argwhere(missing_mask))
            inpainted_channel = channel_input.copy()
            inpainted_channel[missing_mask] = predicted_missing
            inpainted_channels.append(inpainted_channel)
        # 将修复后的通道重新组合成一个包含坐标的量
        inpainted = np.stack(inpainted_channels, axis=-1)
        inpainted_data = np.zeros((inpainted.shape[0], inpainted.shape[1], 5))
        inpainted_data[:, :, 2:] = inpainted
        for ind in range(self.img_shape[0] * self.img_shape[1]):
            j = ind % self.img_shape[1]
            i = (ind // self.img_shape[1]) % self.img_shape[0]
            inpainted_data[i, j, 0] = 5.0 * i / self.img_shape[0]
            inpainted_data[i, j, 1] = 5.0 * j / self.img_shape[1]
        kmeans = KMeans(n_clusters=n_segments, init='k-means++')
        inpainted_data = inpainted_data.reshape(inpainted.shape[0] * inpainted.shape[1], 5)

        kmeans.fit(inpainted_data)
        labels = kmeans.labels_[:]
        labels = labels.reshape(inpainted.shape[0], inpainted.shape[1])
        segment = labels
        self.pixel_num = segment.max()
        # 下面是inr的做法，需要提取每个superpixel对应的坐标和RGB值
        self.count = torch.zeros((self.pixel_num, 1))  # 记录每个superpixel中像素点个数
        self.coords1 = list()  # 每个超像素中像素的整数坐标
        self.coords2 = list()  # 每个超像素中像素的浮点数坐标
        self.gt = list()  # 每个超像素的 RGB 值
        self.gt_mask = list()  # 每个超像素的 RGB 值的mask
        self.gt_ob = list()
        self.segments = list()
        self.coordsf = list()
        SEG_SHAPE = (self.img_shape[0], self.img_shape[1])
        for n in range(self.pixel_num):
            temp1 = list()
            temp2 = list()
            temp3 = list()
            temp4 = list()
            temp5 = list()
            temp6 = list()
            seg = np.zeros(SEG_SHAPE)
            self.coords1.append(temp1)
            self.coords2.append(temp2)
            self.gt.append(temp3)
            self.gt_mask.append(temp4)
            self.gt_ob.append(temp5)
            self.segments.append(seg)
            self.coordsf.append(temp6)
        for ind in range(self.img_shape[0] * self.img_shape[1]):
            j = ind % self.img_shape[1]
            i = (ind // self.img_shape[1]) % self.img_shape[0]
            self.gt[segment[i, j] - 1].append(self.img_gt[i, j, :])
            self.gt_mask[segment[i, j] - 1].append(self.img_mask[i, j, :])
            self.gt_ob[segment[i, j] - 1].append(self.img_ob[i, j, :])
            self.coords1[segment[i, j] - 1].append((i, j))
            self.coords2[segment[i, j] - 1].append((i / (self.img_shape[0] - 1), j / (self.img_shape[1] - 1)))
            self.count[segment[i, j] - 1, 0] += 1
            self.segments[segment[i, j] - 1][i, j] = 1

        expand_pixels = 2  # 外扩像素大小
        for n in range(self.pixel_num):
            mask = self.segments[n]
            mask = mask.astype(np.uint8)
            edges = cv2.Laplacian(mask, cv2.CV_8U)
            # 外扩2像素
            kernel = np.ones((3, 3), np.uint8)
            edges_expanded = cv2.dilate(edges, kernel, iterations=expand_pixels - 1)
            edges_expanded[mask == 1] = 0
            # 遍历新边界所有像素点
            indices = np.where(edges_expanded >= 1)
            coords_edges = list(zip(indices[0], indices[1]))  # 组合成坐标
            for x, y in coords_edges:
                pixel_value = self.img_ob[x, y, :]
                self.coords1[n].append((x, y))
                self.coords2[n].append((x / (self.img_shape[0] - 1), y / (self.img_shape[1] - 1)))
                self.count[n] += 1
                self.gt[n].append(self.img_gt[x, y, :])
                if np.all(pixel_value == np.zeros_like(pixel_value)):
                    self.gt_mask[n].append(list(np.zeros_like(pixel_value)))
                else:
                    self.gt_mask[n].append(np.ones_like(pixel_value))

            self.gt[n] = torch.from_numpy(np.array(self.gt[n], dtype=np.float32)).cuda().float().squeeze()
            self.gt_mask[n] = torch.from_numpy(np.array(self.gt_mask[n], dtype=np.float32)).cuda().float().squeeze()
            self.gt_ob[n] = torch.from_numpy(np.array(self.gt_ob[n], dtype=np.float32)).cuda().float().squeeze()
            self.coords1[n] = torch.from_numpy(np.array(self.coords1[n], dtype=np.int32)).cuda().squeeze()
            self.coords2[n] = torch.from_numpy(np.array(self.coords2[n], dtype=np.float32)).cuda().float().squeeze()
            self.count[n] = self.count[n].cuda()
            coordf2 = (2 ** (freq / (freq - 1)) ** torch.linspace(0., freq - 1, steps=freq)).cuda() * torch.unsqueeze(
                self.coords2[n], -1)
            self.coordsf[n] = torch.cat((torch.cos(coordf2), torch.sin(coordf2)), dim=-1).reshape(coordf2.size(0),
                                                                                                  2 * coordf2.size(
                                                                                                      1) * coordf2.size(
                                                                                                      2))
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {'gt': self.gt, 'img': self.img_gt, 'coords_int': self.coords1, 'coords_float': self.coords2,
                'pixel_num': self.pixel_num, 'pixel_count': self.count.squeeze(), 'gt_mask': self.gt_mask,
                'gt_ob': self.gt_ob, 'segments': self.segments, 'img_ob': self.img_ob}
