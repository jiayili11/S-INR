import numpy as np
import math
from sklearn.metrics.pairwise import rbf_kernel
import torch
import torch.nn as nn
import random
import os
from datatio import *
import warnings
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from models import *
# import open3d as o3d
from utils import *

warnings.filterwarnings("ignore", category=RuntimeWarning, message="Explicit initial center position passed*")
warnings.filterwarnings("ignore", category=UserWarning, message="MiniBatchKMeans is known to have a memory leak*")

os.environ["OMP_NUM_THREADS"] = "1"
np.random.seed(1)


def get_device():
    if torch.cuda.is_available():
        de = 'cuda:0'
    else:
        de = 'cpu'
    return de


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

device = get_device()
setup_seed(0)


def rbf_k(X, Y, gamma=0.2):
    return rbf_kernel(X, Y, gamma)


def update_cluster_center(data_norm, indices):
    """
    计算指定区域内所有点的平均值，并将该平均值设置为聚类中心。
    """
    center = np.mean(data_norm[indices, :], axis=0)
    return center.reshape(1, -1)

class soft(nn.Module):
    def __init__(self):
        super(soft, self).__init__()

    def forward(self, x, lam):
        x_abs = x.abs() - lam
        zeros = x_abs - x_abs
        n_sub = torch.max(x_abs, zeros)
        x_out = torch.mul(torch.sign(x), n_sub)
        return x_out

def optimize(parameters, closure, LR, num_iter, R=0):
    optimizer = torch.optim.Adam(parameters, lr=LR)
    if R == 0:
        for j in range(num_iter):
            optimizer.zero_grad()
            _, _ = closure(j)
            optimizer.step()
    else:
        for j in range(num_iter):
            optimizer.zero_grad()
            _, _, t = closure(j)
            if t >= 5:
                break
            optimizer.step()


def evaluate_function(data, center):
    """
    计算评价函数
    """
    # 利用高斯核函数计算评价
    data = data.reshape(1, -1)
    center = center.reshape(1, -1)
    evaluation = rbf_k(data, center)[0, 0]
    return evaluation


def row_norms(X, squared=False):
    if len(X.shape) == 1:  # handle 1D arrays
        if squared:
            return X.dot(X)
        else:
            return np.sqrt(X.dot(X))

    if X.dtype == np.float32:
        dtype = np.float32
    else:
        dtype = np.float64

    norms = np.empty(X.shape[0], dtype=dtype)

    for i in range(X.shape[0]):
        norm = 0
        for j in range(X.shape[1]):
            norm += X[i, j] * X[i, j]
        norms[i] = norm

    if not squared:
        np.sqrt(norms, norms)

    return norms


class SuperpixelClustering:
    def __init__(self, num=100, threshold=0.01, batch_size=100, cluster_centers=None, distance_threshold=0.5):
        self.centers = None
        self.labels = None
        self.num = num
        self.threshold = threshold
        self.batch_size = batch_size
        self.cluster_centers = cluster_centers
        self.distance_threshold = distance_threshold

    def fit(self, data):
        """
        超像素聚类
        """
        # 初始化
        # self.centers = datas[np.random.choice(datas.shape[0], self.num, replace=False)]
        self.centers = self.cluster_centers
        self.labels = -1 * np.ones((data.shape[0],), dtype=int)

        # 进行MiniBatch-KMeans聚类
        mbk = MiniBatchKMeans(init=self.centers, n_clusters=self.num, batch_size=self.batch_size, random_state=0)
        # mbk = CustomMBKMeans(init=self.centers, n_clusters=self.num, batch_size=self.batch_size,
        #                     distance_threshold=self.distance_threshold, global_interval=20)
        mbk.fit(data)
        self.centers = mbk.cluster_centers_
        self.labels = mbk.labels_
        centers = self.centers

        # 将同一标签的点放在一个列表中
        clusters = [[] for _ in range(self.num)]
        for j in range(data.shape[0]):
            label = self.labels[j]
            clusters[label].append(data[j])
        return clusters, centers


def chamfer_distance(x, y):
    """
    计算两组点之间的 Chamfer 距离
    """
    x = x.unsqueeze(0)
    xx = torch.sum(x * x, dim=-1)
    yy = torch.sum(y * y, dim=-1)
    x = x.double()
    xy = torch.matmul(x, y.transpose(2, 1))
    d = xx.unsqueeze(2) - 2 * xy + yy.unsqueeze(1)
    d1 = torch.min(d, dim=1)[0]
    d2 = torch.min(d, dim=2)[0]
    loss = torch.mean(d1) + torch.mean(d2)
    m = y.shape[1]  # 获取张量的形状
    loss /= m
    return loss

def l2(x, y):
    """
    计算两组点之间的 l2
    """
    x = x.unsqueeze(0)
    y = torch.from_numpy(y)
    loss = torch.norm(x - y, 2)
    return loss


def euclidean_distance(x, y):
    # Compute the Euclidean distance between two numpy arrays x and y
    # Assume that x and y have the same shape
    return np.sqrt(np.sum((x - y) ** 2))


class SuperpixelClustering2:
    def __init__(self, num=100, threshold=1e-4, batch_size=500, cluster_centers=None):
        self.D_value = None
        self.num = num
        self.threshold = threshold
        self.batch_size = batch_size
        self.labels = None
        self.centers = cluster_centers

    # 在current_sum大小的循环内，对 indices为true的所有点，计算点到self.centers[i]的距离开根号，然后把这个
    # 记录为每个点对应的D，然后如果有小于这个D的，就把label设置为这个self.centers[i]的i。
    def fit(self, data):
        data_norm = data

        # 初始化聚类中心标签
        self.labels = -1 * np.ones((data_norm.shape[0],), dtype=int)
        self.D_value = 100 * np.ones((data_norm.shape[0],), dtype=float)

        # 更新聚类中心和标签
        r = 0.05 * math.sqrt(data_norm.shape[0] / self.num)
        while True:
            old_centers = np.copy(self.centers)
            for i in range(self.num):
                # 找到与聚类中心距离小于等于半径 r = sqrt(N/k)的所有点
                indices = np.linalg.norm(data_norm[:, :3] - self.centers[i][:3], axis=1) <= r
                current_sum = np.sum(indices)
                if current_sum > 1:
                    for n in range(data_norm.shape[0]):
                        if indices[n]:
                            D = euclidean_distance(data_norm[n, :], self.centers[i])
                            if D < self.D_value[n]:
                                self.D_value[n] = D
                                self.labels[n] = i
            for i in range(self.num):
                self.centers[i] = np.mean(data_norm[self.labels == i, :], axis=0)
            nan_mask = np.isnan(self.centers).any(axis=1)
            self.centers[nan_mask] = old_centers[nan_mask]
            difference = np.linalg.norm(self.centers - old_centers)
            if difference < self.threshold:
                break
        # 将同一标签的点放在一个列表中
        clusters = []
        centers = []
        for i in range(self.num):
            cluster_i = [data[j] for j in range(data_norm.shape[0]) if self.labels[j] == i]
            if len(cluster_i) > 0:
                clusters.append(cluster_i)
                centers.append(self.centers[i])
        return clusters, centers