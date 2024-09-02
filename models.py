import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import pairwise_distance
import os
from utils import *
import math
from scipy.linalg import eigh
import torch
from sklearn.metrics import r2_score
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans
# from torch_geometric.datas import Data
# from torch_geometric.utils import from_scipy_sparse_matrix
# from torch_geometric.nn import knn_graph
setup_seed(0)
dtype = torch.cuda.FloatTensor


class Siren(nn.Module):  # 输入坐标的原始状态
    def __init__(self, in_features, hidden_features, out_features, outermost_linear=False, first_omega_0=30,
                 hidden_omega_0=30):
        super().__init__()

        self.net = []
        self.hidden_layers = len(hidden_features) - 1
        self.net.append(SineLayer(in_features, hidden_features[0], is_first=True, omega_0=first_omega_0))
        for i in range(self.hidden_layers):
            self.net.append(
                SineLayer(hidden_features[i], hidden_features[i + 1], is_first=False, omega_0=hidden_omega_0))
        if outermost_linear:
            final_linear = nn.Linear(hidden_features[-1], out_features)
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features[-1]) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features[-1]) / hidden_omega_0)
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features[-1], out_features, is_first=False, omega_0=hidden_omega_0))
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        # coords = coords.clone().detach().requires_grad_(True)
        output = self.net(coords)
        return output


class Sin(nn.Module):
    def __init__(self):
        super(Sin, self).__init__()

    def forward(self, x):
        return torch.sin(x)


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30.):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class Inr_block(nn.Module):
    def __init__(self, c_in=2, c_mid=32, c_out=3, hidden_layers=3, first_omega_0=25,
                 hidden_omega_0=30., outermost_linear=True, bias=True):
        super().__init__()
        self.inr = []
        self.inr.append(SineLayer(c_in, c_mid, is_first=True, omega_0=first_omega_0))
        for i in range(hidden_layers):
            self.inr.append(SineLayer(c_mid, c_mid))
        # self.inr.append(nn.Dropout(p=0.01))
        if outermost_linear:
            final_linear = nn.Linear(c_mid, c_out, bias=bias)
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / c_mid) / hidden_omega_0, np.sqrt(6 / c_mid) / hidden_omega_0)
            self.inr.append(final_linear)
        else:
            self.inr.append(SineLayer(c_mid, c_out, is_first=False, omega_0=hidden_omega_0))
        self.inr = nn.Sequential(*self.inr)

    def forward(self, input):
        return self.inr(input)


class Inr_block_se(nn.Module):
    def __init__(self, c_in=2, c_mid=32, c_out=3, hidden_layers=3, first_omega_0=25,
                 hidden_omega_0=30., outermost_linear=True, bias=True):
        super().__init__()
        self.inr = []
        self.inr.append(SineLayer(c_in, c_mid, is_first=True, omega_0=first_omega_0))
        self.inr.append(SE_Block(c_mid, int(c_mid / 4)))
        for i in range(hidden_layers):
            self.inr.append(SineLayer(c_mid, c_mid))
            self.inr.append(SE_Block(c_mid, int(c_mid / 4)))
        # self.inr.append(nn.Dropout(p=0.01))
        if outermost_linear:
            final_linear = nn.Linear(c_mid, c_out, bias=bias)
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / c_mid) / hidden_omega_0, np.sqrt(6 / c_mid) / hidden_omega_0)
            self.inr.append(final_linear)
        else:
            self.inr.append(SineLayer(c_mid, c_out, is_first=False, omega_0=hidden_omega_0))
        if c_out==3:
            self.inr.append(SE_Block(c_out, 1))
        else:
            self.inr.append(SE_Block(c_out, int(c_out / 4)))
        self.inr = nn.Sequential(*self.inr)

    def forward(self, input):
        return self.inr(input)
    

class ComplexReLU(torch.nn.Module):
    def forward(self, input_tensor):
        real_part = F.relu(input_tensor.real)
        imag_part = F.relu(input_tensor.imag)
        return torch.complex(real_part, imag_part)


class ComplexSigmoid(nn.Module):
    def forward(self, x):
        sigmoid_real = torch.sigmoid(x.real)
        sigmoid_imag = torch.sigmoid(x.imag)
        return torch.complex(sigmoid_real, sigmoid_imag)


class SE_Block_complex(nn.Module):
    def __init__(self, ch_in, reduction=16, dtype=torch.float):
        super(SE_Block_complex, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False, dtype=dtype),
            ComplexReLU(),
            nn.Linear(ch_in // reduction, ch_in, bias=False, dtype=dtype),
            ComplexSigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x.T)
        y = self.fc(y.T)
        # a = y.cpu().detach().numpy()
        return x * y.expand_as(x)


class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16, dtype=torch.float):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False, dtype=dtype),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False, dtype=dtype),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x.T)
        y = self.fc(y.T)
        return x * y.expand_as(x)


class NET_S_INR_wio_se(nn.Module):
    def __init__(self, pixel_count, pixel_num, c_in=2, r=None, c_out=3, c_mid=32, hidden_layers=5,
                 first_omega_0=25, hidden_omega_0=30.):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.W = []
        self.W = nn.ModuleList()
        self.pixel_count = pixel_count
        self.pixel_num = pixel_num
        self.act = Sin()
        self.D = nn.Linear(r, c_out, bias=False)
        for i in range(pixel_num):
            self.W.append(Inr_block(c_in, c_mid, r, hidden_layers, first_omega_0, hidden_omega_0))

    def forward(self, coords):
        output_rgb = list()
        for i in range(self.pixel_num):
            coords[i] = self.S[i](coords[i])
            out = self.W[i](coords[i])
            out = self.D(out)
            output_rgb.append(out)
        return output_rgb


class NET_S_INR(nn.Module):
    def __init__(self, pixel_num, c_in=2, r=None, c_out=3, c_mid=32, hidden_layers=5,
                 first_omega_0=25, hidden_omega_0=30.):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.W = []
        self.W = nn.ModuleList()
        self.pixel_num = pixel_num
        self.act = Sin()
        self.D = nn.Linear(r, c_out, bias=False)
        for i in range(pixel_num):
            self.W.append(Inr_block_se(c_in, c_mid, r, hidden_layers, first_omega_0, hidden_omega_0))

    def forward(self, coords):
        output_rgb = list()
        for i in range(self.pixel_num):
            output_rgb.append(self.D(self.W[i](coords[i])))
        return output_rgb


class NET_S_INR_unshare(nn.Module):
    def __init__(self, pixel_count, pixel_num, c_in=2, r=None, c_out=3, c_mid=32, hidden_layers=5,
                 first_omega_0=25,
                 hidden_omega_0=30., t=None):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.W = []
        self.W = nn.ModuleList()
        self.pixel_count = pixel_count
        self.pixel_num = pixel_num
        self.act = Sin()
        self.D = nn.ModuleList()
        for i in range(pixel_num):
            self.W.append(Inr_block(c_in, c_mid, r, hidden_layers, first_omega_0, hidden_omega_0))
            self.D.append(nn.Linear(r, c_out, bias=False))

    def forward(self, coords):
        output_rgb = list()
        for i in range(self.pixel_num):
            out = self.W[i](coords[i])
            out = self.D[i](out)
            output_rgb.append(out)
        return output_rgb

class point2d_pre(Dataset):
    def __init__(self, point_data_path, num, data_series, missingrate):
        super().__init__()

        mat = scipy.io.loadmat(point_data_path + 'train_' + str(missingrate) + '.mat')
        train = mat["train"]

        self.coords = train[:, :2]
        self.gt = train[:, 2:]

        max_value1 = np.max(self.gt[:, 0])
        max_value2 = np.max(self.gt[:, 1])
        max_value3 = np.max(self.gt[:, 2])
        max_value4 = np.max(self.gt[:, 3])
        max_value5 = np.max(self.gt[:, 4])
        self.gt[:, 0] = self.gt[:, 0] / max_value1
        self.gt[:, 1] = self.gt[:, 1] / max_value2
        self.gt[:, 2] = self.gt[:, 2] / max_value3
        self.gt[:, 3] = self.gt[:, 3] / max_value4
        self.gt[:, 4] = self.gt[:, 4] / max_value5

        mean_coords = np.mean(self.coords, axis=0)
        std_coords = np.std(self.coords, axis=0)
        self.coords = (self.coords - mean_coords) / std_coords
        min_val = np.min(self.coords)
        max_val = np.max(self.coords)
        self.coords = (self.coords - min_val) / (max_val - min_val)
        self.data = np.concatenate((self.coords, self.gt), axis=1)

        mat2 = scipy.io.loadmat(point_data_path + 'test_' + str(missingrate) + '.mat')
        test = mat2["test"]
        test_coords = test[:, :2]
        test_coords = (test_coords - mean_coords) / std_coords
        test_coords = (test_coords - min_val) / (max_val - min_val)

        knr = KNeighborsRegressor(n_neighbors=3)  # 设置邻居数量
        knr.fit(self.data[:, :2], self.data[:, 2:])
        y_pred = knr.predict(test_coords)
        add = np.concatenate([test_coords, y_pred], axis=1)
        new = np.concatenate([self.data, add], axis=0)

        kmeans = KMeans(n_clusters=num, init='k-means++', n_init=10)

        kmeans.fit(new)

        cluster_centers = kmeans.cluster_centers_
        self.cluster_centers = torch.from_numpy(cluster_centers).float().cuda()

        data_dict = {
            'mean_coords': mean_coords,
            'std_coords': std_coords,
            'min_val': min_val,
            'max_val': max_val,
            'max_value1': max_value1,
            'max_value2': max_value2,
            'max_value3': max_value3,
            'max_value4': max_value4,
            'max_value5': max_value5
        }

        mat_file_path = 'data_' + str(data_series) + '_swi_train_' + str(missingrate) + '.mat'
        folder = './data_file/'
        mat_file_path = os.path.join(folder, mat_file_path)
        scipy.io.savemat(mat_file_path, data_dict)

        labels = kmeans.labels_[:self.data.shape[0]]  # [:self.datas.shape[0]]
        self.data = torch.from_numpy(self.data).float().cuda()
        self.cluster_gt = []
        for i in range(num):
            cluster_indices = np.where(labels == i)[0]
            self.cluster_gt.append(torch.from_numpy(
                np.concatenate((self.coords[cluster_indices], self.gt[cluster_indices]), axis=1)).float().cuda())

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {'coords': self.coords}, {'cluster_centers': self.cluster_centers}, {
            'datas': self.data}, {'max_value': self.max_value}, {'clusters': self.cluster_gt}


def preprocess(inputdata, gt1, filename):
    mean_coords = np.mean(inputdata, axis=0)
    std_coords = np.std(inputdata, axis=0)
    max_value1 = np.max(gt1[:, 0])
    max_value2 = np.max(gt1[:, 1])
    max_value3 = np.max(gt1[:, 2])
    max_value4 = np.max(gt1[:, 3])
    max_value5 = np.max(gt1[:, 4])
    gt1[:, 0] = gt1[:, 0] / max_value1
    gt1[:, 1] = gt1[:, 1] / max_value2
    gt1[:, 2] = gt1[:, 2] / max_value3
    gt1[:, 3] = gt1[:, 3] / max_value4
    gt1[:, 4] = gt1[:, 4] / max_value5
    gt = torch.from_numpy(gt1).float().cuda()

    coords = (inputdata - mean_coords) / std_coords
    min_val = np.min(coords)
    max_val = np.max(coords)
    coords = (coords - min_val) / (max_val - min_val)
    input_data = torch.from_numpy(coords).float().cuda()

    data_dict = {
        'mean_coords': mean_coords,
        'std_coords': std_coords,
        'min_val': min_val,
        'max_val': max_val,
        'max_value1': max_value1,
        'max_value2': max_value2,
        'max_value3': max_value3,
        'max_value4': max_value4,
        'max_value5': max_value5
    }
    folder = './data_file/'
    mat_file_path = filename + '.mat'
    mat_file_path = os.path.join(folder, mat_file_path)
    scipy.io.savemat(mat_file_path, data_dict)

    return input_data, gt


class point2d_pre_2(Dataset):
    def __init__(self, point_data_path, point_data_gt_path, multiplier):
        super().__init__()

        print("Loading point cloud")

        mat = scipy.io.loadmat(point_data_path)
        self.coords = mat["data_train_200"]
        mat2 = scipy.io.loadmat(point_data_gt_path)
        self.gt = mat2["data_train_gt_200"]
        self.gt = self.gt.T

        print("Finished loading point cloud")
        data = np.concatenate((self.coords, self.gt), axis=1)
        num = int(self.coords.shape[0] * multiplier)

        kmeans = KMeans(n_clusters=num, init='k-means++')
        kmeans.fit(data)

        cluster_centers = kmeans.cluster_centers_
        self.cluster_centers = torch.from_numpy(cluster_centers).float().cuda()

        data_cluster = []
        for i in range(num):
            cluster_indices = np.where(kmeans.labels_ == i)[0]
            data_cluster.append(torch.from_numpy(data[cluster_indices]).float().cuda())
        self.data_cluster = data_cluster
        self.data = torch.from_numpy((np.concatenate((self.coords, self.gt), axis=1))).float().cuda()

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {'coords': self.coords}, {'cluster_centers': self.cluster_centers}, {
            'datas': self.data}, {'clusters': self.data_cluster}


class euclidean_distance(nn.Module):
    def __init__(self):
        super(euclidean_distance, self).__init__()

    def forward(self, x, y):
        results = torch.sqrt(torch.sum((x - y) ** 2))
        return results


def euclidean_distance2(x, y):
    # Compute the Euclidean distance between two numpy arrays x and y
    # Assume that x and y have the same shape
    return np.sqrt(np.sum((x - y) ** 2))


class SuperpixelClustering_2d_point(nn.Module):
    def __init__(self, num=100, threshold=1e-4, cluster_centers=None):
        super().__init__()
        self.D_value = None
        self.num = num
        self.threshold = threshold
        self.labels = None
        self.centers = cluster_centers

    def fit(self, data):
        data_norm = data
        self.labels = -1 * torch.ones(data_norm.shape[0], dtype=torch.int32, device=data.device)
        self.D_value = 1000 * torch.ones(data_norm.shape[0], dtype=torch.float32, device=data.device)
        r = 0.005 * math.sqrt(data_norm.shape[0] / self.num)

        while True:
            old_centers = self.centers.clone()
            for i in range(self.num):
                indices = torch.norm(data_norm[:, :2] - self.centers[i][:2], dim=1) <= r
                current_sum = torch.sum(indices)
                if current_sum > 1:
                    D = pairwise_distance(data_norm[indices, :], self.centers[i])
                    update_mask = D < self.D_value[indices]
                    self.D_value[indices] = update_mask * D
                    self.labels[indices] = torch.where(update_mask, (update_mask * i).to(torch.int32),
                                                       self.labels[indices])

            for i in range(self.num):
                idx = torch.nonzero(self.labels == i)
                self.centers[i] = torch.mean(data_norm[idx], dim=0)
            nan_mask = torch.isnan(self.centers).any(dim=1)
            self.centers[nan_mask] = old_centers[nan_mask]
            difference = torch.norm(self.centers - old_centers)
            if difference < self.threshold:
                break
        clusters = [[] for _ in range(self.num)]
        centers = [[] for _ in range(self.num)]
        for i in range(self.num):
            idx = torch.nonzero(self.labels == i)
            clusters[i] = data[idx].squeeze()
            centers[i] = self.centers[i]
        return clusters, centers


class SuperpixelClustering_2d_point2:
    def __init__(self, num=100, threshold=1e-4, cluster_centers=None):
        self.D_value = None
        self.num = num
        self.threshold = threshold
        self.labels = None
        self.centers = cluster_centers

    def fit(self, data):
        data_norm = data

        self.labels = -1 * np.ones((data_norm.shape[0],), dtype=int)
        self.D_value = 100 * np.ones((data_norm.shape[0],), dtype=float)

        r = 0.02 * math.sqrt(data_norm.shape[0] / self.num)
        while True:
            old_centers = np.copy(self.centers)
            for i in range(self.num):

                indices = np.linalg.norm(data_norm[:, :2] - self.centers[i][:2], axis=1) <= r
                current_sum = np.sum(indices)
                if current_sum > 1:
                    for n in range(data_norm.shape[0]):
                        if indices[n]:
                            D = euclidean_distance2(data_norm[n, :], self.centers[i])
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

        clusters = []
        centers = []
        for i in range(self.num):
            cluster_i = [data[j] for j in range(data_norm.shape[0]) if self.labels[j] == i]
            if len(cluster_i) > 0:
                clusters.append(cluster_i)
                centers.append(self.centers[i])
        return clusters, centers


def detection_results(out_tensor, gt_tensor):
    mse = torch.mean((out_tensor - gt_tensor) ** 2)
    rmse = torch.sqrt(mse)
    max_value = torch.max(gt_tensor)
    min_value = torch.min(gt_tensor)
    target_range = max_value - min_value
    nrmse = rmse / target_range
    return nrmse

def point_2d_input_process(data_clu, pixel_num):
    count = 0
    input_cluster_cor = []
    input_cluster_pre = []
    pixel_count = torch.zeros((pixel_num, 1))
    for sublist in data_clu:
        pixel_count[count] = len(sublist)
        # my_matrices = np.vstack([np.array(arr) for arr in sublist])
        input_cluster_cor.append(sublist[:, :2].unsqueeze(0).float().cuda())
        input_cluster_pre.append(sublist[:, 2:].float().cuda())
        count += 1
    pixel_count = pixel_count.cuda()

    return input_cluster_cor, input_cluster_pre, pixel_count


def point_2d_input_process2(data_clu):
    count = 0
    input_cluster_cor = []
    input_cluster_pre = []
    pixel_count = []
    cluster_tensor = []
    min_len = math.inf
    max_len = 0
    for sublist in data_clu:
        pixel_count.append(len(sublist))
        if len(sublist) > max_len:
            max_len = len(sublist)
        if len(sublist) < min_len:
            min_len = len(sublist)
        my_matrices = np.vstack([np.array(arr) for arr in sublist])
        gt_sub = my_matrices.reshape(pixel_count[count], 3)
        gt_sub_tensor = torch.tensor(gt_sub[:, :3])
        cluster_tensor.append(gt_sub_tensor.unsqueeze(0).float())
        input_cluster_cor.append(torch.from_numpy(gt_sub[np.newaxis, :, :2]).float().cuda())
        input_cluster_pre.append(torch.from_numpy(gt_sub[np.newaxis, :, 2:]).squeeze().float().cuda())
        count += 1
    return input_cluster_cor, input_cluster_pre, pixel_count, min_len, max_len, cluster_tensor


def chamfer_distance(x, y):
    x = x.unsqueeze(0)
    y = torch.from_numpy(y)
    xx = torch.sum(x * x, dim=-1)
    yy = torch.sum(y * y, dim=-1)
    x = x.double()
    xy = torch.matmul(x, y.transpose(2, 1))
    d = xx.unsqueeze(2) - 2 * xy + yy.unsqueeze(1)
    d1 = torch.min(d, dim=1)[0]
    d2 = torch.min(d, dim=2)[0]
    loss = torch.mean(d1) + torch.mean(d2)
    m = y.shape[1]
    loss /= m
    return loss


def preprocess_swi_3d(num, point_cloud_path, centers_filename, data_series, missingrate):
    mat = scipy.io.loadmat(point_cloud_path + "_test_" + str(missingrate) + '.mat')
    test_np = mat[data_series+"_test_" + str(missingrate)]
    test_data = test_np[:, :3]
    test_torch = torch.from_numpy(test_np).float().cuda()

    folder_path = './centers/'
    file_path = os.path.join(folder_path, centers_filename)
    centers = torch.from_numpy(np.load(file_path)).float().cuda()
    data_norm = test_torch.clone()

    folder = './data_file/'
    data_file = scipy.io.loadmat(
        os.path.join(folder, 'data_' + str(data_series) + '_sinr_train_' + str(missingrate) + '.mat'))

    coords = (test_data.copy() - data_file["mean_coords"]) / data_file["std_coords"]
    min_val = data_file["min_val"]
    max_val = data_file["max_val"]
    coords = (coords - min_val) / (max_val - min_val)
    test_coords = torch.from_numpy(coords).float().cuda()
    data_norm[:, :3] = test_coords
    test_torch_ini = test_torch.clone()
    test_torch[:, :3] = test_coords

    labels = -1 * torch.ones(data_norm.shape[0], dtype=torch.int32, device=centers.device)
    # labels = labels.unsqueeze(1)
    D_value = 1000 * torch.ones(data_norm.shape[0], dtype=torch.float32, device=centers.device)
    # D_value = D_value.unsqueeze(1)

    data_norm[:, 0] = 20 * data_norm[:, 0]
    data_norm[:, 1] = 20 * data_norm[:, 1]
    data_norm[:, 2] = 20 * data_norm[:, 2]

    centers[:, 0] = 20 * centers[:, 0]
    centers[:, 1] = 20 * centers[:, 1]
    centers[:, 2] = 20 * centers[:, 2]

    for i in range(num):
        D = pairwise_distance(data_norm[:, :], centers[i])
        update_mask = D < D_value[:]
        D_value[update_mask] = D[update_mask]
        labels[:] = torch.where(update_mask, (update_mask * i).to(torch.int32), labels[:])

    clusters = [[] for _ in range(num)]
    clusters_ini = [[] for _ in range(num)]
    for i in range(num):
        idx = (labels == i).nonzero()
        idx = idx[:, 0]
        clusters[i] = test_torch[idx].squeeze()
        clusters_ini[i] = test_torch_ini[idx].squeeze()
    input_cluster_cor = []
    input_cluster_cor_ini = []
    input_cluster_pre = []
    for sublist, sublist_ini in zip(clusters, clusters_ini):
        input_cluster_cor.append(sublist[:, :3].unsqueeze(0).float().cuda())
        input_cluster_cor_ini.append(sublist_ini[:, :3].unsqueeze(0).float().cuda())
        input_cluster_pre.append((sublist[:, 3:]).float().cuda())

    pre_tensor = torch.cat(input_cluster_pre, dim=0).detach()
    pre_np = pre_tensor.detach().cpu().numpy()

    max_value = torch.max(test_torch[:, 3:])
    min_value = torch.min(test_torch[:, 3:])

    return input_cluster_cor, input_cluster_cor_ini, pre_tensor, max_value, min_value, pre_np


def perform_inference_3d_uv(num, c_mid, rank, filename1, cluster_cor_test, pre_tensor, cor_ini, max_value, min_value, pre_np, last_time):

    setup_seed(0)
    model = NET_S_INR(num, c_in=3, r=rank, c_out=3, c_mid=c_mid, hidden_layers=3,
                         first_omega_0=30, hidden_omega_0=30.).cuda()
    folder_path = './pth/'
    filename = filename1 + '.pth'
    file_path = os.path.join(folder_path, filename)
    model.load_state_dict(torch.load(file_path))

    model.eval()

    output = model(cluster_cor_test)

    output = torch.cat(output, dim=1).squeeze()

    out = output.detach().cpu().numpy()

    nrmse = 0
    for i in range(3):
        nrmse += detection_results(output[:, i], pre_tensor[:, i])
    nrmse = nrmse/3
    r2 = r2_score(pre_np, out)
    # print("filename", filename, "mae:", mae, "rmse:", rmse, "nrmse:", nrmse)

    data2 = np.concatenate([torch.cat(cor_ini, dim=1).squeeze().detach().cpu().numpy(), out], axis=1)
    # data_dict = {
    #     'clean_data': data2,
    #     'time': str(last_time),
    # }
    # mat_file_path = f'out_test_swi_{filename1}.mat'
    # scipy.io.savemat(mat_file_path, data_dict)
    return nrmse, r2, data2


