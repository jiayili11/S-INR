import numpy as np
import torch
from plyfile import PlyData
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from torch.nn.functional import pairwise_distance
import math
from models import *
from sklearn.neighbors import KNeighborsRegressor

setup_seed(0)


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords


def lin2img(tensor, image_resolution=None):
    batch_size, num_samples, channels = tensor.shape
    if image_resolution is None:
        width = np.sqrt(num_samples).astype(int)
        height = width
    else:
        height = image_resolution[0]
        width = image_resolution[1]

    return tensor.permute(0, 2, 1).view(batch_size, channels, height, width)


class PointCloud(Dataset):
    def __init__(self, pointcloud_path, on_surface_points, keep_aspect_ratio=True):
        super().__init__()

        print("Loading point cloud")

        # plydata = PlyData.read(pointcloud_path)
        # x = plydata['vertex']['x']
        # y = plydata['vertex']['y']
        # z = plydata['vertex']['z']
        # xyz = np.vstack((x, y, z)).T
        # r = plydata['vertex']['r']
        # g = plydata['vertex']['g']
        # b = plydata['vertex']['b']
        # rgb = np.vstack((r, g, b)).T
        point_cloud = np.genfromtxt(pointcloud_path)
        print("Finished loading point cloud")

        coords = point_cloud[:, :3]
        self.normals = point_cloud[:, 3:]
        # coords = xyz
        # self.normals = rgb

        # Reshape point cloud such that it lies in bounding box of (-1, 1) (distorts geometry, but makes for high
        # sample efficiency)
        coords -= np.mean(coords, axis=0, keepdims=True)
        if keep_aspect_ratio:
            coord_max = np.amax(coords)
            coord_min = np.amin(coords)
        else:
            coord_max = np.amax(coords, axis=0, keepdims=True)
            coord_min = np.amin(coords, axis=0, keepdims=True)

        self.coords = (coords - coord_min) / (coord_max - coord_min)
        self.coords -= 0.5
        self.coords *= 2.

        normals_length = np.linalg.norm(self.normals, axis=1, keepdims=True)
        self.normals /= normals_length

        num = int(self.coords.shape[0] / 30)
        kmeans = KMeans(n_clusters=num, init='k-means++')
        kmeans.fit(self.coords)

        cluster_centers = kmeans.cluster_centers_
        cluster_normals = []
        for i in range(num):
            cluster_indices = np.where(kmeans.labels_ == i)[0]
            cluster_normals.append(np.mean(self.normals[cluster_indices], axis=0))

        self.cluster_centers = np.concatenate((cluster_centers, np.array(cluster_normals)), axis=1)
        self.data = np.concatenate((self.coords, self.normals), axis=1)

        self.on_surface_points = on_surface_points

    def __len__(self):
        return self.coords.shape[0] // self.on_surface_points

    def __getitem__(self, idx):
        point_cloud_size = self.coords.shape[0]

        off_surface_samples = self.on_surface_points  # **2
        total_samples = self.on_surface_points + off_surface_samples

        # Random coords
        rand_idcs = np.random.choice(point_cloud_size, size=self.on_surface_points)

        on_surface_coords = self.coords[rand_idcs, :]
        on_surface_normals = self.normals[rand_idcs, :]

        off_surface_coords = np.random.uniform(-1, 1, size=(off_surface_samples, 3))
        off_surface_normals = np.ones((off_surface_samples, 3)) * -1

        sdf = np.zeros((total_samples, 1))  # on-surface = 0
        sdf[self.on_surface_points:, :] = -1  # off-surface = -1

        coords = np.concatenate((on_surface_coords, off_surface_coords), axis=0)
        normals = np.concatenate((on_surface_normals, off_surface_normals), axis=0)
        cluster_centers = self.cluster_centers
        data = self.data
        return {'coords': torch.from_numpy(coords).float()}, {'sdf': torch.from_numpy(sdf).float(),
                                                              'normals': torch.from_numpy(normals).float()}, {
                   'cluster_centers': torch.from_numpy(cluster_centers).float()}, {
                   'datas': torch.from_numpy(data).float()}


class SuperpixelClustering_3d:
    def __init__(self, num=100, threshold=1e-4, batch_size=500, cluster_centers=None):
        self.D_value = None
        self.num = num
        self.threshold = threshold
        self.batch_size = batch_size
        self.labels = None
        self.centers = torch.from_numpy(cluster_centers).float().cuda()

    def fit(self, data):
        data_norm = torch.from_numpy(data).float().cuda()
        self.labels = -1 * torch.ones(data_norm.shape[0], dtype=torch.int32, device=data_norm.device)
        self.D_value = 1000 * torch.ones(data_norm.shape[0], dtype=torch.float32, device=data_norm.device)
        r = 0.001 * math.sqrt(data_norm.shape[0] / self.num)
        while True:
            old_centers = self.centers.clone()
            for i in range(self.num):
                indices = torch.norm(data_norm[:, :3] - self.centers[i][:3], dim=1) <= r
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
        # 将同一标签的点放在一个列表中
        clusters = [[] for _ in range(self.num)]
        centers = [[] for _ in range(self.num)]
        for i in range(self.num):
            idx = torch.nonzero(self.labels == i)
            clusters[i] = data_norm[idx].squeeze()
            centers[i] = self.centers[i]
        return clusters, centers


class point_pre_3d(Dataset):
    def __init__(self, pointcloud_path, missingrate, data_series, num):
        super().__init__()

        mat = scipy.io.loadmat(pointcloud_path + "_train_" + str(missingrate) + '.mat')
        self.train = mat[data_series + "_train_" + str(missingrate)]

        mat2 = scipy.io.loadmat(pointcloud_path + "_test_" + str(missingrate) + '.mat')
        self.test = mat2[data_series + "_test_" + str(missingrate)]

        mean = self.train[:, :3].mean(axis=0)
        std = self.train[:, :3].std(axis=0)
        min_val = np.min(self.train[:, :3])
        max_val = np.max(self.train[:, :3])
        # 归一化
        train_coor_norm = (self.train[:, :3].copy() - mean) / std
        train_coords = (train_coor_norm - min_val) / (max_val - min_val)
        data_dict = {
            'mean_coords': mean,
            'std_coords': std,
            'min_val': min_val,
            'max_val': max_val
        }
        mat_file_path = 'data_'+str(data_series)+'_sinr_train_' + str(missingrate) + '.mat'
        folder = './data_file/'
        mat_file_path = os.path.join(folder, mat_file_path)
        scipy.io.savemat(mat_file_path, data_dict)

        test_coords = self.test[:, :3]
        test_coords = (test_coords - mean) / std
        test_coords = (test_coords - min_val) / (max_val - min_val)
        knr = KNeighborsRegressor(n_neighbors=3)
        train = np.concatenate([train_coords, self.train[:, 3:]], axis=1)
        knr.fit(train[:, :3], train[:, 3:])
        y_pred = knr.predict(test_coords)
        add = np.concatenate([test_coords, y_pred], axis=1)
        new = np.concatenate([train, add], axis=0)

        self.coords = train_coords
        self.rgb = self.train[:, 3:]
        self.num = int(num)

        kmeans = KMeans(n_clusters=self.num, init='k-means++', n_init=10)
        self.input = new.copy()
        self.input[:, 0] = 20 * self.input[:, 0]
        self.input[:, 1] = 20 * self.input[:, 1]
        self.input[:, 2] = 20 * self.input[:, 2]
        kmeans.fit(self.input)
        self.cluster_centers = torch.from_numpy(kmeans.cluster_centers_).float().cuda()
        self.cluster_centers[:, 0] = self.cluster_centers[:, 0] / 20
        self.cluster_centers[:, 1] = self.cluster_centers[:, 1] / 20
        self.cluster_centers[:, 2] = self.cluster_centers[:, 2] / 20
        labels = kmeans.labels_[:self.train.shape[0]]

        self.clusters = []
        for i in range(self.num):
            cluster_indices = np.where(labels == i)[0]
            self.clusters.append(torch.from_numpy(np.concatenate((self.coords[cluster_indices],
                                                                  self.train[cluster_indices, 3:]),
                                                                 axis=1)).float().cuda())
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {'cluster_centers': self.cluster_centers}, {'num': self.num}, {
            'clusters': torch.from_numpy(self.clusters).float().cuda()}


class PointCloud_ply(Dataset):
    def __init__(self, pointcloud_path, missingrate):
        super().__init__()
        print("Loading point cloud")
        # 读取PLY文件
        plydata = PlyData.read(pointcloud_path)
        x = plydata['vertex']['x']
        y = plydata['vertex']['y']
        z = plydata['vertex']['z']
        xyz = np.vstack((x, y, z)).T
        r = plydata['vertex']['red']
        g = plydata['vertex']['green']
        b = plydata['vertex']['blue']
        rgb = np.vstack((r, g, b)).T
        print("Finished loading point cloud")
        self.coords = xyz
        self.rgb = rgb / 255.
        self.num = int(self.coords.shape[0] * 1e-3)
        gt = np.concatenate((self.coords, self.rgb), axis=1)
        missing_num = int(gt.shape[0] * missingrate)
        test_indices = np.random.choice(gt.shape[0], missing_num, replace=False)
        self.gt = gt
        self.observe = gt.copy()
        for i in [3, 4, 5]:
            self.observe[test_indices, i] = 0
        self.mask = np.ones([self.observe.shape[0], 3])
        self.mask[test_indices, :] = 0
        kmeans = KMeans(n_clusters=self.num, init='k-means++')
        self.input = self.observe.copy()
        self.input[:, 0] = 30 * self.input[:, 0]
        self.input[:, 1] = 30 * self.input[:, 1]
        self.input[:, 2] = 30 * self.input[:, 2]
        kmeans.fit(self.input)
        self.cluster_centers = kmeans.cluster_centers_
        self.cluster_ob = []
        self.cluster_gt = []
        self.cluster_mask = []
        for i in range(self.num):
            cluster_indices = np.where(kmeans.labels_ == i)[0]
            self.cluster_ob.append(torch.from_numpy(np.concatenate((self.coords[cluster_indices],
                                                                    self.observe[cluster_indices, 3:]),
                                                                   axis=1)).float().cuda())
            self.cluster_gt.append(torch.from_numpy(np.concatenate((self.coords[cluster_indices],
                                                                    self.gt[cluster_indices, 3:]),
                                                                   axis=1)).float().cuda())
            self.cluster_mask.append(torch.from_numpy(self.mask[cluster_indices, :]).float().cuda())
        # mat_file_path = f'out_test_swi_{filename}.mat'
        # scipy.io.savemat(mat_file_path, data_dict)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {'coords': torch.from_numpy(self.coords).float().cuda()}, {
            'cluster_centers': torch.from_numpy(self.cluster_centers).float().cuda()}, {
                   'gt': torch.from_numpy(self.gt).float().cuda()}, {'num': self.num}, {'observe': self.observe}, {
                   'cluster_gt': self.cluster_gt}, {'cluster_ob': self.cluster_ob}, {
                   'cluster_mask': self.cluster_mask}, {
                   'mask': self.mask}


def point_3d_input_process(clu_gt, pixel_num):
    count = 0
    coordinate_in = []
    color_in = []
    pixel_count = torch.zeros((pixel_num, 1))
    for sublist1 in clu_gt:
        pixel_count[count] = len(sublist1)
        coordinate_in.append(sublist1[:, :3].float().cuda())
        color_in.append(sublist1[:, 3:].float().cuda())
        count += 1
    pixel_count = pixel_count.cuda()
    return coordinate_in, color_in, pixel_count


def point_3d_input_process2(clu_gt, clu_ob, pixel_num):
    count = 0
    coordinate_gt = []
    color_gt = []
    coordinate_ob = []
    color_ob = []
    pixel_count = torch.zeros((pixel_num, 1))
    for sublist1, sublist2 in zip(clu_gt, clu_ob):
        pixel_count[count] = len(sublist1)
        coordinate_gt.append(sublist1[:, :3].float().cuda())
        color_gt.append(sublist1[:, 3:].float().cuda())
        coordinate_ob.append(sublist2[:, :3].float().cuda())
        color_ob.append(sublist2[:, 3:].float().cuda())
        count += 1
    pixel_count = pixel_count.cuda()
    return coordinate_gt, color_gt, coordinate_ob, color_ob, pixel_count


class point_pre_3d_mask(Dataset):
    def __init__(self, pointcloud_path, missingrate, multiplier):
        super().__init__()
        print("Loading point cloud")
        mat = scipy.io.loadmat(pointcloud_path + '.mat')
        mat2 = scipy.io.loadmat(pointcloud_path + '_ob_' + str(missingrate) + '.mat')
        mat3 = scipy.io.loadmat(pointcloud_path + '_mask_' + str(missingrate) + '.mat')

        self.gt = mat["Squirrel"]
        self.observe = mat2["Squirrel_ob_" + str(missingrate)]
        self.mask = mat3["Squirrel_mask_" + str(missingrate)]

        self.coords = self.observe[:, :3]
        self.rgb = self.observe[:, 3:]
        self.num = int(self.coords.shape[0] * multiplier)

        kmeans = KMeans(n_clusters=self.num, init='k-means++')
        self.input = self.observe.copy()
        self.input[:, 0] = 20 * self.input[:, 0]
        self.input[:, 1] = 20 * self.input[:, 1]
        self.input[:, 2] = 20 * self.input[:, 2]
        kmeans.fit(self.input)
        self.cluster_centers = torch.from_numpy(kmeans.cluster_centers_).float().cuda()
        self.cluster_centers[:, 0] = self.cluster_centers[:, 0] / 20
        self.cluster_centers[:, 1] = self.cluster_centers[:, 1] / 20
        self.cluster_centers[:, 2] = self.cluster_centers[:, 2] / 20
        self.cluster_ob = []
        self.cluster_gt = []
        self.cluster_mask = []
        for i in range(self.num):
            cluster_indices = np.where(kmeans.labels_ == i)[0]
            self.cluster_ob.append(torch.from_numpy(np.concatenate((self.coords[cluster_indices],
                                                                    self.observe[cluster_indices, 3:]),
                                                                   axis=1)).float().cuda())
            self.cluster_gt.append(torch.from_numpy(np.concatenate((self.coords[cluster_indices],
                                                                    self.gt[cluster_indices, 3:]),
                                                                   axis=1)).float().cuda())
            self.cluster_mask.append(torch.from_numpy(self.mask[cluster_indices, :]).float().cuda())

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {'coords': torch.from_numpy(self.coords).float().cuda()}, {
            'cluster_centers': torch.from_numpy(self.cluster_centers).float().cuda()}, {
                   'gt': torch.from_numpy(self.gt).float().cuda()}, {'num': self.num}, {'observe': self.observe}, {
                   'cluster_gt': self.cluster_gt}, {'cluster_ob': self.cluster_ob}, {
                   'cluster_mask': self.cluster_mask}, {
                   'mask': self.mask}
