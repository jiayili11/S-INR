from __future__ import print_function
from modules import *
import scipy.io
import torch.optim
from sklearn.metrics import r2_score
from models import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

torch.backends.cudnn.enabled = True  # type: ignore
torch.backends.cudnn.benchmark = True  # type: ignore
dtype = torch.cuda.FloatTensor
setup_seed(0)


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
                         first_omega_0=30, hidden_omega_0=30.).cuda()  # 替换为您的模型类
    folder_path = './pth/'
    filename = filename1 + '.pth'
    file_path = os.path.join(folder_path, filename)
    model.load_state_dict(torch.load(file_path))

    model.eval()
    # colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple']
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # for i, cluster_data in enumerate(clusters):
    #     color = colors[i % len(colors)]
    #     x = cluster_data[:, 0].cpu().numpy()
    #     y = cluster_data[:, 1].cpu().numpy()
    #     z = cluster_data[:, 2].cpu().numpy()
    #     ax.scatter(x, y, z, c=color, label=f'Class {i + 1}')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.legend()
    # plt.show()
    # fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_subplot(111)
    # for i, cluster_data in enumerate(clusters):
    #     color = colors[i % len(colors)]
    #     x = cluster_data[:, 0].cpu().numpy()
    #     y = cluster_data[:, 1].cpu().numpy()
    #     ax.scatter(x, y, c=color, label=f'Class {i + 1}')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.legend()
    # plt.show()

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