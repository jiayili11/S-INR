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

def preprocess_test(num, centers_filename, missingrate, data_series, point_data_path):
    def pairwise_distance(x, y):
        return torch.norm(x[:, None] - y, dim=2)

    mat = scipy.io.loadmat(point_data_path + 'test_' + str(missingrate) + '.mat')
    inputdata = mat["test"]

    test_data = inputdata[:, :2]
    test_data_gt = inputdata[:, 2:]

    folder_path = './data_file/'
    file_path = os.path.join(folder_path, 'data_'+str(data_series)+'_swi_train_' + str(missingrate) + '.mat')
    data_file = scipy.io.loadmat(file_path)
    coords = (test_data - data_file["mean_coords"]) / data_file["std_coords"]

    min_val = data_file["min_val"]
    max_val = data_file["max_val"]
    coords = (coords - min_val) / (max_val - min_val)

    data1 = np.concatenate((coords, test_data_gt), axis=1)
    data = torch.from_numpy(data1).float().cuda()
    data_xy = torch.from_numpy(test_data).float().cuda()

    folder_path = './centers/'
    file_path = os.path.join(folder_path, centers_filename)
    centers = torch.from_numpy(np.load(file_path)).float().cuda()
    data_norm = data1

    data_norm[:, 2] /= data_file["max_value1"].item()
    data_norm[:, 3] /= data_file["max_value2"].item()
    data_norm[:, 4] /= data_file["max_value3"].item()
    data_norm[:, 5] /= data_file["max_value4"].item()
    data_norm[:, 6] /= data_file["max_value5"].item()

    data_norm = torch.from_numpy(data_norm).float().cuda()

    labels = -1 * torch.ones(data_norm.shape[0], dtype=torch.int32, device=data.device)
    labels = labels.unsqueeze(1)
    D_value = 1000 * torch.ones(data_norm.shape[0], dtype=torch.float32, device=data.device)
    D_value = D_value.unsqueeze(1)

    for i in range(num):
        D = pairwise_distance(data_norm[:, :], centers[i])
        update_mask = D < D_value[:]
        D_value[update_mask] = D[update_mask]
        labels[:] = torch.where(update_mask, (update_mask * i).to(torch.int32), labels[:])

    clusters = [[] for _ in range(num)]
    clusters_xy = [[] for _ in range(num)]
    for i in range(num):
        idx = (labels == i).nonzero()
        idx = idx[:, 0]
        clusters[i] = data[idx].squeeze()
        clusters_xy[i] = data_xy[idx].squeeze()
    input_cluster_cor = []
    input_cluster_pre = []
    input_cluster_cor_ini = []
    for sublist, sublist_xy in zip(clusters, clusters_xy):
        input_cluster_cor.append(sublist[:, :2].unsqueeze(0).float().cuda())
        input_cluster_pre.append((sublist[:, 2:]).float().cuda())
        input_cluster_cor_ini.append(sublist_xy.float().cuda())

    pre_tensor = torch.cat(input_cluster_pre, dim=0).detach()
    pre_np = pre_tensor.detach().cpu().numpy()
    input_xy = (torch.cat(input_cluster_cor_ini, dim=0).detach()).cpu().numpy()

    return input_cluster_cor, pre_np, input_xy


def perform_inference_uv_2d(num, c_mid, r, filename1, missingrate, input_cluster_cor, prcp_np, test_xy, data_series):
    setup_seed(0)
    model = NET_S_INR(num, c_in=2, r=r, c_out=5, c_mid=c_mid, hidden_layers=3,
                         first_omega_0=30, hidden_omega_0=30.).cuda()  # 替换为您的模型类
    folder_path = './pth/'
    filename = filename1 + '.pth'
    file_path = os.path.join(folder_path, filename)
    model.load_state_dict(torch.load(file_path))

    model.eval()
    folder_path = './data_file/'
    file_path = os.path.join(folder_path, 'data_' + str(data_series) + '_swi_train_' + str(missingrate) + '.mat')
    data_file = scipy.io.loadmat(file_path)

    output = model(input_cluster_cor)

    output = torch.cat(output, dim=1).squeeze()

    output[:, 0] *= data_file["max_value1"].item()
    output[:, 1] *= data_file["max_value2"].item()
    output[:, 2] *= data_file["max_value3"].item()
    output[:, 3] *= data_file["max_value4"].item()
    output[:, 4] *= data_file["max_value5"].item()

    out = output.detach().cpu().numpy()

    nrmse = 0
    for i in range(5):
        mse = np.mean((out[:, i] - prcp_np[:, i]) ** 2)
        rmse = np.sqrt(mse)
        max_value = np.max(prcp_np[:, i])
        min_value = np.min(prcp_np[:, i])
        target_range = max_value - min_value
        nrmse += rmse / target_range
    nrmse = nrmse / 5
    r2 = r2_score(prcp_np, out)
    # print("filename", filename, "mae:", mae, "rmse:", rmse, "nrmse:", nrmse)
    data2 = np.concatenate([test_xy, out], axis=1)
    return nrmse, r2, data2
