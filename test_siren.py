from __future__ import print_function
from modules import *
import scipy.io
import torch.optim
from fourier import fourier_features
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

torch.backends.cudnn.enabled = True  # type: ignore
torch.backends.cudnn.benchmark = True  # type: ignore
dtype = torch.cuda.FloatTensor
setup_seed(0)


def evaluate_model(nfreqs, c_mid, filename, missingrate=0, data_series=0):
    # Load model
    setup_seed(0)
    if nfreqs == 0:
        model = Siren(in_features=2, out_features=5, hidden_features=[c_mid, c_mid, c_mid, c_mid],
                      outermost_linear=True).cuda()
    elif nfreqs == 1:
        model = Siren(in_features=32, out_features=5, hidden_features=[c_mid, c_mid, c_mid, c_mid],
                      outermost_linear=True).cuda()
    else:
        model = Siren(in_features=nfreqs * 4, out_features=1, hidden_features=[c_mid, c_mid, c_mid, c_mid],
                      outermost_linear=True).cuda()
    folder_path = './pth/'
    filename = filename + '.pth'
    file_path = os.path.join(folder_path, filename)
    model.load_state_dict(torch.load(file_path))
    model.eval()

    # Load test datas
    mat = scipy.io.loadmat('./datas/rain_data/data_' + str(data_series) + '_test_' + str(missingrate) + '.mat')
    test = mat["test"]
    test_data = torch.from_numpy(test[:, :2]).float().cuda()
    test_data_gt = test[:, 2:]

    folder = './data_file/'
    data_file = scipy.io.loadmat(
        os.path.join(folder, 'data_' + str(data_series) + '_siren_train_' + str(missingrate) + '.mat'))
    # Make predictions
    if nfreqs == 0:
        coords = (test[:, :2] - data_file["mean_coords"]) / data_file["std_coords"]
        # 如果需要将结果缩放到 [0, 1] 范围内，可以进行最小-最大缩放
        min_val = data_file["min_val"]
        max_val = data_file["max_val"]
        coords = (coords - min_val) / (max_val - min_val)
        test_coords = torch.from_numpy(coords).float().cuda()
        output = model(test_coords)
    elif nfreqs == 1:
        coords = (test[:, :2] - data_file["mean_coords"]) / data_file["std_coords"]
        # 如果需要将结果缩放到 [0, 1] 范围内，可以进行最小-最大缩放
        min_val = data_file["min_val"]
        max_val = data_file["max_val"]
        coords = (coords - min_val) / (max_val - min_val)
        test_coords = torch.from_numpy(coords).float().cuda()
        train_coor = test_coords
        freq = 8
        freqs = (2 ** (freq / (freq - 1)) ** torch.linspace(0., freq - 1, steps=freq)).cuda()
        train_coor_pe = freqs * torch.unsqueeze(train_coor, -1)
        # 高斯噪声
        # train_coor_pe = train_coor_pe + Normal(0, 0.1).sample(coordf2.shape).cuda() * torch.unsqueeze(train_coor, -1)
        train_coor_pe = torch.cat((torch.cos(train_coor_pe), torch.sin(train_coor_pe)), dim=-1).reshape(
            train_coor_pe.size(0), 2 * train_coor_pe.size(1) * train_coor_pe.size(2))
        output = model(train_coor_pe)
    else:
        test_fourier = fourier_features(test_data, nfreqs)
        output = model(test_fourier)

    out = output.detach().cpu().numpy()

    out[:, 0] *= data_file["max_value1"].item()
    out[:, 1] *= data_file["max_value2"].item()
    out[:, 2] *= data_file["max_value3"].item()
    out[:, 3] *= data_file["max_value4"].item()
    out[:, 4] *= data_file["max_value5"].item()

    nrmse = 0
    for i in range(5):
        mse = np.mean((out[:, i] - test_data_gt[:, i]) ** 2)
        rmse = np.sqrt(mse)
        max_value = np.max(test_data_gt[:, i])
        min_value = np.min(test_data_gt[:, i])
        target_range = max_value - min_value
        # 计算标准化均方根误差 (NRMSE)
        nrmse += rmse / target_range
    nrmse = nrmse / 5
    r2 = r2_score(test_data_gt, out)

    # data1 = np.concatenate([test_data1, test_data_gt], axis=1)
    data2 = np.concatenate([test[:, :2], out], axis=1)
    # fig = plt.figure()
    # ax1 = fig.add_subplot(121, projection='3d')
    # ax2 = fig.add_subplot(122, projection='3d')
    #
    # # 第一个子图
    # x1 = data1[:,0]
    # y1 = data1[:,1]
    # z1 = data1[:,2]
    # ax1.scatter(x1, y1, z1, c='red')
    # ax1.set_xlabel('x')
    # ax1.set_ylabel('y')
    # ax1.set_zlabel('z')
    #
    # # 第二个子图
    # x2 = data2[:,0]
    # y2 = data2[:,1]
    # z2 = data2[:,2]
    # ax2.scatter(x2, y2, z2, c='blue')
    # ax2.set_xlabel('x')
    # ax2.set_ylabel('y')
    # ax2.set_zlabel('z')
    #
    # # 获取第一个子图的轴限制
    # x1_lim = ax1.get_xlim()
    # y1_lim = ax1.get_ylim()
    # z1_lim = ax1.get_zlim()
    #
    # # 应用轴限制到第二个子图
    # ax2.set_xlim(x1_lim)
    # ax2.set_ylim(y1_lim)
    # ax2.set_zlim(z1_lim)
    # plt.pause(1)
    # plt.show()

    return nrmse, r2, data2


def evaluate_model3d(nfreqs, c_mid, filename1, point_cloud_path, missingrate, data_series, last_time=0):
    # Load model
    setup_seed(0)
    if nfreqs == 0:
        model = Siren(in_features=3, out_features=3, hidden_features=[c_mid, c_mid, c_mid, c_mid],
                      outermost_linear=True).cuda()
    elif nfreqs == 1:
        model = Siren(in_features=32, out_features=5, hidden_features=[c_mid, c_mid, c_mid, c_mid],
                      outermost_linear=True).cuda()
    else:
        model = Siren(in_features=nfreqs * 4, out_features=1, hidden_features=[c_mid, c_mid, c_mid, c_mid],
                      outermost_linear=True).cuda()

    folder_path = './pth/'
    filename = filename1 + '.pth'
    file_path = os.path.join(folder_path, filename)
    model.load_state_dict(torch.load(file_path))
    model.eval()

    mat = scipy.io.loadmat(point_cloud_path + '_test_' + str(missingrate) + '.mat')
    test_np = mat[data_series + "_test_" + str(missingrate)]

    test_torch = torch.from_numpy(test_np).float().cuda()

    # Make predictions
    if nfreqs == 0:
        output = model(test_torch[:, :3])
    else:
        test_fourier = fourier_features(test_torch, nfreqs)
        output = model(test_fourier)

    out = output.detach().cpu().numpy()
    mae = np.mean(np.abs(out - test_np[:, 3:]))
    mse = np.mean((out - test_np[:, 3:]) ** 2)
    rmse = np.sqrt(mse)
    max_value = torch.max(test_torch[:, 3:])
    min_value = torch.min(test_torch[:, 3:])
    target_range = max_value - min_value
    nrmse = rmse / target_range
    r2 = r2_score(test_np[:, 3:], out)
    # data1 = np.concatenate([test_data1, test_data_gt], axis=1)
    data2 = np.concatenate([test_np[:, :3], out], axis=1)
    # fig = plt.figure()
    # ax1 = fig.add_subplot(121, projection='3d')
    # ax2 = fig.add_subplot(122, projection='3d')
    #
    # # 第一个子图
    # x1 = data1[:,0]
    # y1 = data1[:,1]
    # z1 = data1[:,2]
    # ax1.scatter(x1, y1, z1, c='red')
    # ax1.set_xlabel('x')
    # ax1.set_ylabel('y')
    # ax1.set_zlabel('z')
    #
    # # 第二个子图
    # x2 = data2[:,0]
    # y2 = data2[:,1]
    # z2 = data2[:,2]
    # ax2.scatter(x2, y2, z2, c='blue')
    # ax2.set_xlabel('x')
    # ax2.set_ylabel('y')
    # ax2.set_zlabel('z')
    #
    # # 获取第一个子图的轴限制
    # x1_lim = ax1.get_xlim()
    # y1_lim = ax1.get_ylim()
    # z1_lim = ax1.get_zlim()
    #
    # # 应用轴限制到第二个子图
    # ax2.set_xlim(x1_lim)
    # ax2.set_ylim(y1_lim)
    # ax2.set_zlim(z1_lim)
    # plt.pause(1)
    # plt.show()

    # data_dict = {
    #     'clean_data': data2,  # 你要保存的数据
    #     'time': str(last_time),  # 当前时间信息
    # }
    # # 指定.mat文件的保存路径和文件名
    # mat_file_path = f'out_test_siren_{filename1}.mat'
    # folder_path = './mat_3d/'
    # scipy.io.savemat(os.path.join(folder_path, mat_file_path), data_dict)

    return mae, mse, rmse, nrmse, r2
