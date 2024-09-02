from __future__ import print_function
import configargparse
from models import *
import numpy as np
from torch.utils.data import DataLoader
from test_swi import perform_inference_uv_2d, preprocess_test
import time
import torch.optim
import os
from sklearn.neighbors import KNeighborsRegressor
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.segmentation import slic, mark_boundaries
os.environ['OMP_NUM_THREADS'] = '10'
torch.backends.cudnn.enabled = True  # type: ignore
torch.backends.cudnn.benchmark = True  # type: ignore
dtype = torch.cuda.FloatTensor
p = configargparse.ArgumentParser()
p.add_argument('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')
p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
# General training options
p.add_argument('--epochs', type=int, default=2001)
p.add_argument('--show_epochs', type=int, default=200)
p.add_argument('--seed', type=int, default=0, help='Checkpoint to trained model.')
opt = p.parse_args()
for data_series in [1]:  #
    point_data_path = './datas/rain_data/data_' + str(data_series) + '_'
    for missingrate in [0.75, 0.8, 0.85, 0.9]:  #
        for num in [15]:
            for r in [30]:
                t_in = time.time()
                mat = scipy.io.loadmat(point_data_path + 'train_' + str(missingrate) + '.mat')
                train = mat["train"]
                pre_dataset = point2d_pre(point_data_path, num, data_series, missingrate)
                dataloader = DataLoader(pre_dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)
                clusters = pre_dataset.cluster_gt
                centers = pre_dataset.cluster_centers
                centers_np = [c.cpu().numpy() for c in centers]
                center_name = f'data_{data_series}_centers_{num}_{missingrate}.npy'
                folder_path = './centers/'
                np.save(os.path.join(folder_path, center_name), centers_np)
                print("\n", point_data_path, "missingrate", missingrate)
                print("pixel_num", num, "average num of cluster", int(pre_dataset.coords.shape[0] / num), "missingrate",
                      missingrate)
                input_cluster_cor, input_cluster_pre, pixel_count = point_2d_input_process(clusters, num) #train
                print('datas load and process costs %.5f seconds' % (time.time() - t_in))
                c_mid = [60]
                pre_tensor = torch.cat(input_cluster_pre, dim=0).detach()
                learning_rates = [5e-5, 1e-4, 5e-4, 1e-3]
                weight_decay = 1e-6
                rank = int(r * pre_dataset.coords.shape[0] / num)
                for omega in [30]:
                    for cmid in c_mid:
                        for lr in learning_rates:
                            net = NET_S_INR(pixel_num=num, c_in=2, r=r, c_out=5, c_mid=cmid,
                                          hidden_layers=3,
                                          first_omega_0=omega,
                                          hidden_omega_0=omega).cuda()
                            p = [x for x in net.parameters()]
                            num_params = sum([param.numel() for param in p])
                            print("\n", "cmid", cmid, "Number of params:%.3fMb" % (num_params * 4 / 1024 / 1024),
                                  "weight_decay"
                                  , weight_decay, "learning rates", lr, "rank", r)
                            optim = torch.optim.Adam(lr=lr, params=net.parameters(), weight_decay=weight_decay)
                            count = 0
                            best_nrmse = float('inf')
                            best_r2 = -float('inf')
                            best_time = 0
                            best_data = 0
                            t0 = time.time()
                            cluster_cor_test, prcp_np, test_xy = preprocess_test(num, centers_filename=center_name,
                                                                                     missingrate=missingrate,
                                                                                     data_series=data_series,
                                                                                     point_data_path=point_data_path)
                            for iteration in range(opt.epochs):
                                output = net(input_cluster_cor)
                                loss = 0
                                for j in range(num):
                                    loss += torch.norm(output[j] - input_cluster_pre[j], 2)
                                if iteration % 40 == 0 and iteration != 0:
                                    filename = f'train_swi_{data_series}_{lr}_{cmid}_{iteration}_{num}.pth'
                                    folder_path = './pth/'
                                    torch.save(net.state_dict(), os.path.join(folder_path, filename))
                                    t1 = time.time()
                                    nrmse, r2, data2 = perform_inference_uv_2d(num=num, c_mid=cmid, r=r,
                                                                               filename1=f'train_swi_{data_series}_{lr}_{cmid}_{iteration}_{num}',
                                                                               missingrate=missingrate,
                                                                               input_cluster_cor=cluster_cor_test,
                                                                               prcp_np=prcp_np,
                                                                               test_xy=test_xy,
                                                                               data_series=data_series)
                                    if iteration % 1000 == 0 and iteration != 0:
                                        print(filename, "iter", iteration, "loss", loss.item(),
                                              "nrmse:", nrmse.item(), "r2:", r2.item(), "time", t1 - t0)
                                    if nrmse < best_nrmse:
                                        t1 = time.time()
                                        best_nrmse = nrmse
                                        best_data = data2
                                        count = iteration
                                        best_time = t1 - t0
                                    if r2 > best_r2:
                                        best_r2 = r2
                                if iteration == opt.epochs - 1:
                                    print("iter", iteration, "best_iter", count, "loss", loss.item(), "nrmse:",
                                          best_nrmse.item(), "r2:", best_r2.item(), "time",
                                          best_time)
                                    data_dict = {
                                        'clean_data': np.concatenate([best_data, train], axis=0),
                                        'time': str(best_time),
                                    }
                                    mat_file_path = f'datas{data_series}_{missingrate}_sacinr_{cmid}{num}{lr}.mat'
                                    folder_path = './mat_2d/'
                                    scipy.io.savemat(os.path.join(folder_path, mat_file_path), data_dict)
                                optim.zero_grad()
                                loss.backward()
                                optim.step()
