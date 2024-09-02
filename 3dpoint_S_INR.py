import os
import time
from sklearn.metrics import r2_score
import configargparse
import numpy as np
import torch
import torch.nn.functional as F
from datatio import *
from modules import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from test_swi_3d import perform_inference_3d_uv, preprocess_swi_3d
# ok<*UNRCH>
p = configargparse.ArgumentParser()
p.add_argument('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')
p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, default='./storage',
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')
# General training options
p.add_argument('--epochs', type=int, default=6001)
p.add_argument('--point_cloud_path', type=str, default='./data/point_3d/Squirrel')
p.add_argument('--seed', type=int, default=0)
opt = p.parse_args()
setup_seed(opt.seed)
for data_series in ['Scene1']:  # , 'Scene3', 'Scene4'
    point_cloud_path = './datas/point_3d/' + data_series
    for missingrate in [0.9, 0.925, 0.95, 0.975]:
        for num in [12]:
            for r in [20]:
                t_ini = time.time()
                mat = scipy.io.loadmat(point_cloud_path + "_train_" + str(missingrate) + '.mat')
                train = mat[data_series + "_train_" + str(missingrate)]
                dataset = point_pre_3d(point_cloud_path, missingrate=missingrate, data_series=data_series, num=num)
                dataloader = DataLoader(dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)
                clusters = dataset.clusters
                centers = dataset.cluster_centers
                centers_np = [c.cpu().numpy() for c in centers]
                center_name = f'{data_series}_train_centers_{dataset.num}_{missingrate}.npy'
                folder_path = './centers/'
                np.save(os.path.join(folder_path, center_name), centers_np)
                coordinate_in, color_in, pixel_count = point_3d_input_process(clusters, dataset.num)
                print("\n", point_cloud_path, "missingrate", missingrate)
                print("pixel_num", dataset.num, "average num of cluster", int(dataset.train.shape[0] / dataset.num),
                      "missingrate", missingrate)
                print("data load and process costs %.5f seconds" % (time.time() - t_ini))
                cluster_cor_test, input_cluster_cor_ini, pre_tensor, max_value, min_value, pre_np = preprocess_swi_3d(num,
                                                                                               point_cloud_path=point_cloud_path,
                                                                                               centers_filename=center_name,
                                                                                               data_series=data_series,
                                                                                               missingrate=missingrate)
                rgb_gt = torch.cat(color_in, dim=0)
                save = 0
                c_mid = [60]
                weight_decay = 1e-6
                learning_rates = [5e-5]
                rank = int(r * dataset.train.shape[0] / dataset.num)
                for omega in [30]:
                    for cmid in c_mid:
                        for lr in learning_rates:
                            net = NET_S_INR(pixel_num=int(dataset.num), c_in=3, r=r, c_out=3, c_mid=cmid,
                                          hidden_layers=3,
                                          first_omega_0=omega,
                                          hidden_omega_0=omega).cuda()
                            optim = torch.optim.Adam(lr=lr, params=net.parameters(), weight_decay=weight_decay)
                            p = [x for x in net.parameters()]
                            num_params = sum([param.numel() for param in p])
                            print("\n", "cmid", cmid, "Number of params:%.3fMb" % (num_params * 4 / 1024 / 1024),
                                  "weight_decay", weight_decay, "learning rates", lr, "rank", r)
                            best_nrmse = float('inf')
                            best_r2 = -float('inf')
                            best_result = 0
                            count = 0
                            best_time = 0
                            t0 = time.time()
                            for iteration in range(opt.epochs):
                                output_rgb = net(coordinate_in)
                                loss = 0
                                for j in range(dataset.num):
                                    loss += torch.norm(output_rgb[j] - color_in[j], 2)
                                if iteration % 40 == 0 and iteration != 0:
                                    filename = f'{data_series}_train_swi_{lr}_{cmid}_{iteration}_{dataset.num}.pth'
                                    folder_path = './pth/'
                                    torch.save(net.state_dict(), os.path.join(folder_path, filename))
                                    t1 = time.time()
                                    nrmse, r2, data2 = perform_inference_3d_uv(num=dataset.num, c_mid=cmid,
                                                                               rank=r,
                                                                               filename1=f'{data_series}_train_swi_{lr}_{cmid}_{iteration}_{dataset.num}',
                                                                               cluster_cor_test=cluster_cor_test,
                                                                               pre_tensor=pre_tensor,
                                                                               max_value=max_value,
                                                                               min_value=min_value,
                                                                               pre_np=pre_np,cor_ini = input_cluster_cor_ini,
                                                                               last_time=t1 - t0)
                                    if iteration % 2000 == 0 and iteration != 0:
                                        print(filename, "iter", iteration, "loss", loss.item(),
                                              "nrmse:", nrmse.item(), "r2:", r2.item(), "time", t1 - t0)
                                    if nrmse < best_nrmse:
                                        t1 = time.time()
                                        best_nrmse = nrmse
                                        best_result = data2
                                        count = iteration
                                        best_time = t1 - t0
                                    if r2 > best_r2:
                                        best_r2 = r2
                                if iteration == opt.epochs - 1:
                                    clean_data = np.concatenate([best_result, train], axis=0)
                                    print("iter", iteration, "best_iter", count, "loss", loss.item(), "nrmse:",
                                          best_nrmse.item(), "r2:", best_r2.item(), "time", best_time)
                                    data_dict = {
                                        'clean_data': clean_data,
                                        'time': str(best_time),
                                    }
                                    mat_file_path = f'{data_series}_{missingrate}_sacinr_{num}{cmid}.mat'
                                    folder_path = './mat_3d/'
                                    scipy.io.savemat(os.path.join(folder_path, mat_file_path), data_dict)
                                optim.zero_grad()
                                loss.backward()
                                optim.step()

