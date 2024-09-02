from dataloader import *
from models import *
from modules import *
import time
import configargparse
import warnings
import math
import torch.fft as fft

warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
p = configargparse.ArgumentParser()
p.add_argument('--data_path', type=str, default="./datas/completion/")
p.add_argument('--lr', type=float, default=1e-5)
p.add_argument('--epochs', type=int, default=8001)
p.add_argument('--epochs_til_ckpt', type=int, default=100)
p.add_argument('--steps_til_summary', type=int, default=100)
p.add_argument('--model_type', type=str, default='sine', help='')
p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
p.add_argument('--seed', type=int, default=0, help='Checkpoint to trained model.')
opt = p.parse_args()
pixel_set = 2500
h = 5
freq = 0
r = 120
cmid = 35
omega = 150
for dataname in ['mor']:
    for case in ['case1']:  # 'case1','case2'
        for lr in [1e-3]:  # 1e-5, 5e-5, 1e-4, 5e-4, 1e-3
            t0 = time.time()
            data_path = opt.data_path + dataname + case
            gt_path = opt.data_path + dataname + "gt"
            mask_path = opt.data_path + dataname + case + "mask"
            data_test = data_load_img_completion_overlap_bound_kmeans(data_path, gt_path, mask_path,
                                                                      pixel_set=pixel_set, freq=freq)
            dataloader = DataLoader(data_test, batch_size=1, pin_memory=False, num_workers=0)
            datas = next(iter(dataloader))
            img_gt = datas['img'][0].numpy()
            pixel_count = datas['pixel_count'].squeeze()
            coords_int = data_test.coords1
            pixel_num = int(datas['pixel_num'])
            gt = data_test.gt
            gt_mask = data_test.gt_mask
            gt_ob = datas['gt_ob']
            ave_num = datas['img'][0].shape[0] * datas['img'][0].shape[1] / datas['pixel_num']
            c_in = 2 if freq == 0 else 4 * freq
            net = NET_S_INR(pixel_count, pixel_num, c_in=c_in, r=r, c_out=31, c_mid=cmid,
                            hidden_layers=h - 2,
                            first_omega_0=omega, hidden_omega_0=omega).cuda()
            optim = torch.optim.Adam(lr=lr, params=net.parameters(), weight_decay=1e-6)
            p = [x for x in net.parameters()]
            num_params = sum([param.numel() for param in p])
            print('superpixel num:%d, average pixel num:%d.' % (pixel_num, ave_num))
            print("Number of params:%.3fMb" % (num_params * 4 / 1024 / 1024),
                  'datas load costs %.5f seconds' % (time.time() - t0))
            psnr_best = 0
            count_best = 0
            clean_image = 0
            best_time = 0
            img_gt_torch = torch.from_numpy(img_gt).cuda().float()
            gt_all = torch.cat(gt, dim=0).cuda().float()
            coord_all = torch.cat(coords_int, dim=0).cuda()
            coord_np = coord_all.cpu().numpy()
            unique, counts = np.unique(coord_np, axis=0, return_counts=True)
            point_counts = dict(zip(map(tuple, unique), counts))
            weight_all = torch.zeros_like(gt_all).float().cuda()
            for i, coord in enumerate(coord_all):
                count = point_counts[tuple(coord.tolist())]
                weight_all[i, :] = 1. / torch.sqrt(torch.tensor(count))
            t1 = time.time()
            input_coords = data_test.coords2 if freq == 0 else data_test.coordsf
            for iteration in range(opt.epochs):
                out = net(input_coords)
                loss = 0
                for j in range(pixel_num):
                    loss += torch.norm(gt_mask[j] * out[j] - gt_mask[j] * gt[j], 2)
                if iteration % 50 == 0 and iteration != 0:
                    out_tensor = torch.cat(out, dim=0).cuda().float()
                    mse = torch.mean((weight_all * (out_tensor - gt_all)) ** 2)
                    Psnr_gt = 10 * math.log10(1 / mse)
                    if Psnr_gt > psnr_best:
                        psnr_best = Psnr_gt
                        clean_image = out
                        count_best = iteration
                        best_time = time.time() - t1
                    if iteration % 4000 == 0 and iteration != 0:
                        print(dataname, case, omega, "loss", loss.item(), "psnr", Psnr_gt, "time", time.time() - t1)
                    if iteration == opt.epochs - 1:
                        mask = torch.zeros_like(torch.empty(img_gt_torch.shape[:2])).cuda()
                        out_torch = torch.zeros_like(img_gt_torch).cuda().float()
                        for i in range(len(coords_int)):
                            for j in range(coords_int[i].size(0)):
                                out_torch[coords_int[i][j, 0], coords_int[i][j, 1], :] += clean_image[i][j, :]
                                mask[coords_int[i][j, 0], coords_int[i][j, 1]] += 1
                        out_torch = out_torch / mask[:, :, None]
                        out_np = out_torch.detach().cpu().numpy()
                        Ssim = ssim(img_gt, out_np, data_range=1, channel_axis=2)
                        Psnr_last = psnr(img_gt, out_np)
                        folder_path = './mat_image/'
                        data_dict = {'clean_data': out_np,
                            'time': str(best_time)}
                        mat_file_path = f'{dataname}_{case}_sinr_{pixel_num}.mat'
                        scipy.io.savemat(os.path.join(folder_path, mat_file_path), data_dict)
                        print(dataname, case, cmid, omega, lr, "loss", loss.item(), "psnr_best",
                              Psnr_last, "ssim_best", Ssim, "best_iter", count_best, "time", best_time)
                        # fig, axes = plt.subplots(1, 3, figsize=(20, 10))
                        # axes[0].imshow(np.clip(data_test.img_ob(:,:,[10, 20 ,30]), 0, 1))
                        # axes[1].imshow(np.clip(out_np[], 0, 1))
                        # axes[2].imshow(np.clip(img_gt[], 0, 1))
                        # plt.show()
                optim.zero_grad()
                loss.backward()
                optim.step()
