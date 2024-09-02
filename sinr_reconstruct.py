from dataloader import *
from models import *
from modules import *
import time
import configargparse
import warnings
import math

warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
p = configargparse.ArgumentParser()
p.add_argument('--data_path', type=str, default="./datas/reconstruct/")
p.add_argument('--lr', type=float, default=1e-5)
p.add_argument('--epochs', type=int, default=10001)
p.add_argument('--epochs_til_ckpt', type=int, default=100)
p.add_argument('--steps_til_summary', type=int, default=100)
p.add_argument('--model_type', type=str, default='sine', help='')
p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
p.add_argument('--seed', type=int, default=0, help='Checkpoint to trained model.')
opt = p.parse_args()
pixel_set = 8000
h = 5
freq = 0
r = 120
lr = 5e-4
omega = 300
cmid = 35
for dataname in ["kodim24"]:  # kodim24
    for lr in [5e-4]:  # 1e-5, 5e-5, 1e-4, 5e-4, 1e-3
                t0 = time.time()
                data_path = opt.data_path + dataname
                data_test = data_load_img(data_path, pixel_set=pixel_set)
                dataloader = DataLoader(data_test, batch_size=1, pin_memory=False, num_workers=0)
                ave_num = data_test.img.shape[0] * data_test.img.shape[1] / data_test.pixel_num
                c_gt = data_test.img.shape[2]
                c_in = 2 if freq == 0 else 4 * freq
                net = NET_S_INR(data_test.pixel_num, c_in=c_in, r=r, c_out=c_gt, c_mid=cmid,
                                hidden_layers=h - 2,
                                first_omega_0=omega, hidden_omega_0=omega).cuda()
                optim = torch.optim.Adam(lr=lr, params=net.parameters())
                params = net.parameters()
                p = [x for x in net.parameters()]
                num_params = sum([param.numel() for param in p])
                coords_int = data_test.coords1
                gt = data_test.gt
                pixel_count = data_test.count.squeeze()
                img_gt_np = data_test.img
                img_gt_torch = torch.from_numpy(img_gt_np).cuda().float()
                psnr_best = 0
                count_best = 0
                clean_image = 0
                best_time = 0
                t1 = time.time()
                coords_int_all = torch.cat(coords_int, dim=0).cuda().float().long()
                print('superpixel num:%d, average pixel num:%d.' % (data_test.pixel_num, ave_num),
                      "Number of params:%.3fMb" % (num_params * 4 / 1024 / 1024),
                      'datas load costs %.5f seconds' % (time.time() - t0))
                gt_all = torch.cat(gt, dim=0).cuda().float()
                input_coords = data_test.coords2 if freq == 0 else data_test.coordsf
                for iteration in range(opt.epochs):
                    out = net(input_coords)
                    loss = 0
                    for j in range(data_test.pixel_num):
                        loss += torch.norm(out[j] - gt[j], 2)
                    if iteration % 50 == 0 and iteration != 0:
                        out_tensor = torch.cat(out, dim=0).cuda().float()
                        mse = torch.mean(((out_tensor - gt_all)) ** 2)
                        Psnr_gt = 10 * math.log10(1 / mse)
                        if Psnr_gt > psnr_best:
                            psnr_best = Psnr_gt
                            clean_image = out_tensor
                            count_best = iteration
                            best_time = time.time() - t1
                        if iteration % 5000 == 0 and iteration != 0:
                            print(dataname, lr, omega, r, "loss", loss.item(), "psnr", psnr_best, "best_iter",
                                  count_best, "time", time.time() - t1)
                        if iteration == opt.epochs - 1:
                            out_torch = torch.zeros_like(img_gt_torch).cuda().float()
                            out_torch[coords_int_all[:, 0], coords_int_all[:, 1], :] = clean_image
                            clean_image_np = out_torch.detach().cpu().numpy()
                            Psnr_gt = psnr(img_gt_np, clean_image_np)
                            Ssim = ssim(img_gt_np, clean_image_np, data_range=1, channel_axis=2)
                            print(dataname, lr, omega, r, "loss", loss.item(), "psnr_best", Psnr_gt, "ssim_best",
                                  Ssim, "best_iter", count_best, "time", best_time)
                            folder_path = './mat_image/'
                            data_dict = { 'clean_data': clean_image_np,
                                'time': str(best_time)}
                            mat_file_path = f'{dataname}_sinr_{lr}.mat'
                            scipy.io.savemat(os.path.join(folder_path, mat_file_path), data_dict)
                        # fig, axes = plt.subplots(1, 2, figsize=(20, 10))
                        # axes[0].imshow(np.clip(clean_image_np, 0, 1))
                        # axes[1].imshow(np.clip(img_gt_np, 0, 1))
                        # # axes[2].imshow(np.clip(img_gt, 0, 1))
                        # plt.show()
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
