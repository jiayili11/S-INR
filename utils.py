import torch
from torch.utils.data import DataLoader, Dataset
import cv2 as cv
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import random
import numpy as np
import matplotlib.pyplot as plt
import time
import torch.nn as nn
import scipy.io


def get_device():
    if torch.cuda.is_available():
        de = 'cuda:0'
    else:
        de = 'cpu'
    return de


class data_load(Dataset):
    def __init__(self, filename):
        super().__init__()
        img = cv.imread(filename)
        self.img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {'gt': self.img_tensor}


def get_meshgrid(spatial_size):
    X, Y = np.meshgrid(np.arange(0, spatial_size[1]) / float(spatial_size[1] - 1),
                       np.arange(0, spatial_size[0]) / float(spatial_size[0] - 1))
    meshgrid = np.concatenate([X[None, :], Y[None, :]])
    return torch.from_numpy(meshgrid).unsqueeze(0)


def get_meshgrid_slic(spatial_size):
    X, Y = np.meshgrid(np.arange(0, spatial_size[1]) / float(spatial_size[1] - 1),
                       np.arange(0, spatial_size[0]) / float(spatial_size[0] - 1))
    meshgrid = np.concatenate([X[None, :], Y[None, :]])
    meshgrid = torch.from_numpy(meshgrid)
    meshgrid_sum = torch.cat([meshgrid, meshgrid], 0)  # 4
    for i in range(spatial_size[2] - 2):
        meshgrid_sum = torch.cat([meshgrid_sum, meshgrid], 0)
    return meshgrid_sum.unsqueeze(0)


def get_mgrid(sidelen):
    mgrid = torch.stack(
        torch.meshgrid(torch.linspace(-1, 1, steps=sidelen[0]), torch.linspace(-1, 1, steps=sidelen[1])), dim=-1)
    mgrid = mgrid.reshape(-1, 2)
    return mgrid.unsqueeze(0)


def get_mgrid_slic(sidelen):
    mgrid = torch.stack(
        torch.meshgrid(torch.linspace(-1, 1, steps=sidelen[0]), torch.linspace(-1, 1, steps=sidelen[1])), dim=-1)
    mgrid = mgrid.reshape(-1, 2)  # 2
    mgrid_sum = torch.cat([mgrid, mgrid], 1)  # 4
    for i in range(sidelen[2] - 2):
        mgrid_sum = torch.cat([mgrid_sum, mgrid], 1)
    return mgrid_sum.unsqueeze(0)


def fill_noise(x, noise_type):
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_()
    else:
        assert False


def get_noise(input_depth, spatial_size, noise_type='u', var=1. / 10):
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    shape = [1, input_depth, spatial_size[0], spatial_size[1]]
    net_input = torch.zeros(shape)
    fill_noise(net_input, noise_type)
    net_input *= var
    return net_input


def pre_process_slic(num_label, segments, pixels, slic_mask, slic_data, slic_set):
    max_w = 0
    max_h = 0
    for i in range(1, num_label + 1):
        a = np.array(segments == i)
        a1 = torch.Tensor(a.reshape(a.shape[0], a.shape[1]))
        a2 = torch.stack((a1, a1, a1), 0).permute(1, 2, 0)
        b = a2 * pixels
        w, h = [], []
        for x in range(b.shape[0]):
            for y in range(b.shape[1]):
                if b[x][y][0] != 0:
                    h.append(x)
                    w.append(y)
        slic_mask.append(a1)
        slic_data.append([min(h), max(h), min(w), max(w)])
        if (max(h) - min(h)) > max_h:
            max_h = max(h) - min(h)
        if (max(w) - min(w)) > max_w:
            max_w = max(w) - min(w)
        c = b[min(h):max(h), min(w):max(w)]
        e = c.reshape(c.shape[0], c.shape[1], 3)
        slic_set.append(e)
        # plt.figure()
        # plt.imshow(np.stack([e[:, :, 2], e[:, :, 1], e[:, :, 0]], 2))
        # plt.title("Recovered")
        # plt.show()
    if max_h % 2 != 0:
        max_h = max_h + 1
    if max_w % 2 != 0:
        max_w = max_w + 1
    return slic_mask, slic_data, slic_set, max_h, max_w


def get_input_inr(spatial_1, spatial_2):
    net_input = get_mgrid([spatial_1, spatial_2])
    return net_input


def get_input_inr_slic(spatial_1, spatial_2, c):
    net_input = get_mgrid_slic([spatial_1, spatial_2, c])
    return net_input


def get_input(input_depth, input_dict):
    method = input_dict['method']
    spatial_size = input_dict['input_size']
    noise_type = input_dict['noise_type']
    var = input_dict['var']
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)
        fill_noise(net_input, noise_type)
        net_input *= var
    elif method == 'meshgrid':
        assert input_depth == 2
        net_input = get_meshgrid(spatial_size)
    elif method == 'meshgrid2':
        spatial_size = input_dict['input_size2']
        net_input = get_mgrid([spatial_size, spatial_size])
    elif method == 'fourier':
        freqs = input_dict['base'] ** torch.linspace(0., input_dict['n_freqs'] - 1, steps=input_dict['n_freqs'])
        net_input = generate_fourier_feature_maps(freqs, spatial_size, only_cosine=input_dict['cosine_only'])
    else:
        assert False

    return net_input


def generate_fourier_feature_maps(freqs, spatial_size, only_cosine=False):
    meshgrid = get_meshgrid(spatial_size).permute(0, 2, 3, 1)
    vp = freqs * torch.unsqueeze(meshgrid, -1)
    if only_cosine:
        vp_cat = torch.cat((torch.cos(vp),), dim=-1)
    else:
        vp_cat = torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)
    return vp_cat.flatten(-2, -1).permute(0, 3, 1, 2)


def out_process_data(out, slic_data, target, slic_mask, num_label):  # B C H W
    out = out.squeeze().permute(1, 2, 0)  # H W C
    for i in range(num_label):
        sub = slic_data[i]  # [min(h), max(h), min(w), max(w)]
        sub_mask = slic_mask[i]
        for cha in range(3):
            for x in range(sub[0], sub[1]):
                for y in range(sub[2], sub[3]):
                    target[x, y, cha] = target[x, y, cha] + sub_mask[x, y] * out[x - sub[0], y - sub[2], cha + 3 * i]
        # plt.figure()
        # plt.imshow(np.stack([target[:, :, 2], target[:, :, 1], target[:, :, 0]], 2))
        # plt.title("Recovered")
        # plt.show()
    return target


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def cul_draw_img_gt_completion(out_np, img_gt, iter, loss, psnr_best, count_best, t1, t0, clean_image):
    Psnr_gt = psnr(img_gt, out_np)
    Ssim = ssim(img_gt, out_np, channel_axis=2)

    if Psnr_gt > psnr_best:
        psnr_best = Psnr_gt
        clean_image = out_np
        count_best = iter
        t1 = time.time()

    print('iter:%3.d, loss:%4.5f, psnr:%2.5f, ssim:%.5f, psnr_best:%2.5f, iter_best:%3.d, time:%4.5f.'
          % (iter, loss, Psnr_gt, Ssim, psnr_best, count_best, t1 - t0))
    if iter % 100 == 0 and iter != 0:
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        axes[0].imshow(np.clip(img_gt, 0, 1))
        axes[1].imshow(np.clip(out_np, 0, 1))
        # axes[2].imshow(np.clip(img_ggt, 0, 1))
        plt.show()
    return psnr_best, t1, count_best, clean_image


def cul_draw_img_gt(coords, out, img_gt, iter, psnr_best, count_best, t1, clean_image):
    return psnr_best, t1, count_best, clean_image


def cul_draw_img_gt_overlap(coords, out, img_gt, iter, loss, psnr_best, count_best, t1, t0, clean_image):
    # if iter % 50 == 0 and iter != 0:
    #     fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    #     axes[0].imshow(np.clip(img_gt, 0, 1))
    #     axes[1].imshow(np.clip(out_np, 0, 1))
    #     # axes[2].imshow(np.clip(img_ggt, 0, 1))
    #     plt.show()
    return psnr_best, t1, count_best, clean_image


def cul_draw_img_gt_noise(coords, out, img_gt, img_ob, iter, loss, psnr_best, count_best, t1, t0, clean_image):
    out_np = np.zeros_like(img_gt)
    mask = np.zeros_like(np.empty(img_gt.shape[:2]))
    for i in range(len(coords)):
        for j in range(coords[i].shape[1]):
            out_np[coords[i][0, j, 0], coords[i][0, j, 1], :] += out[i][j, :].cpu().detach().numpy()
            mask[coords[i][0, j, 0], coords[i][0, j, 1]] += 1
    out_np = out_np / mask[:, :, None]
    Psnr_gt = psnr(img_gt, out_np)
    Ssim = ssim(img_gt, out_np, channel_axis=2)
    if Psnr_gt > psnr_best:
        psnr_best = Psnr_gt
        clean_image = out_np
        count_best = iter
        t1 = time.time()
    print('iter:%3.d, loss:%4.5f, psnr:%2.5f, ssim:%.5f, psnr_best:%2.5f, iter_best:%3.d, time:%4.5f.'
          % (iter, loss, Psnr_gt, Ssim, psnr_best, count_best, t1 - t0))
    if iter % 100 == 0 and iter != 0:
        fig, axes = plt.subplots(1, 3, figsize=(30, 10))
        axes[0].imshow(np.clip(img_gt, 0, 1))
        axes[1].imshow(np.clip(clean_image, 0, 1))
        axes[2].imshow(np.clip(img_ob, 0, 1))
        plt.show()
    # if iter == 1000 and iter != 0:
    #     t1 = time.time()
    #     scipy.io.savemat("swi" + "_noise"+".mat", {"clean_image": clean_image, "time_used": t1 - t0})
    return psnr_best, t1, count_best, clean_image


def cul_draw_img_gt_siren(out, img_gt, iter, loss, psnr_best, count_best, t1, t0, clean_image):
    out_np = out.detach().numpy()
    Psnr_gt = psnr(img_gt, out_np)
    Ssim = ssim(img_gt, out_np, channel_axis=2)

    if Psnr_gt > psnr_best:
        psnr_best = Psnr_gt
        clean_image = out_np
        count_best = iter
        t1 = time.time()

    print('iter:%3.d, loss:%4.5f, psnr:%2.5f, ssim:%.5f, psnr_best:%2.5f, iter_best:%3.d, time:%4.5f.'
          % (iter, loss, Psnr_gt, Ssim, psnr_best, count_best, t1 - t0))
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))
    axes[0].imshow(np.clip(clean_image, 0, 1))
    axes[1].imshow(np.clip(out_np, 0, 1))
    axes[2].imshow(np.clip(img_gt, 0, 1))
    plt.show()

    return psnr_best, t1, count_best, clean_image


def cul_draw_vid_gt(coords, out, vid_gt, iter, loss, psnr_best, count_best, t1, t0, clean_image):
    out_np = np.zeros_like(vid_gt)
    for i in range(len(coords)):
        for j in range(coords[i].shape[1]):
            out_np[coords[i][0, j, 0], coords[i][0, j, 1], coords[i][0, j, 2], :] = out[i][j, :].detach().numpy()
    Psnr = 0
    Ssim = 0
    for i in range(vid_gt.shape[0]):
        Psnr += psnr(vid_gt[i], out_np[i])
        Ssim += ssim(vid_gt[i], out_np[i], channel_axis=2)
    Psnr /= vid_gt.shape[0]
    Ssim /= vid_gt.shape[0]

    if Psnr > psnr_best:
        psnr_best = Psnr
        clean_image = out_np
        count_best = iter
        t1 = time.time()

    print('iter:%3.d, loss:%4.5f, psnr:%2.5f, ssim:%.5f, psnr_best:%2.5f, iter_best:%3.d, time:%4.5f.'
          % (iter, loss, Psnr, Ssim, psnr_best, count_best, t1 - t0))
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))
    axes[0].imshow(np.clip(clean_image[0], 0, 1))
    axes[1].imshow(np.clip(out_np[0], 0, 1))
    axes[2].imshow(np.clip(vid_gt[0], 0, 1))
    plt.show()

    return psnr_best, t1, count_best, clean_image

