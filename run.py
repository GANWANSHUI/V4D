# -*- encoding: utf-8 -*-
import os, sys, copy, time, random, argparse, pdb
from tqdm import tqdm, trange
import mmcv, shutil
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from lib import utils
from lib import V4D as V4D
from lib.load_data import load_data


def config_parser():
    '''Define command line arguments
    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True,
                        help='config file path')
    parser.add_argument("--seed", type=int, default=777,
                        help='Random seed')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--no_reload_optimizer", action='store_true',
                        help='do not reload optimizer state from saved ckpt')
    parser.add_argument("--ft_path", type=str, default='',
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--export_bbox_and_cams_only", type=str, default='',
                        help='export scene bbox and camera poses for debugging and 3d visualization')
    parser.add_argument("--export_coarse_only", type=str, default='')

    # testing options
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true')
    parser.add_argument("--render_train", action='store_true')
    parser.add_argument("--render_video", action='store_true')
    parser.add_argument("--render_video_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # logging/saving options
    parser.add_argument("--i_print", type=int, default=500,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=100000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_vis_weight", type=int, default=1,
                        help='frequency of weight ckpt saving')

    # experiment setting
    parser.add_argument("--density_feat_size", type=int, default=12, help='number of the voxel feature')
    parser.add_argument("--rgbnet_width", type=int, default=256, help='width of the mlp')
    parser.add_argument("--N_rand", type=int, default=4196, help='batch size (number of random rays per optimization step)')
    parser.add_argument("--rgbnet_depth", type=int, default=5, help='depth of the mlp')


    parser.add_argument("--savename", type=str, action='append', default=[],
                        help='discard')

    parser.add_argument("--weight_density_tv", type=float, default=0.1, help='weight_density_tv')
    parser.add_argument("--weight_rgb_tv", type=float, default=0.1, help='weight_rgb_tv')


    parser.add_argument("--weight_main0", type=float, default=0.0, help='weight for the lut')
    parser.add_argument("--weight_main1", type=float, default=1.0, help='weight for the lut')


    parser.add_argument("--tv_decay_factor", type=float, default=0.005, help='decay factor for weight_density_tv')
    parser.add_argument("--tv_from", type=int, default=1, help='tv loss from N iters')
    parser.add_argument("--tv_every", type=int, default=1, help='tv loss interval')
    parser.add_argument("--cuda_tv", type=lambda x: x.lower() == 'true', default=False, help='enable the cuda_tv')

    parser.add_argument("--decay_density_tv", type=lambda x: x.lower() == 'true', default=True,
                        help='decay loss weight to tv_decay_factor')
    parser.add_argument("--decay_rgb_tv", type=lambda x: x.lower() == 'true', default=True,
                        help='decay loss weight to tv_decay_factor')

    parser.add_argument("--feature_shift", type=lambda x: x.lower() == 'true', default=True,
                        help='use feature_shift or not')
    parser.add_argument("--feature_t_shift", type=lambda x: x.lower() == 'true', default=True,
                        help='use feature_t_shift or not')
    parser.add_argument("--search_geometry", type=lambda x: x.lower() == 'true', default=False,
                        help='use search_geometry or not')


    parser.add_argument("--dual_voxel", type=lambda x: x.lower() == 'true', default=True,
                        help='use dual_voxel or not, separate the density voxel and the flow and rgb voxel')



    parser.add_argument("--viewbase_pe", type=int, default=4, help='position encoding for the view direction ')
    parser.add_argument("--view_direction", type=lambda x: x.lower() == 'true', default=True, help='use view_direction or not')
    # loss
    parser.add_argument("--voxel_lr", type=float, default=0.1, help='use two_rgb_loss in lut or not')



    parser.add_argument("--basedir", type=str, default='./logs',
                        help='basedir for save the network')
    parser.add_argument("--coarse_iters", type=int, default=10000, help='number of coarse_iters')
    parser.add_argument("--fine_iters", type=int, default=250000, help='number of fine_iters')

    parser.add_argument("--total_data_device", type=str, default='cuda', help='load total dynamic data to gpu or cpu ')

    parser.add_argument("--static_type", type=str, default='dynamic', help='dynamic')

    parser.add_argument("--coarse_index", type=int, default=9, help='time interval for the fine bounding box')
    parser.add_argument("--view_dependent", type=str, default='post', help='using [pre, post] view dependent in rgb')

    parser.add_argument("--video_only", type=str, default='no', help='the path of the pretrain model [no, ]')


    return parser


@torch.no_grad()
def render_viewpoints(model, render_poses, HW, Ks, ndc, render_kwargs,
                      gt_imgs=None, savedir=None, render_factor=0,
                      eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False, eval_mse=False, times=None,
                      data_record=None, render_type='test'):
    '''Render images for the given viewpoints; run evaluation if gt given.
    '''
    assert len(render_poses) == len(HW) and len(HW) == len(Ks)

    if render_factor != 0:
        HW = np.copy(HW)
        Ks = np.copy(Ks)
        HW //= render_factor
        Ks[:, :2, :3] //= render_factor

    rgbs = []
    normals = []
    disps = []
    psnrs = []
    ssims = []
    lpips_alex = []
    lpips_vgg = []
    mse = []
    total_inference_time = []

    render_poses = render_poses.to('cuda')
    times = times.to('cuda')

    for i, c2w in enumerate(tqdm(render_poses)):

        H, W = HW[i]
        K = Ks[i]

        # start counting time
        eps_time = time.time()

        rays_o, rays_d, viewdirs = V4D.get_rays_of_a_view(
            H, W, K, c2w, ndc, inverse_y=render_kwargs['inverse_y'],
            flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)

        times_test = torch.ones([H, W, 1]) * times[i]


        keys = ['rgb_marched', 'disp']

        print('testing i:', i)

        # global_step = -1
        render_result_chunks = [
            {k: v for k, v in model(ro, rd, vd, rt, global_step=0, mode='test', **render_kwargs).items() if k in keys}
            for ro, rd, vd, rt in
            zip(rays_o.split(16, 0), rays_d.split(16, 0), viewdirs.split(16, 0), times_test.split(16, 0))]


        render_result = {
            k: torch.cat([ret[k][-1] for ret in render_result_chunks])
            for k in render_result_chunks[0].keys()
        }

        eps_time = time.time() - eps_time

        total_inference_time.append(eps_time)
        print('single image inference: finish (eps time:', eps_time, 'secs)')

        rgb = render_result['rgb_marched'].cpu().numpy()
        disp = render_result['disp'].cpu().numpy()

        rgbs.append(rgb)
        disps.append(disp)

        if i == 0:
            print('Testing', rgb.shape, disp.shape)

        if gt_imgs is not None and render_factor == 0:

            mse_i = np.square(rgb - gt_imgs[i])
            p = -10. * np.log10(np.mean(mse_i))
            psnrs.append(p)

            if eval_ssim:
                ssims.append(utils.rgb_ssim(rgb, gt_imgs[i], max_val=1))
            if eval_lpips_alex:
                lpips_alex.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='alex', device=c2w.device))
            if eval_lpips_vgg:
                lpips_vgg.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='vgg', device=c2w.device))
            if eval_mse:
                mse.append(mse_i)

        if savedir is not None:
            rgb8 = utils.to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.array(rgbs)
    disps = np.array(disps)

    if len(psnrs):
        mean_psnr = np.mean(psnrs)

        mean_total_inference_time = np.mean(total_inference_time)

        print('{} psnr avg = {}'.format(render_type, mean_psnr))

        print('{} inference time: avg = {}'.format(render_type, mean_total_inference_time))

        data_record.write('\n {} psnr avg = {} \n'.format(render_type, mean_psnr))
        data_record.write('{} inference time: avg = {} \n'.format(render_type, mean_total_inference_time))

        if eval_ssim:
            mean_ssim = np.mean(ssims)
            data_record.write('{} ssim avg = {} \n'.format(render_type, mean_ssim))
            print('{} ssim avg = {}'.format(render_type, mean_ssim))
        if eval_lpips_vgg:
            mean_lpips_vgg = np.mean(lpips_vgg)
            data_record.write('{} lpips_vgg avg = {} \n'.format(render_type, mean_lpips_vgg))
            print('{} lpips_vgg avg = {}'.format(render_type, mean_lpips_vgg))
        if eval_lpips_alex:
            mean_lpips_alex = np.mean(lpips_alex)
            data_record.write('{} lpips_alex avg = {} \n'.format(render_type, mean_lpips_alex))
            print('{} lpips_alex avg = {}'.format(render_type, mean_lpips_alex))

        if mse:
            mean_mse = np.mean(mse)
            data_record.write('{} mse avg = {} \n'.format(render_type, mean_mse))
            print('{} mse avg = {}'.format(render_type, mean_mse))

    return rgbs, disps


@torch.no_grad()
def render_viewpoints_hyper(model, data_class, render_kwargs, ndc,
                      savedir=None, render_factor=0,
                      eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False, eval_mse=False,
                      data_record=None, render_type='test'):
    '''Render images for the given viewpoints; run evaluation if gt given.
    '''

    rgbs = []
    disps = []
    psnrs = []
    ssims = []
    lpips_alex = []
    lpips_vgg = []
    mse = []
    total_inference_time = []

    rgbs_tensor = []
    rgbs_gt_tensor = []

    H = data_class['data_class'].h
    W = data_class['data_class'].w

    for i, j in enumerate(data_class['data_class'].i_test):


        rays_o, rays_d, viewdirs, gt_imgs, times = data_class['data_class'].load_idx_1(j, not_dic=True, ndc=ndc)
        # pdb.set_trace()
        times_test = torch.ones([H, W, 1], device='cuda') * times

        # start counting time
        eps_time = time.time()
        keys = ['rgb_marched', 'disp']
        render_result_chunks = [
            {k: v for k, v in model(ro, rd, vd, rt, global_step=0, mode='test', **render_kwargs).items() if k in keys}
            for ro, rd, vd, rt in
            zip(rays_o.split(16, 0), rays_d.split(16, 0), viewdirs.split(16, 0), times_test.split(16, 0))]

        render_result = {
            k: torch.cat([ret[k][-1] for ret in render_result_chunks])
            for k in render_result_chunks[0].keys()
        }

        eps_time = time.time() - eps_time

        total_inference_time.append(eps_time)
        print('single image inference: finish (eps time:', eps_time, 'secs)')

        rgb = render_result['rgb_marched'].cpu().numpy()
        disp = render_result['disp'].cpu().numpy()

        rgbs.append(rgb)
        disps.append(disp)

        if i == 0:
            print('Testing', rgb.shape, disp.shape)

        if gt_imgs is not None and render_factor == 0:
            gt_imgs = gt_imgs.cpu().numpy()
            mse_i = np.square(rgb - gt_imgs)
            p = -10. * np.log10(np.mean(mse_i))
            psnrs.append(p)

            rgbs_tensor.append(torch.from_numpy(np.clip(rgb, 0, 1)).permute(2,0,1))
            rgbs_gt_tensor.append(torch.from_numpy(np.clip(gt_imgs, 0, 1)).permute(2,0,1))
            # pdb.set_trace()
            if eval_ssim:
                ssims.append(utils.rgb_ssim(rgb, gt_imgs, max_val=1))
            if eval_lpips_alex:
                lpips_alex.append(utils.rgb_lpips(rgb, gt_imgs, net_name='alex', device=times_test.device))
            if eval_lpips_vgg:
                lpips_vgg.append(utils.rgb_lpips(rgb, gt_imgs, net_name='vgg', device=times_test.device))
            if eval_mse:
                mse.append(mse_i)

        if savedir is not None:
            rgb8 = utils.to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.array(rgbs)
    disps = np.array(disps)

    if 1:
        rgbs_tensor = torch.stack(rgbs_tensor, 0)
        rgbs_gt_tensor = torch.stack(rgbs_gt_tensor, 0)
        ms_ssims = ms_ssim(rgbs_gt_tensor, rgbs_tensor, data_range=1, size_average=True)

    if len(psnrs):
        mean_psnr = np.mean(psnrs)
        mean_total_inference_time = np.mean(total_inference_time)
        print('{} psnr avg = {}'.format(render_type, mean_psnr))
        print('{} inference time: avg = {}'.format(render_type, mean_total_inference_time))
        data_record.write('\n {} psnr avg = {} \n'.format(render_type, mean_psnr))
        data_record.write('{} inference time: avg = {} \n'.format(render_type, mean_total_inference_time))

        if eval_ssim:
            mean_ssim = np.mean(ssims)
            data_record.write('{} ssim avg = {} \n'.format(render_type, mean_ssim))
            print('{} ssim avg = {}'.format(render_type, mean_ssim))

        if ms_ssims:
            data_record.write('{} ms_ssims avg = {} \n'.format(render_type, ms_ssims))
            print('{} ms_ssims avg = {}'.format(render_type, ms_ssims))

        if eval_lpips_vgg:
            mean_lpips_vgg = np.mean(lpips_vgg)
            data_record.write('{} lpips_vgg avg = {} \n'.format(render_type, mean_lpips_vgg))
            print('{} lpips_vgg avg = {}'.format(render_type, mean_lpips_vgg))

        if eval_lpips_alex:
            mean_lpips_alex = np.mean(lpips_alex)
            data_record.write('{} lpips_alex avg = {} \n'.format(render_type, mean_lpips_alex))
            print('{} lpips_alex avg = {}'.format(render_type, mean_lpips_alex))

        if mse:
            mean_mse = np.mean(mse)
            data_record.write('{} mse avg = {} \n'.format(render_type, mean_mse))
            print('{} mse avg = {}'.format(render_type, mean_mse))

    return rgbs, disps


def seed_everything():
    '''Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    '''
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def load_everything(args, cfg):
    '''Load images / poses / camera settings / data split.
    '''
    data_dict = load_data(cfg.data)

    if cfg.data.dataset_type == 'hyper_dataset':
        kept_keys = {
            'data_class',
            'near', 'far',
            'i_train', 'i_val', 'i_test',}
        for k in list(data_dict.keys()):
            if k not in kept_keys:
                data_dict.pop(k)

        # pdb.set_trace()
        return data_dict

    # remove useless field
    kept_keys = {
        'hwf', 'HW', 'Ks', 'near', 'far',
        'i_train', 'i_val', 'i_test', 'irregular_shape',
        'poses', 'render_poses', 'images', 'times', 'render_times'}

    for k in list(data_dict.keys()):
        if k not in kept_keys:
            data_dict.pop(k)

    # construct data tensor
    if data_dict['irregular_shape']:
        data_dict['images'] = [torch.FloatTensor(im, device='cpu') for im in data_dict['images']]
    else:
        data_dict['images'] = torch.FloatTensor(data_dict['images'], device='cpu')

    data_dict['poses'] = torch.tensor(data_dict['poses'], device='cpu')
    data_dict['render_times'] = torch.tensor(data_dict['render_times'], device='cpu')
    data_dict['times'] = torch.tensor(data_dict['times'], device='cpu')

    return data_dict


def compute_bbox_by_cam_frustrm(args, cfg, HW, Ks, poses, i_train, near, far, **kwargs):
    print('compute_bbox_by_cam_frustrm: start')
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min

    for (H, W), K, c2w in zip(HW[i_train], Ks[i_train], poses[i_train]):
        rays_o, rays_d, viewdirs = V4D.get_rays_of_a_view(
            H=H, W=W, K=K, c2w=c2w,
            ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
            flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)

        pts_nf = torch.stack([rays_o + viewdirs * near, rays_o + viewdirs * far])
        pts_nf = pts_nf.to('cuda')

        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0, 1, 2)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0, 1, 2)))

    print('compute_bbox_by_cam_frustrm: xyz_min', xyz_min)
    print('compute_bbox_by_cam_frustrm: xyz_max', xyz_max)
    print('compute_bbox_by_cam_frustrm: finish')

    return xyz_min, xyz_max


def compute_bbox_by_cam_frustrm_hyper(args, cfg, data_class):
    print('compute_bbox_by_cam_frustrm: start')
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for i in data_class.i_train:

        rays_o, rays_d, viewdirs, _ = data_class.load_idx(i,not_dic=True)

        # pdb.set_trace()
        if data_class.near == 0:
            pts_nf = torch.stack([rays_o + rays_d * data_class.near, rays_o + rays_d * data_class.far])

        else:
            pts_nf = torch.stack([rays_o+viewdirs*data_class.near, rays_o+viewdirs*data_class.far])


        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0,1,2)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0,1,2)))


    print('compute_bbox_by_cam_frustrm: xyz_min', xyz_min)
    print('compute_bbox_by_cam_frustrm: xyz_max', xyz_max)

    # pdb.set_trace()
    print('compute_bbox_by_cam_frustrm: finish')
    return xyz_min, xyz_max


@torch.no_grad()
def compute_bbox_by_coarse_geo(model_class, model_path, thres, time_concat, cfg):
    print('compute_bbox_by_coarse_geo: start')
    eps_time = time.time()
    model = utils.load_model(model_class, model_path)

    interp = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, model.density.get_dense_grid().shape[2]),
        torch.linspace(0, 1, model.density.get_dense_grid().shape[3]),
        torch.linspace(0, 1, model.density.get_dense_grid().shape[4]),
    ), -1)
    dense_xyz = model.xyz_min * (1 - interp) + model.xyz_max * interp

    time_index = [i / (cfg.coarse_index) for i in range(cfg.coarse_index)]
    time_index.append(1)

    xyz_min_total = torch.zeros(len(time_index), 3)
    xyz_max_total = torch.zeros(len(time_index), 3)

    for i, times in enumerate(time_index):
        ray_t = torch.ones(dense_xyz[:, :, :, 1].shape).unsqueeze(3) * times

        density = model.get_occupancy_alpha_coarse_geo(dense_xyz, ray_t)

        alpha = model.activate_density(density)

        mask = (alpha > thres)
        active_xyz = dense_xyz[mask]

        xyz_min = active_xyz.amin(0)
        xyz_max = active_xyz.amax(0)

        xyz_min_total[i] = xyz_min
        xyz_max_total[i] = xyz_max

        print('compute_bbox_by_coarse_geo: xyz_min', xyz_min)
        print('compute_bbox_by_coarse_geo: xyz_max', xyz_max)

    xyz_min, _ = torch.min(xyz_min_total, dim=0)
    xyz_max, _ = torch.max(xyz_max_total, dim=0)

    eps_time = time.time() - eps_time
    print('compute_bbox_by_coarse_geo: finish (eps time:', eps_time, 'secs)')

    return xyz_min, xyz_max


def scene_rep_reconstruction(args, cfg, cfg_model, cfg_train, xyz_min, xyz_max, xyz_min_coarse, xyz_max_coarse,
                             data_dict, stage, data_record=None, coarse_ckpt_path=None):
    # init
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total_data_device = torch.device(cfg.total_data_device)

    if abs(cfg_model.world_bound_scale - 1) > 1e-9:
        xyz_shift = (xyz_max - xyz_min) * (cfg_model.world_bound_scale - 1) / 2
        xyz_min -= xyz_shift
        xyz_max += xyz_shift

    if cfg.data.dataset_type =='hyper_dataset':
        data_class = data_dict['data_class']
        near = data_class.near
        far = data_class.far
        i_train = data_class.i_train
        i_test = data_class.i_test
    else:
        HW, Ks, near, far, i_train, i_val, i_test, poses, render_poses, images, times, render_times = [
            data_dict[k] for k in [
                'HW', 'Ks', 'near', 'far', 'i_train', 'i_val', 'i_test', 'poses', 'render_poses', 'images', 'times',
                'render_times']]


    # find whether there is existing checkpoint path
    last_ckpt_path = os.path.join(cfg.save_path, f'{stage}_last.tar')
    if args.no_reload:
        reload_ckpt_path = None
    elif args.ft_path:
        reload_ckpt_path = args.ft_path
    elif os.path.isfile(last_ckpt_path):
        reload_ckpt_path = last_ckpt_path
    else:
        reload_ckpt_path = None

    # init model
    model_kwargs = copy.deepcopy(cfg_model)
    num_voxels = model_kwargs.pop('num_voxels')
    if len(cfg_train.pg_scale) and reload_ckpt_path is None:
        num_voxels = int(num_voxels / (2 ** len(cfg_train.pg_scale)))

    if stage == 'coarse':

        model = V4D.v4d(
            xyz_min=xyz_min, xyz_max=xyz_max, num_voxels=num_voxels, rgbnet_width=cfg.rgbnet_width,
            N_iters=cfg_train.N_iters, density_feat_size=cfg.density_feat_size, norm_xyz=True,  **model_kwargs)

    else:
        model = V4D.v4d(
            xyz_min=xyz_min, xyz_max=xyz_max, lut_from=cfg_train.lut_from, viewbase_pe=cfg.viewbase_pe, lut_dim=cfg_train.lut_dim,
            num_voxels=num_voxels, view_direction=cfg.view_direction,
            dual_voxel=cfg.dual_voxel, feature_t_shift=cfg.feature_t_shift,
            norm_xyz=False, rgbnet_depth=cfg.rgbnet_depth, rgbnet_width=cfg.rgbnet_width,
            feature_shift=cfg.feature_shift, N_iters=cfg_train.N_iters,
            density_feat_size=cfg.density_feat_size, lut_field=cfg_train.lut_field, lut_pe=cfg_train.lut_pe,
            lut_iter=cfg_train.lut_iter,
            pertur_surf=cfg_train.pertur_surf,  xyz_min_coarse=xyz_min_coarse, xyz_max_coarse=xyz_max_coarse,
            view_dependent=cfg.view_dependent, **model_kwargs)


    model = model.to(device)

    # init optimizer
    optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)

    # load checkpoint if there is
    if reload_ckpt_path is None:
        print(f'scene_rep_reconstruction ({stage}): train from scratch')
        start = 0
    else:
        print(f'scene_rep_reconstruction ({stage}): reload from {reload_ckpt_path}')
        model, optimizer, start = utils.load_checkpoint(
            model, optimizer, reload_ckpt_path, args.no_reload_optimizer)

    # init rendering setup
    render_kwargs = {
        'near': data_dict['near'],
        'far': data_dict['far'],
        'bg': 1 if cfg.data.white_bkgd else 0,
        'stepsize': cfg_model.stepsize,
        'inverse_y': cfg.data.inverse_y,
        'flip_x': cfg.data.flip_x,
        'flip_y': cfg.data.flip_y,}

    # init batch rays sampler
    def gather_training_rays():
        if data_dict['irregular_shape']:
            rgb_tr_ori = [images[i].to(total_data_device) for i in i_train]
        else:
            rgb_tr_ori = images[i_train].to(total_data_device)

        # pdb.set_trace()

        if cfg_train.ray_sampler == 'in_maskcache':
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = V4D.get_training_rays_in_maskcache_sampling(
                rgb_tr_ori=rgb_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train],
                ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,
                model=model, render_kwargs=render_kwargs)

        elif cfg_train.ray_sampler == 'flatten':
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, times_tr = V4D.get_training_rays_flatten(
                rgb_tr_ori=rgb_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y, times=times[i_train])

        else:
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, times_tr = V4D.get_training_rays(
                rgb_tr=rgb_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y, times=times[i_train])

        index_generator = V4D.batch_indices_generator(len(rgb_tr), cfg_train.N_rand)
        batch_index_sampler = lambda: next(index_generator)

        return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler, times_tr


    def gather_training_rays_hyper():

        now_device = 'cuda' #total_data_device # if cfg.data.load2gpu_on_the_fly else device
        H = data_class.h
        W = data_class.w
        rgb_tr = torch.zeros([len(data_class.i_train), H, W, 3], device=now_device)
        rays_o_tr = torch.zeros([len(data_class.i_train), H, W, 3], device=now_device)
        rays_d_tr = torch.zeros([len(data_class.i_train), H, W, 3], device=now_device)
        viewdirs_tr = torch.zeros([len(data_class.i_train), H, W, 3], device=now_device)
        times_tr = torch.zeros([len(data_class.i_train), H, W, 1], device=now_device)
        imsz = [1] * len(data_class.i_train)

        for i, j in enumerate(data_class.i_train):
            rays_o, rays_d, viewdirs, rgb, times = data_class.load_idx_1(j, not_dic=True, ndc = cfg.data.ndc)
            rgb_tr[i].copy_(rgb.to(now_device))
            rays_o_tr[i].copy_(rays_o.to(now_device))
            rays_d_tr[i].copy_(rays_d.to(now_device))
            viewdirs_tr[i].copy_(viewdirs.to(now_device))

            # pdb.set_trace()
            times_id = torch.ones([H, W, 1], device=now_device) * times.cpu()
            times_tr[i].copy_(times_id.to(now_device))

            del rays_o, rays_d, viewdirs, times_id

        index_generator = V4D.batch_indices_generator(data_class.i_train, cfg_train.N_rand)
        batch_index_sampler = lambda: next(index_generator)

        return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, times_tr, batch_index_sampler

    if cfg.data.dataset_type == 'hyper_dataset':
        rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, times_tr, batch_index_sampler = gather_training_rays_hyper()
    else:
        rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler, times_tr = gather_training_rays()

    # view-count-based learning rate  # coarse train true, fine train false
    if cfg_train.pervoxel_lr:
        def per_voxel_init():
            cnt = model.voxel_count_views(
                rays_o_tr=rays_o_tr, rays_d_tr=rays_d_tr, imsz=imsz, near=near, far=far,
                stepsize=cfg_model.stepsize, downrate=cfg_train.pervoxel_lr_downrate,
                irregular_shape=data_dict['irregular_shape'])

            optimizer.set_pervoxel_lr(cnt)
        per_voxel_init()

    # GOGO
    torch.cuda.empty_cache()
    psnr_lst0 = []
    psnr_lst1 = []
    time0 = time.time()

    for global_step in trange(1 + start, 1 + cfg_train.N_iters):
        torch.cuda.empty_cache()

        # progress scaling checkpoint
        if global_step in cfg_train.pg_scale:
            ori_world_size, new_world_size = model.scale_volume_grid(model.num_voxels * 2)
            data_record.write(
                'v4d: scale_volume_grid scale world_size from {} to {} \n'.format(ori_world_size, new_world_size))
            optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
            # reset the lr after the scale

        # random sample rays
        if cfg_train.ray_sampler in ['flatten', 'in_maskcache']:
            sel_i = batch_index_sampler()
            target = rgb_tr[sel_i]
            rays_o = rays_o_tr[sel_i]
            rays_d = rays_d_tr[sel_i]
            viewdirs = viewdirs_tr[sel_i]
            times_o = times_tr[sel_i]

        elif cfg_train.ray_sampler == 'random':
            sel_b = torch.randint(rgb_tr.shape[0], [cfg_train.N_rand])
            sel_r = torch.randint(rgb_tr.shape[1], [cfg_train.N_rand])
            sel_c = torch.randint(rgb_tr.shape[2], [cfg_train.N_rand])
            target = rgb_tr[sel_b, sel_r, sel_c]
            rays_o = rays_o_tr[sel_b, sel_r, sel_c]
            rays_d = rays_d_tr[sel_b, sel_r, sel_c]
            viewdirs = viewdirs_tr[sel_b, sel_r, sel_c]
            times_o = times_tr[sel_b, sel_r, sel_c]

        else:
            raise NotImplementedError

        if cfg.data.load2gpu_on_the_fly:
            target = target.to(device)

        target = target.to(device)
        rays_o = rays_o.to(device)
        rays_d = rays_d.to(device)
        viewdirs = viewdirs.to(device)
        times_o = times_o.to(device)

        # volume rendering
        render_result = model(rays_o, rays_d, viewdirs, times_o, global_step=global_step, **render_kwargs)

        # gradient descent step
        optimizer.zero_grad(set_to_none=True)

        rgb_loss0 = F.mse_loss(render_result['rgb_marched'][0], target)
        rgb_loss1 = F.mse_loss(render_result['rgb_marched'][-1], target)


        if (global_step > cfg_train.lut_from) and cfg_train.lut_field and cfg.weight_main0 > 0:
            loss = cfg.weight_main0 * rgb_loss0 + cfg.weight_main1 * rgb_loss1

        else:
            loss = cfg.weight_main1 * rgb_loss1

        psnr0 = utils.mse2psnr(rgb_loss0.detach()).item()
        psnr1 = utils.mse2psnr(rgb_loss1.detach()).item()

        density_tv_loss = torch.zeros(1)
        rgb_tv_loss = torch.zeros(1)
        lut_tv_loss = torch.zeros(1)
        surface_normal_loss = torch.zeros(1)


        if cfg_train.weight_entropy_last > 0:
            pout = render_result['alphainv_cum'][..., -1].clamp(1e-6, 1 - 1e-6)
            entropy_last_loss = -(pout * torch.log(pout) + (1 - pout) * torch.log(1 - pout)).mean()
            loss += cfg_train.weight_entropy_last * entropy_last_loss

        if cfg_train.weight_rgbper > 0:
            rgbper = (render_result['raw_rgb'] - target.unsqueeze(-2)).pow(2).sum(-1)
            rgbper_loss = (rgbper * render_result['weights'].detach()).sum(-1).mean()
            loss += cfg_train.weight_rgbper * rgbper_loss

        if stage == 'coarse':

            # pdb.set_trace()
            density_tv_loss = cfg_train.weight_density_tv * model.density_feature_total_variation()
            loss += density_tv_loss
            loss.backward()

        else:

            if cfg.cuda_tv:
                loss.backward()
                dense_tv = 1000000
                if cfg_train.weight_density_tv > 0 and global_step > cfg_train.tv_from and global_step % cfg_train.tv_every == 0:
                    model.density_total_variation_add_grad( cfg_train.weight_density_tv, global_step < dense_tv)

                if cfg_train.weight_rgb_tv > 0 and global_step>cfg_train.tv_from and global_step%cfg_train.tv_every==0 and cfg.dual_voxel:
                    model.k0_total_variation_add_grad(cfg_train.weight_rgb_tv, global_step < dense_tv)
            else:
                if cfg_train.weight_density_tv > 0 and global_step > cfg_train.tv_from and global_step % cfg_train.tv_every == 0:
                    density_tv_loss = cfg_train.weight_density_tv * model.density_feature_total_variation()
                    loss += density_tv_loss

                if cfg_train.weight_rgb_tv > 0 and global_step > cfg_train.tv_from and global_step % cfg_train.tv_every == 0 and cfg.dual_voxel:
                    rgb_tv_loss = cfg_train.weight_rgb_tv * model.rgb_feature_total_variation()
                    loss += rgb_tv_loss

                loss.backward()

        optimizer.step()

        psnr_lst0.append(psnr0)
        psnr_lst1.append(psnr1)


        # pdb.set_trace()

        if stage == 'coarse':
            decay_steps = cfg_train.lrate_decay * 1000

        else:
            decay_steps = cfg_train.N_iters

        decay_factor = 0.1 ** (1 / decay_steps)

        density_decay_factor = cfg.tv_decay_factor ** (1 / decay_steps)


        for i_opt_g, param_group in enumerate(optimizer.param_groups):
            if ('lut' in param_group['name'] or 'LUT' in param_group['name']) and not cfg_train.lut_decay_lr:
                pass
            else:
                param_group['lr'] = param_group['lr'] * decay_factor


        if cfg.decay_density_tv:
            cfg_train.weight_density_tv = cfg_train.weight_density_tv * density_decay_factor

        if cfg.decay_rgb_tv:
            cfg_train.weight_rgb_tv = cfg_train.weight_rgb_tv * density_decay_factor


        # check log & save
        if global_step % args.i_print == 0:
            eps_time = time.time() - time0

            data_record.write('save_path：{} \n'.format(cfg.save_path))
            data_record.write('learning rate：{} \n'.format(param_group['lr']))

            eps_time_str = f'{eps_time // 3600:02.0f}:{eps_time // 60 % 60:02.0f}:{eps_time % 60:02.0f}'
            # flow = render_result['flow'][0]

            lr_temp = param_group['lr']
            tqdm.write(f'scene_rep_reconstruction ({stage}): iter {global_step:6d} / '

                       f'save_path：{cfg.save_path} / '
                       f'learning rate: {lr_temp:.9f} / '
                       f'total_loss: {loss.item():.9f} / '
                       f'rgb_loss0: {rgb_loss0.item():.9f} / PSNR0: {np.mean(psnr_lst0):5.2f} / '
                       f'rgb_loss1: {rgb_loss1.item():.9f} / PSNR1: {np.mean(psnr_lst1):5.2f} / '
                       f'surface_normal_loss Loss: {surface_normal_loss.item():.9f} / '
                       f'density_tv_loss Loss: {density_tv_loss.item():.9f} / '
                       f'rgb_tv_loss Loss: {rgb_tv_loss.item():.9f} / '
                       f'lut_tv_loss Loss: {lut_tv_loss.item():.9f} / '
                       f'Eps: {eps_time_str}')

            data_record.write(f'{eps_time // 3600:02.0f}:{eps_time // 60 % 60:02.0f}:{eps_time % 60:02.0f} \n')
            data_record.write(f'scene_rep_reconstruction ({stage}): iter {global_step:6d} / '

                              f'total_loss: {loss.item():.9f} / '
                              f'rgb_loss0: {rgb_loss0.item():.9f} / PSNR0: {np.mean(psnr_lst0):5.2f} / '
                              f'rgb_loss1: {rgb_loss1.item():.9f} / PSNR1: {np.mean(psnr_lst1):5.2f} / '

                              f'surface_normal_loss Loss: {surface_normal_loss.item():.9f} / '
                              f'density_tv_loss Loss: {density_tv_loss.item():.9f} / '
                              f'rgb_tv_loss Loss: {rgb_tv_loss.item():.9f} / '
                              f'lut_tv_loss Loss: {lut_tv_loss.item():.9f} / '
                              f'Eps: {eps_time_str} \n')


        if global_step % args.i_vis_weight == -1:
            path = os.path.join(cfg.save_path, f'{stage}_{global_step:06d}.tar')

            model.get_voxel_vis(path, global_step)

            print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', path)

    torch.save({
        'global_step': cfg_train.N_iters,
        'model_kwargs': model.get_kwargs(),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, last_ckpt_path)

    print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', last_ckpt_path)

    return data_record


def train(args, cfg, data_dict):
    # init
    print('train: start')
    eps_time = time.time()
    print(os.path.join(cfg.save_path, 'code'))
    os.makedirs(os.path.join(cfg.save_path, 'code'), exist_ok=True)

    # back up files
    source1 = './run.py'
    source6 = './configs'
    source7 = './lib'
    source8 = './utils'

    source = [source1]
    for i in source:
        shutil.copy(i, os.path.join(cfg.save_path, 'code'))

    if not os.path.exists(os.path.join(cfg.save_path, 'code' + '/configs')):
        shutil.copytree(source6, os.path.join(cfg.save_path, 'code' + '/configs'))

    if not os.path.exists(os.path.join(cfg.save_path, 'code' + '/lib')):
        shutil.copytree(source7, os.path.join(cfg.save_path, 'code' + '/lib'))

    if not os.path.exists(os.path.join(cfg.save_path, 'code' + '/lib')):
        shutil.copytree(source8, os.path.join(cfg.save_path, 'code' + '/lib'))


    data_record = open(os.path.join(cfg.save_path, 'args.txt'), 'a')

    for arg in sorted(vars(args)):
        attr = getattr(args, arg)
        data_record.write('{} = {}\n'.format(arg, attr))

    cfg.dump(os.path.join(cfg.save_path, 'config.py'))


    xyz_min_coarse, xyz_max_coarse = 0, 0

    # pdb.set_trace()
    if cfg.coarse_model_and_render.geometry_search:
        # coarse geometry searching
        eps_coarse = time.time()
        xyz_min_coarse, xyz_max_coarse = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, **data_dict)

        data_record = scene_rep_reconstruction(
            args=args, cfg=cfg,
            cfg_model=cfg.coarse_model_and_render, cfg_train=cfg.coarse_train,
            xyz_min=xyz_min_coarse, xyz_max=xyz_max_coarse, xyz_min_coarse=xyz_min_coarse,
            xyz_max_coarse=xyz_max_coarse,
            data_dict=data_dict, stage='coarse', data_record=data_record)

        eps_coarse = time.time() - eps_coarse
        eps_time_str = f'{eps_coarse // 3600:02.0f}:{eps_coarse // 60 % 60:02.0f}:{eps_coarse % 60:02.0f}'
        print('train: coarse geometry searching in', eps_time_str)

        eps_fine = time.time()
        coarse_ckpt_path = os.path.join(cfg.save_path, f'coarse_last.tar')

        xyz_min_fine, xyz_max_fine = compute_bbox_by_coarse_geo(
            model_class=V4D.v4d, model_path=coarse_ckpt_path,
            thres=cfg.fine_model_and_render.bbox_thres, time_concat=cfg.fine_model_and_render.time_concat, cfg=cfg)

        print('xyz_min_coarse, xyz_max_coarse:', xyz_min_coarse, xyz_max_coarse)
        print('xyz_min_fine, xyz_max_fine:', xyz_min_fine, xyz_max_fine)
        data_record.write('\n geometry searching \n: xyz_min_coarse = {}, xyz_max_coarse = {} \n '
                          'xyz_min_fine = {}, xyz_max_fine = {}\n'.format(xyz_min_coarse, xyz_max_coarse, xyz_min_fine,
                                                                          xyz_max_fine))

    elif cfg.data.dataset_type == 'hyper_dataset':
        xyz_min_fine, xyz_max_fine = compute_bbox_by_cam_frustrm_hyper(args=args, cfg=cfg, data_class=data_dict['data_class'])
        eps_fine = time.time()

    else:

        if 'lego' in cfg.save_path:
            xyz_min_fine = np.array([-0.6651, -1.1022, -0.4391])
            xyz_max_fine = np.array([0.5691, 1.0999, 0.7508])

        elif 'mutant' in cfg.save_path:
            xyz_min_fine = np.array([-0.8582, -0.8581, -0.6661])
            xyz_max_fine = np.array([0.8460, 0.1607, 0.8073])

        elif 'bouncingballs' in cfg.save_path:
            xyz_min_fine = np.array([-1.3542, -1.2514, -0.4963])
            xyz_max_fine = np.array([1.2538, 1.3660, 1.0336])

        elif 'hellwarrior' in cfg.save_path:
            xyz_min_fine = np.array([-0.6724, -0.8591, -1.1523])
            xyz_max_fine = np.array([0.6524, 0.8041, 1.0948])


        elif 'standup' in cfg.save_path:
            xyz_min_fine = np.array([-0.5807, -0.5698, -1.2333])
            xyz_max_fine = np.array([0.3268, 0.4537, 1.0893])

        elif 'jumpingjacks' in cfg.save_path:
            xyz_min_fine = np.array([-0.9271, -0.2787, -1.3461])
            xyz_max_fine = np.array([0.9333, 0.3433, 1.0906])

        elif 'hook' in cfg.save_path:
            xyz_min_fine = np.array([-0.4363, -1.0708, -1.0063])
            xyz_max_fine = np.array([0.4688, 0.6876, 1.0903])

        elif 'trex' in cfg.save_path:
            xyz_min_fine = np.array([-0.6203, -1.1333, -0.3828])
            xyz_max_fine = np.array([0.6306, 0.8566, 1.0339])

        elif 'static' in cfg.save_path:
            xyz_min_fine, xyz_max_fine = np.array([-1.2682, -1.2635, -1.0545]), np.array(
                [1.3802, 1.3275, 1.1109])


        eps_fine = time.time()


    data_record = scene_rep_reconstruction(
        args=args, cfg=cfg,
        cfg_model=cfg.fine_model_and_render, cfg_train=cfg.fine_train,
        xyz_min=xyz_min_fine, xyz_max=xyz_max_fine, xyz_min_coarse=xyz_min_coarse, xyz_max_coarse=xyz_max_coarse,
        data_dict=data_dict, stage='fine', data_record=data_record,
        coarse_ckpt_path=None)


    eps_fine = time.time() - eps_fine
    eps_time_str = f'{eps_fine // 3600:02.0f}:{eps_fine // 60 % 60:02.0f}:{eps_fine % 60:02.0f}'
    print('train: fine detail reconstruction in', eps_time_str)
    data_record.write('train: fine detail reconstruction in {} \n'.format(eps_time_str))

    eps_time = time.time() - eps_time
    eps_time_str = f'{eps_time // 3600:02.0f}:{eps_time // 60 % 60:02.0f}:{eps_time % 60:02.0f}'
    print('train: finish (eps time', eps_time_str, ')')
    data_record.write('train: finish! eps time = {} \n'.format(eps_time_str))

    data_record.close()


if __name__ == '__main__':

    # load setup
    parser = config_parser()
    args = parser.parse_args()
    # pdb.set_trace()
    cfg = mmcv.Config.fromfile(args.config)

    cfg.density_feat_size = args.density_feat_size
    cfg.rgbnet_width = args.rgbnet_width
    cfg.rgbnet_depth = args.rgbnet_depth
    cfg.savename = args.savename

    cfg.feature_shift = args.feature_shift
    cfg.feature_t_shift = args.feature_t_shift
    cfg.dual_voxel = args.dual_voxel
    cfg.coarse_model_and_render.geometry_search = args.search_geometry

    cfg.coarse_index = args.coarse_index

    cfg.coarse_train.weight_density_tv = args.weight_density_tv
    cfg.fine_train.weight_density_tv = args.weight_density_tv

    cfg.coarse_train.weight_rgb_tv = args.weight_rgb_tv
    cfg.fine_train.weight_rgb_tv = args.weight_rgb_tv


    cfg.coarse_train.tv_from = args.tv_from
    cfg.fine_train.tv_from = args.tv_from
    cfg.decay_density_tv = args.decay_density_tv
    cfg.decay_rgb_tv = args.decay_rgb_tv

    cfg.cuda_tv = args.cuda_tv
    cfg.coarse_train.tv_every = args.tv_every
    cfg.fine_train.tv_every = args.tv_every
    cfg.view_direction = args.view_direction
    cfg.viewbase_pe = args.viewbase_pe
    cfg.tv_decay_factor = args.tv_decay_factor

    cfg.weight_main0 = args.weight_main0
    cfg.weight_main1 = args.weight_main1


    cfg.coarse_train.lrate_density = args.voxel_lr
    cfg.coarse_train.lrate_k0_1 = args.voxel_lr


    cfg.fine_train.lrate_density = args.voxel_lr
    cfg.fine_train.lrate_k0_1 = args.voxel_lr


    # cfg.coarse_train.N_iters = args.coarse_iters
    # cfg.fine_train.N_iters = args.fine_iters

    cfg.coarse_train.N_rand = args.N_rand
    cfg.fine_train.N_rand = args.N_rand

    cfg.basedir = args.basedir
    cfg.total_data_device = args.total_data_device
    cfg.static_type = args.static_type
    cfg.view_dependent = args.view_dependent


    for i in ['hellwarrior', 'standup', 'hook', 'bouncingballs', 'lego', 'trex', 'jumpingjacks', 'mutant', 'printer', 'broom', 'chicken', 'peel']:
        if i in args.config:
            category = i

    cfg.save_path = os.path.join(cfg.basedir, category)


    # init environment
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    seed_everything()

    data_dict = load_everything(args=args, cfg=cfg)

    # export scene bbox and camera poses in 3d for debugging and visualization
    if args.export_bbox_and_cams_only:
        print('Export bbox and cameras...')
        xyz_min, xyz_max = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, **data_dict)
        poses, HW, Ks, i_train = data_dict['poses'], data_dict['HW'], data_dict['Ks'], data_dict['i_train']
        near, far = data_dict['near'], data_dict['far']
        cam_lst = []
        for c2w, (H, W), K in zip(poses[i_train], HW[i_train], Ks[i_train]):
            rays_o, rays_d, viewdirs = V4D.get_rays_of_a_view(
                H, W, K, c2w, cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y, )
            cam_o = rays_o[0, 0].cpu().numpy()
            cam_d = rays_d[[0, 0, -1, -1], [0, -1, 0, -1]].cpu().numpy()
            cam_lst.append(np.array([cam_o, *(cam_o + cam_d * max(near, far * 0.05))]))
        np.savez_compressed(args.export_bbox_and_cams_only,
                            xyz_min=xyz_min.cpu().numpy(), xyz_max=xyz_max.cpu().numpy(),
                            cam_lst=np.array(cam_lst))
        print('done')
        sys.exit()

    if args.export_coarse_only:
        print('Export coarse visualization...')
        with torch.no_grad():
            ckpt_path = os.path.join(cfg.save_path, 'coarse_last.tar')
            model = utils.load_model(V4D.v4d, ckpt_path).to(device)
            alpha = model.activate_density(model.density).squeeze().cpu().numpy()
            rgb = torch.sigmoid(model.k0).squeeze().permute(1, 2, 3, 0).cpu().numpy()
        np.savez_compressed(args.export_coarse_only, alpha=alpha, rgb=rgb)
        print('done')
        sys.exit()


    # for the visualization in synthesis dataset
    if args.video_only != 'no':
        ckpt_path = args.video_only
        ckpt_name = ckpt_path.split('/')[-1][:-4]

        model = utils.load_model(V4D.v4d, ckpt_path).to(device)
        stepsize = cfg.fine_model_and_render.stepsize
        render_viewpoints_kwargs = {
            'model': model,
            'ndc': cfg.data.ndc,
            'render_kwargs': {
                'near': data_dict['near'],
                'far': data_dict['far'],
                'bg': 1 if cfg.data.white_bkgd else 0,
                'stepsize': stepsize,
                'inverse_y': cfg.data.inverse_y,
                'flip_x': cfg.data.flip_x,
                'flip_y': cfg.data.flip_y,
            },
        }

        # render video
        # pdb.set_trace()
        # view time
        fps = 20
        testsavedir = os.path.join(cfg.save_path, f'render_time_view')
        os.makedirs(testsavedir, exist_ok=True)
        rgbs, disps = render_viewpoints(
            render_poses=data_dict['render_poses'],
            HW=data_dict['HW'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
            Ks=data_dict['Ks'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
            render_factor=args.render_video_factor,
            savedir=testsavedir,
            times=data_dict['render_times'],
            **render_viewpoints_kwargs)

        # data_dict['render_times']
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=fps, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.disp.mp4'), utils.to8b(disps / np.max(disps)), fps=fps,
                         quality=8)


        # fixed time
        time_size = data_dict['render_times'].shape[0]

        for i in range(0, time_size, 10):

            testsavedir = os.path.join(cfg.save_path, 'render_fixed_time', '{}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            rgbs, disps = render_viewpoints(
                render_poses=data_dict['render_poses'],
                HW=data_dict['HW'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                Ks=data_dict['Ks'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                render_factor=args.render_video_factor,
                savedir=testsavedir,
                times=data_dict['render_times'][i].repeat(time_size),
                **render_viewpoints_kwargs)

            # data_dict['render_times']
            imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=fps, quality=8)
            imageio.mimwrite(os.path.join(testsavedir, 'video.disp.mp4'), utils.to8b(disps / np.max(disps)), fps=fps,
                             quality=8)


        # fixed view
        view_size = data_dict['render_poses'].shape[0]
        for i in range(0, view_size, 10):
            testsavedir = os.path.join(cfg.save_path, 'render_fixed_view', '{}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            rgbs, disps = render_viewpoints(
                render_poses=data_dict['render_poses'][i ,...].repeat(view_size,1,1),
                HW=data_dict['HW'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                Ks=data_dict['Ks'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                render_factor=args.render_video_factor,
                savedir=testsavedir,
                times=data_dict['render_times'],
                **render_viewpoints_kwargs)

            # data_dict['render_times']
            imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=fps, quality=8)
            imageio.mimwrite(os.path.join(testsavedir, 'video.disp.mp4'), utils.to8b(disps / np.max(disps)), fps=fps,
                             quality=8)

        exit()


    # train
    if not args.render_only:
        # pass
        train(args, cfg, data_dict)


    # load model for rendring
    if args.render_test or args.render_train or args.render_video:
        if args.ft_path:
            ckpt_path = args.ft_path
        else:
            ckpt_path = os.path.join(cfg.save_path, 'fine_last.tar')

        ckpt_name = ckpt_path.split('/')[-1][:-4]

        model = utils.load_model(V4D.v4d, ckpt_path).to(device)
        stepsize = cfg.fine_model_and_render.stepsize
        render_viewpoints_kwargs = {
            'model': model,
            'ndc': cfg.data.ndc,
            'render_kwargs': {
                'near': data_dict['near'],
                'far': data_dict['far'],
                'bg': 1 if cfg.data.white_bkgd else 0,
                'stepsize': stepsize,
                'inverse_y': cfg.data.inverse_y,
                'flip_x': cfg.data.flip_x,
                'flip_y': cfg.data.flip_y,
            },
        }

    # render trainset and eval
    if args.render_train:
        testsavedir = os.path.join(cfg.save_path, f'render_train_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)

        data_record = open(os.path.join(cfg.save_path, 'args.txt'), 'a')
        rgbs, disps = render_viewpoints(
            render_poses=data_dict['poses'][data_dict['i_train']],
            HW=data_dict['HW'][data_dict['i_train']],
            Ks=data_dict['Ks'][data_dict['i_train']],
            gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_train']],
            savedir=testsavedir,
            eval_ssim=cfg.data.eval_ssim, eval_lpips_alex=cfg.data.eval_lpips_alex,
            eval_lpips_vgg=cfg.data.eval_lpips_vgg,
            eval_mse=cfg.data.eval_mse,
            times=data_dict['times'][data_dict['i_train']],
            data_record=data_record,
            render_type='train',
            **render_viewpoints_kwargs)

        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=3, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.disp.mp4'), utils.to8b(disps / np.max(disps)), fps=3,
                         quality=8)

    # render testset and eval
    if args.render_test:
        testsavedir = os.path.join(cfg.save_path, f'render_test_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)

        data_record = open(os.path.join(cfg.save_path, 'args.txt'), 'a')

        if cfg.data.dataset_type != 'hyper_dataset':
            rgbs, disps = render_viewpoints(
                render_poses=data_dict['poses'][data_dict['i_test']],

                HW=data_dict['HW'][data_dict['i_test']],
                Ks=data_dict['Ks'][data_dict['i_test']],
                gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_test']],
                savedir=testsavedir,
                eval_ssim=cfg.data.eval_ssim, eval_lpips_alex=cfg.data.eval_lpips_alex,
                eval_lpips_vgg=cfg.data.eval_lpips_vgg,
                eval_mse=cfg.data.eval_mse,
                times=data_dict['times'][data_dict['i_test']],
                data_record=data_record,
                render_type='test',

                **render_viewpoints_kwargs)

        else:
            rgbs, disps = render_viewpoints_hyper(
                data_class = data_dict,
                savedir=testsavedir,
                eval_ssim=cfg.data.eval_ssim, eval_lpips_alex=cfg.data.eval_lpips_alex,
                eval_lpips_vgg=cfg.data.eval_lpips_vgg,
                eval_mse=cfg.data.eval_mse,
                data_record=data_record,
                render_type='test', **render_viewpoints_kwargs)

        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=3, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.disp.mp4'), utils.to8b(disps / np.max(disps)), fps=3, quality=8)


    # render video
    if args.render_video:
        testsavedir = os.path.join(cfg.save_path, f'render_video_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        rgbs, disps = render_viewpoints(
            render_poses=data_dict['render_poses'],
            HW=data_dict['HW'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
            Ks=data_dict['Ks'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
            render_factor=args.render_video_factor,
            savedir=testsavedir,
            times=data_dict['times'][data_dict['i_test']],
            **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=3, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.disp.mp4'), utils.to8b(disps / np.max(disps)), fps=3,
                         quality=8)

    print('Done')
