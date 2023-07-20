# -*- encoding: utf-8 -*-
import time
import pdb, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import Generator3DLUT_zero, Generator3DLUT_identity, rgb_net
import numpy as np
from . import grid

'''Model'''
class v4d(torch.nn.Module):

    def __init__(self, xyz_min, xyz_max, xyz_min_coarse=0, xyz_max_coarse=0,
                 num_voxels=0, num_voxels_base=0,  lut_from=0, lut_field=False, view_dependent='pre',
                 alpha_init=None, use_cuda=False, view_direction=True,
                 nearest=False, N_iters=40000, density_feat_size=12, norm_xyz=False,
                 time_concat=True,  feature_shift=False, dual_voxel=False, feature_t_shift=False,
                 fast_color_thres=0, lut_dim=33,
                 rgbnet_dim=0, rgbnet_direct=False, rgbnet_full_implicit=False,
                 rgbnet_depth=3, rgbnet_width=128,
                 posbase_pe=5, viewbase_pe=4, lut_pe=5,
                 lut_iter=3,   pertur_surf=0.0,
                 **kwargs):


        super(v4d, self).__init__()
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        self.register_buffer('xyz_min_coarse', torch.Tensor(xyz_min_coarse))
        self.register_buffer('xyz_max_coarse', torch.Tensor(xyz_max_coarse))

        self.fast_color_thres = fast_color_thres
        self.nearest = nearest

        self.N_iters = N_iters

        self.norm_xyz = norm_xyz
        self.rgbnet_width = rgbnet_width
        self.rgbnet_depth = rgbnet_depth
        self.use_cuda = use_cuda
        self.view_direction = view_direction
        self.viewbase_pe = viewbase_pe


        self.lut_from = lut_from
        self.lut_field = lut_field
        self.pertur_surf = pertur_surf
        self.lut_dim = lut_dim
        self.lut_pe = lut_pe
        self.lut_iter = lut_iter

        self.view_dependent = view_dependent

        rgb_out = 3

        # determine based grid resolution
        self.num_voxels_base = num_voxels_base
        self.voxel_size_base = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels_base).pow(1 / 3)


        # determine the density bias shift
        self.alpha_init = alpha_init
        self.act_shift = np.log(1 / (1 - alpha_init) - 1)
        print('dvgo: set density bias shift to', self.act_shift)  # 1e-5 -> -11.512915464924033

        # determine init grid resolution
        self._set_grid_resolution(num_voxels)


        # define occupancy net
        self.density_feat_size = density_feat_size

        density_type = 'DenseGrid'
        self.density_type = density_type
        self.density_config = dict()
        self.density = grid.create_grid(
            density_type, channels=self.density_feat_size, world_size=self.world_size,
            xyz_min=self.xyz_min, xyz_max=self.xyz_max,
            config=self.density_config)

        self.time_concat = time_concat
        self.feature_shift = feature_shift
        self.feature_t_shift = feature_t_shift
        self.dual_voxel = dual_voxel

        if self.time_concat in ['concat_t_phase', 'concat_t']:

            occupancy_input_dim = self.density_feat_size + (4 + 4 * posbase_pe * 2)
            # self.density_feat_size (12, 24) + xyzt (4+4*posbase_pe*2)

            if self.dual_voxel:
                feature_size = self.density_feat_size * 2
            else:
                feature_size = self.density_feat_size

            if self.feature_shift:
                rgb_input = ((1 + feature_size) + (1 + feature_size) * posbase_pe * 2)  # t + feature

            else:
                rgb_input = (1 + 1 * posbase_pe * 2) + feature_size  # t + feature

        else:
            print('error implementation!')
            exit()

        occupancy_net_width = rgbnet_width

        occupancy_output_dim = 1

        self.occupancy_net = nn.Sequential(
            nn.Linear(occupancy_input_dim, occupancy_net_width), nn.ReLU(inplace=True),
            *[nn.Sequential(nn.Linear(occupancy_net_width, occupancy_net_width), nn.ReLU(inplace=True))
              for _ in range(self.rgbnet_depth - 2)], nn.Linear(occupancy_net_width, occupancy_output_dim), )

        nn.init.constant_(self.occupancy_net[-1].bias, 0)

        if self.lut_field:

            # if self.lut_pe == 10:
            self.register_buffer('lut_posfreq', torch.FloatTensor([(2 ** i) for i in range(lut_pe)]))
            lut_in_channels = 4 + 4 * 2 * lut_pe  # + 3 + 3*2*4


            lut_out_channels = 5
            lut_net_width = 256

            lut_net_depth = self.lut_pe

            self.lut_net = nn.Sequential(
                nn.Linear(lut_in_channels, lut_net_width), nn.ReLU(inplace=True),
                *[nn.Sequential(nn.Linear(lut_net_width, lut_net_width), nn.ReLU(inplace=True))
                  for _ in range(lut_net_depth - 2)], nn.Linear(lut_net_width, lut_out_channels), )
            nn.init.constant_(self.lut_net[-1].bias, 1)

            # 33 or 64
            self.LUT0 = Generator3DLUT_identity(dim=self.lut_dim).cuda()
            self.LUT1 = Generator3DLUT_zero(dim=self.lut_dim).cuda()
            self.LUT2 = Generator3DLUT_zero(dim=self.lut_dim).cuda()
            self.LUT3 = Generator3DLUT_zero(dim=self.lut_dim).cuda()
            self.LUT4 = Generator3DLUT_zero(dim=self.lut_dim).cuda()



        # init color representation
        self.rgbnet_kwargs = {
            'rgbnet_dim': rgbnet_dim, 'rgbnet_direct': rgbnet_direct,
            'rgbnet_full_implicit': rgbnet_full_implicit,
            'rgbnet_depth': rgbnet_depth, 'rgbnet_width': rgbnet_width,
            'posbase_pe': posbase_pe, 'viewbase_pe': viewbase_pe,
        }
        self.rgbnet_full_implicit = rgbnet_full_implicit

        if rgbnet_dim <= 0:

            # color voxel grid  (coarse stage)
            self.k0_dim = 3
            # self.k0 = torch.nn.Parameter(torch.zeros([1, self.k0_dim, *self.world_size]))

            k0_type = 'DenseGrid'
            self.k0_type = k0_type
            self.k0_config = dict()

            self.k0_1 = grid.create_grid(
                k0_type, channels=self.k0_dim, world_size=self.world_size,
                xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                config=self.k0_config)

            self.rgbnet = None
            self.register_buffer('posfreq', torch.FloatTensor([(2 ** i) for i in range(posbase_pe)]))
            self.register_buffer('viewfreq', torch.FloatTensor([(2 ** i) for i in range(viewbase_pe)]))


        else:
            # feature voxel grid + shallow MLP  (fine stage)
            if self.rgbnet_full_implicit:
                self.k0_dim = 0
            else:
                self.k0_dim = density_feat_size

            k0_type = 'DenseGrid'
            self.k0_type = k0_type
            self.k0_config = dict()

            self.k0_1 = grid.create_grid(
                k0_type, channels=self.k0_dim, world_size=self.world_size,
                xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                config=self.k0_config)


            self.rgbnet_direct = rgbnet_direct
            self.register_buffer('posfreq', torch.FloatTensor([(2 ** i) for i in range(posbase_pe)]))
            self.register_buffer('viewfreq', torch.FloatTensor([(2 ** i) for i in range(viewbase_pe)]))

            if self.view_direction:
                dim0 = (3 + 3 * posbase_pe * 2) + (3 + 3 * viewbase_pe * 2)  # xyzt+ view

            else:
                dim0 = (3 + 3 * posbase_pe * 2)  # xyzt+ view


            self.rgbnet = rgb_net(rgb_input=dim0 + rgb_input, rgbnet_width=rgbnet_width, rgb_out=rgb_out,
                                  view_dependent=self.view_dependent, input_ch_views=3 + 3 * viewbase_pe * 2,
                                  rgbnet_depth=self.rgbnet_depth)
            print('rgbmlp:', self.rgbnet)

        self.mask_cache = None
        self.nonempty_mask = None

    def _set_grid_resolution(self, num_voxels):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1 / 3)  # 107,107,88

        world_size_tmp = ((self.xyz_max - self.xyz_min) / self.voxel_size).long()

        self.world_size = world_size_tmp  # + (4 - world_size_tmp % 4)  # 使得做shift operation 的时候，能被4整除

        self.voxel_size_ratio = self.voxel_size / self.voxel_size_base
        print('v4d: voxel_size      ', self.voxel_size)
        print('v4d: world_size      ', self.world_size)
        print('v4d: voxel_size_base ', self.voxel_size_base)
        print('v4d: voxel_size_ratio', self.voxel_size_ratio)

    def get_kwargs(self):
        return {
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'xyz_min_coarse': self.xyz_min_coarse.cpu().numpy(),
            'xyz_max_coarse': self.xyz_max_coarse.cpu().numpy(),
            'num_voxels': self.num_voxels,
            'num_voxels_base': self.num_voxels_base,
            'alpha_init': self.alpha_init,
            'nearest': self.nearest,
            'fast_color_thres': self.fast_color_thres,
            'time_concat': self.time_concat,
            'density_feat_size': self.density_feat_size,
            'rgbnet_width': self.rgbnet_width,
            'rgbnet_depth': self.rgbnet_depth,
            'norm_xyz': self.norm_xyz,

            'feature_shift': self.feature_shift,
            'feature_t_shift': self.feature_t_shift,
            'dual_voxel': self.dual_voxel,
            'use_cuda': self.use_cuda,
            'view_direction': self.view_direction,
            'lut_field': self.lut_field,
            'lut_from': self.lut_from,
            'viewbase_pe': self.viewbase_pe,
            'lut_dim': self.lut_dim,
            'lut_pe': self.lut_pe,
            'lut_iter': self.lut_iter,
            'pertur_surf': self.pertur_surf,
            'view_dependent': self.view_dependent,
            **self.rgbnet_kwargs,
        }

    def voxel_count_views(self, rays_o_tr, rays_d_tr, imsz, near, far, stepsize, downrate=1, irregular_shape=False):
        print('voxel_count_views start')
        eps_time = time.time() # self.density_1.get_dense_grid().shape[2:]
        N_samples = int(np.linalg.norm(np.array(self.density.get_dense_grid().shape[2:]) + 1) / stepsize) + 1
        rng = torch.arange(N_samples)[None].float()
        count = torch.zeros_like(self.density.get_dense_grid().detach())
        device = rng.device
        for rays_o_, rays_d_ in zip(rays_o_tr.split(imsz), rays_d_tr.split(imsz)):
            ones = torch.ones_like(self.density.get_dense_grid()).requires_grad_()
            if irregular_shape:
                rays_o_ = rays_o_.split(10000)
                rays_d_ = rays_d_.split(10000)
            else:
                rays_o_ = rays_o_[::downrate, ::downrate].to(device).flatten(0, -2).split(10000)
                rays_d_ = rays_d_[::downrate, ::downrate].to(device).flatten(0, -2).split(10000)

            for rays_o, rays_d in zip(rays_o_, rays_d_):
                vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
                rate_a = (self.xyz_max - rays_o) / vec
                rate_b = (self.xyz_min - rays_o) / vec
                t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)
                t_max = torch.maximum(rate_a, rate_b).amin(-1).clamp(min=near, max=far)
                step = stepsize * self.voxel_size * rng
                interpx = (t_min[..., None] + step / rays_d.norm(dim=-1, keepdim=True))
                rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
                self.grid_sampler(rays_pts, ones).sum().backward()
            with torch.no_grad():
                count += (ones.grad > 1)
        eps_time = time.time() - eps_time
        print('voxel_count_views finish (eps time:', eps_time, 'sec)')
        return count


    @torch.no_grad()
    def scale_volume_grid(self, num_voxels):
        print('dvgo: scale_volume_grid start')
        ori_world_size = self.world_size
        self._set_grid_resolution(num_voxels)
        print('dvgo: scale_volume_grid scale world_size from', ori_world_size, 'to', self.world_size)
        self.density.scale_volume_grid(self.world_size)
        self.k0_1.scale_volume_grid(self.world_size)

        return ori_world_size, self.world_size


    def density_feature_total_variation(self):
        return total_variation(self.density.get_dense_grid(), None)  # 注意这个mask

    def rgb_feature_total_variation(self):
        return total_variation(self.k0_1.get_dense_grid(), None)  # 注意这个mask


    def density_total_variation_add_grad(self, weight, dense_mode):
        w = weight # * self.world_size.max() / 128
        self.density.total_variation_add_grad(w, w, w, dense_mode)


    def k0_total_variation_add_grad(self, weight, dense_mode):
        w = weight # * self.world_size.max() / 128
        self.k0_1.total_variation_add_grad(w, w, w, dense_mode)


    def activate_density(self, density, interval=None):

        interval = interval if interval is not None else self.voxel_size_ratio
        return 1 - torch.exp(-F.softplus(density + self.act_shift) * interval)


    def sample_ray(self, rays_o, rays_d, rays_t, near, far, stepsize, is_train=False, **render_kwargs):
        '''Sample query points on rays'''
        N_samples = int(
            np.linalg.norm(
                np.array(self.density.get_dense_grid().shape[2:]) + 1) / stepsize) + 1

        # 2. determine the two end-points of ray bbox intersection
        vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)

        rate_a = (self.xyz_max - rays_o) / vec
        rate_b = (self.xyz_min - rays_o) / vec

        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)
        t_max = torch.maximum(rate_a, rate_b).amin(-1).clamp(min=near, max=far)

        # 3. check whether a raw intersect the bbox or not
        mask_outbbox = (t_max <= t_min)

        # 4. sample points on each ray
        rng = torch.arange(N_samples)[None].float()
        if is_train == 'train':
            rng = rng.repeat(rays_d.shape[-2], 1)
            rng += torch.rand_like(rng[:, [0]])  # add some noise to the sample

        step = stepsize * self.voxel_size * rng
        interpx = (t_min[..., None] + step / rays_d.norm(dim=-1, keepdim=True))
        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        # rays_t = rays_t.unsqueeze(1).repeat(1, N_samples)


        if len(rays_t.shape) > 2:
            rays_t = rays_t.repeat(1, 1, N_samples).unsqueeze(3)
        else:
            rays_t = rays_t.repeat(1, N_samples).unsqueeze(2)

        # 5. update mask for query points outside bbox
        mask_outbbox = mask_outbbox[..., None] | ((self.xyz_min > rays_pts) | (rays_pts > self.xyz_max)).any(dim=-1)

        return rays_pts, mask_outbbox, rays_t #, ray_id, step_id

    def grid_sampler(self, xyz, *grids, mode=None, align_corners=True):

        '''Wrapper for the interp operation'''
        if mode is None:
            # bilinear is actually trilinear if 5D input is given to grid_sample
            mode = 'nearest' if self.nearest else 'bilinear'

        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1, 1, 1, -1, 3)

        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1

        ret_lst = [
            F.grid_sample(grid, ind_norm, mode=mode, align_corners=align_corners).reshape(grid.shape[1], -1).T.reshape(
                *shape, grid.shape[1]).squeeze()
            for grid in grids]  # len(grids) = 1

        if len(ret_lst) == 1:
            return ret_lst[0]
        return ret_lst

    def lut_grid_sampler(self, xyz, *grids, mode=None, align_corners=True):
        '''Wrapper for the interp operation'''
        if mode is None:
            # bilinear is actually trilinear if 5D input is given to grid_sample
            mode = 'nearest' if self.nearest else 'bilinear'
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1, 1, 1, -1, 3)

        grids = [grid.unsqueeze(0) for grid in grids]

        ind_norm = xyz * 2 - 1

        ret_lst = [
            F.grid_sample(grid, ind_norm, mode=mode, align_corners=align_corners).reshape(grid.shape[1], -1).T.reshape(
                *shape, grid.shape[1]).squeeze() for grid in grids]

        if len(ret_lst) == 1:
            return ret_lst[0]
        return ret_lst

    def get_occupancy_alpha(self, rays_pts, rays_t):

        density_feat_0 = self.density(rays_pts)

        if len(density_feat_0.shape) == 1:
            density_feat_0 = density_feat_0.unsqueeze(0)

        if self.norm_xyz:
            rays_xyz = (rays_pts - self.xyz_min) / (self.xyz_max - self.xyz_min)
        else:
            rays_xyz = rays_pts

        xyz_emb = (rays_xyz.unsqueeze(-1) * self.posfreq.cuda()).flatten(-2)
        rays_t_emb = (rays_t.unsqueeze(-1) * self.posfreq.cuda()).flatten(-2)
        xyz_emb = torch.cat(
            [density_feat_0, rays_xyz, xyz_emb.sin(), xyz_emb.cos(), rays_t, rays_t_emb.sin(), rays_t_emb.cos()], -1)
        # [voxel feature, xyz, t] = 12 + 3 + 15 + 15 + 1 + 5 + 5 = 56

        alpha1 = self.occupancy_net(xyz_emb).squeeze()

        return alpha1

    def get_occupancy_alpha_coarse_geo(self, rays_pts, rays_t):

        density_feat_0 = self.density(rays_pts)

        if len(density_feat_0.shape) == 1:
            density_feat_0 = density_feat_0.unsqueeze(0)

        if self.norm_xyz:
            rays_xyz = (rays_pts - self.xyz_min) / (self.xyz_max - self.xyz_min)
        else:
            rays_xyz = rays_pts


        xyz_emb = (rays_xyz.unsqueeze(-1) * self.posfreq.cuda()).flatten(-2)
        rays_t_emb = (rays_t.unsqueeze(-1) * self.posfreq.cuda()).flatten(-2)
        xyz_emb = torch.cat(
            [density_feat_0, rays_xyz, xyz_emb.sin(), xyz_emb.cos(), rays_t, rays_t_emb.sin(), rays_t_emb.cos()], -1)
        # [voxel feature, xyz, t] = 12 + 3 + 15 + 15 + 1 + 5 + 5 = 56

        alpha1 = self.occupancy_net(xyz_emb).squeeze()

        return alpha1

    def get_rgb_feat(self, viewdirs, rays_pts, rays_t, mask, k0_view, density_feat_by_alpha, weights):


        viewdirs = viewdirs.unsqueeze(-2)

        if len(viewdirs.shape) == 4:
            viewdirs = viewdirs.repeat(1, 1, weights.shape[-1], 1)
        else:
            viewdirs = viewdirs.repeat(1, weights.shape[-1], 1)

        viewdirs = viewdirs[mask].unsqueeze(-1)

        viewdirs_mask = (viewdirs).flatten(-2)
        viewdirs_emb = (viewdirs * self.viewfreq).flatten(-2)
        viewdirs_emb = torch.cat([viewdirs_mask, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)

        if self.norm_xyz:
            rays_xyz = (rays_pts - self.xyz_min) / (self.xyz_max - self.xyz_min)
        else:
            rays_xyz = rays_pts

        rays_t = rays_t[mask]

        xyz_emb = (rays_xyz.unsqueeze(-1) * self.posfreq).flatten(-2)
        xyz_t_emb = (rays_t.unsqueeze(-1) * self.posfreq).flatten(-2)
        xyz_t_emb = torch.cat([rays_xyz, xyz_emb.sin(), xyz_emb.cos(), rays_t, xyz_t_emb.sin(), xyz_t_emb.cos()], -1)

        if self.dual_voxel:
            feature_to_shift = torch.cat([k0_view, density_feat_by_alpha], 1)

        else:
            feature_to_shift = density_feat_by_alpha

        if self.feature_shift:
            feature_phase = (rays_t.unsqueeze(1).repeat(1, feature_to_shift.shape[-1], 1) * (2 * math.pi) / self.posfreq.cuda()).flatten(-2)
            feature_to_shift_emb = (feature_to_shift.unsqueeze(-1) * self.posfreq.cuda()).flatten(-2)

            if self.feature_t_shift:
                feature_to_shift_emb += feature_phase

            voxel_embedding = torch.cat([feature_to_shift, feature_to_shift_emb.sin(), feature_to_shift_emb.cos()], 1)

            return torch.cat([
                voxel_embedding,
                xyz_t_emb,
                viewdirs_emb,  # 27
            ], -1), viewdirs_emb

        else:
            return torch.cat([
                feature_to_shift,
                xyz_t_emb,
                viewdirs_emb,  # 27
            ], -1), viewdirs_emb

    def sample_lut_ray(self, rays_o, rays_d, rays_t, viewdirs, rays_depth, mode):

        rays_pts_0 = rays_o + rays_d * rays_depth.unsqueeze(-1)
        if self.pertur_surf > 0 and mode == 'train':
            rays_pts_0 = rays_pts_0 + (torch.rand_like(rays_pts_0) - 0.5) * self.pertur_surf
        else:
            rays_pts_0 = rays_pts_0

        lut_mask = 0

        xyz_emb = (rays_pts_0.unsqueeze(-1) * self.lut_posfreq).flatten(-2)
        rays_t_emb = (rays_t.unsqueeze(-1) * self.lut_posfreq).flatten(-2)
        lut_emb = torch.cat([rays_pts_0, xyz_emb.sin(), xyz_emb.cos(), rays_t, rays_t_emb.sin(), rays_t_emb.cos()], -1)

        return lut_emb, lut_mask

    def lut_for_rgb(self, img, lut_weight, mode='train'):

        gen_A0 = self.lut_grid_sampler(img, self.LUT0.LUT)
        gen_A1 = self.lut_grid_sampler(img, self.LUT1.LUT)
        gen_A2 = self.lut_grid_sampler(img, self.LUT2.LUT)
        gen_A3 = self.lut_grid_sampler(img, self.LUT3.LUT)
        gen_A4 = self.lut_grid_sampler(img, self.LUT4.LUT)

        combine_rgb = lut_weight[..., 0].unsqueeze(-1) * gen_A0 + lut_weight[..., 1].unsqueeze(-1) * gen_A1 + \
                      lut_weight[..., 2].unsqueeze(-1) * gen_A2 + \
                      lut_weight[..., 3].unsqueeze(-1) * gen_A3 + lut_weight[..., 4].unsqueeze(-1) * gen_A4

        return combine_rgb.clamp(0, 1)


    def forward(self, rays_o, rays_d, viewdirs, times_o, global_step=0, mode='train', **render_kwargs):
        '''Volume rendering'''

        ret_dict = {}
        # sample points on rays
        rays_pts, mask_outbbox, rays_t = self.sample_ray(rays_o=rays_o, rays_d=rays_d, rays_t=times_o, is_train=mode,
                                                         **render_kwargs)

        interval = render_kwargs['stepsize'] * self.voxel_size_ratio

        # query for alpha
        alpha = torch.zeros_like(rays_pts[..., 0])
        density_set = [torch.zeros(4)]
        flow = [torch.zeros(4)]
        density = self.get_occupancy_alpha(rays_pts[~mask_outbbox], rays_t[~mask_outbbox])
        alpha[~mask_outbbox] = self.activate_density(density, interval)

        # compute accumulated transmittance
        weights, alphainv_cum = get_ray_marching_ray(alpha)

        # query for color
        mask = (weights > self.fast_color_thres)  # # coarse: 0, fine: 1e-4

        if not self.rgbnet_full_implicit:

            k0 = self.k0_1(rays_pts[mask])

            # get density feature by  alpha masks
            density_feat_by_alpha = self.density(rays_pts[mask])

            if len(density_feat_by_alpha.shape) == 1:
                density_feat_by_alpha = density_feat_by_alpha.unsqueeze(0)

        rgb_marched = []
        if self.rgbnet is None:
            # no view-depend effect
            rgb = torch.zeros(*weights.shape, self.k0_dim).to(weights)
            rgb[mask] = torch.sigmoid(k0)
            # Ray marching
            rgb_marched_0 = (weights[..., None] * rgb).sum(-2) + alphainv_cum[..., [-1]] * render_kwargs['bg']
            rgb_marched.append(rgb_marched_0.clamp(0, 1))

            depth = 0
            disp = 0
            weights_norm = 0
            surface_normal_loss = torch.zeros(1)
            surface_normal = 0

        else:
            # view-dependent color emission self.rgbnet_direct true for both coarse and fine
            rgb_feat, viewdirs_emb = self.get_rgb_feat(viewdirs, rays_pts[mask], rays_t, mask, k0, density_feat_by_alpha,
                                                       weights)

            rgb_logit = torch.zeros(*weights.shape, 3).to(weights)


            rgb_logit[mask] = self.rgbnet(rgb_feat)
            rgb = torch.sigmoid(rgb_logit)

            # Ray marching
            rgb_marched_0 = (weights[..., None] * rgb).sum(-2) + alphainv_cum[..., [-1]] * render_kwargs['bg']
            rgb_marched.append(rgb_marched_0.clamp(0, 1))

            # set LUT to the rgb_marched
            depth = (rays_o[..., None, :] - rays_pts).norm(dim=-1)
            depth = (weights * depth).sum(-1) + alphainv_cum[..., -1] * render_kwargs['far']
            disp = 1 / depth

            surface_normal_loss = torch.zeros(1)
            surface_normal = 0


            weights_norm = 0
            # ray level
            if self.lut_field:
                if global_step > self.lut_from or mode == 'test':
                    # offering the local information
                    lut_emb, lut_mask = self.sample_lut_ray(rays_o=rays_o, rays_d=rays_d, rays_t=times_o, viewdirs=viewdirs, rays_depth=depth.detach(), mode=mode)
                    lut_weight = self.lut_net(lut_emb)
                    for i in range(self.lut_iter):
                        rgb_marched.append(self.lut_for_rgb(rgb_marched[-1], lut_weight, mode))


        ret_dict.update({
            'alphainv_cum': alphainv_cum,
            'weights': weights,
            'rgb_marched': rgb_marched,
            'raw_alpha': alpha,
            'raw_rgb': rgb,
            'depth': depth,
            'disp': [disp],
            'mask': mask,
            'flow': flow,
            'density_set': density_set,
            'weights_norm': weights_norm,
            'surface_normal': [surface_normal],
            'surface_normal_loss': surface_normal_loss,
        })
        return ret_dict


''' Module for the searched coarse geometry
It supports query for the known free space and unknown space.
'''


''' Misc
'''


def cumprod_exclusive(p):
    return torch.cat([torch.ones_like(p[..., [0]]), p.clamp_min(1e-10).cumprod(-1)], -1)


def get_ray_marching_ray(alpha):
    alphainv_cum = cumprod_exclusive(1 - alpha)

    weights = alpha * alphainv_cum[..., :-1]

    return weights, alphainv_cum


def total_variation(v, mask=None):

    tv2 = v.diff(dim=2).abs()
    tv3 = v.diff(dim=3).abs()
    tv4 = v.diff(dim=4).abs()

    return (tv2.mean() + tv3.mean() + tv4.mean()) / 3


''' Ray and batch
'''


def get_rays(H, W, K, c2w, inverse_y, flip_x, flip_y, mode='center'):
    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W, device=c2w.device),
        torch.linspace(0, H - 1, H, device=c2w.device))  # pytorch's meshgrid has indexing='ij'
    i = i.t().float()
    j = j.t().float()
    if mode == 'lefttop':
        pass
    elif mode == 'center':
        i, j = i + 0.5, j + 0.5
    elif mode == 'random':
        i = i + torch.rand_like(i)
        j = j + torch.rand_like(j)
    else:
        raise NotImplementedError

    if flip_x:
        i = i.flip((1,))
    if flip_y:
        j = j.flip((0,))
    if inverse_y:
        dirs = torch.stack([(i - K[0][2]) / K[0][0], (j - K[1][2]) / K[1][1], torch.ones_like(i)], -1)
    else:
        dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1)

    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3],
                       -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]

    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, 3].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3],
                    -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3, 3], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


def get_rays_of_a_view(H, W, K, c2w, ndc, inverse_y, flip_x, flip_y, mode='center'):
    rays_o, rays_d = get_rays(H, W, K, c2w, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y, mode=mode)
    viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
    if ndc:
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)
    return rays_o, rays_d, viewdirs


@torch.no_grad()
def get_training_rays(rgb_tr, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y, times):
    print('get_training_rays: start')
    assert len(np.unique(HW, axis=0)) == 1
    assert len(np.unique(Ks.reshape(len(Ks), -1), axis=0)) == 1
    assert len(rgb_tr) == len(train_poses) and len(rgb_tr) == len(Ks) and len(rgb_tr) == len(HW)
    H, W = HW[0]
    K = Ks[0]
    DEVICE = rgb_tr[0].device
    eps_time = time.time()
    # rays_o_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    # rays_d_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    # viewdirs_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    # times_tr = torch.zeros([len(rgb_tr), H, W, 1], device=rgb_tr.device)

    rays_o_tr = torch.zeros([len(rgb_tr), H, W, 3], device=DEVICE)
    rays_d_tr = torch.zeros([len(rgb_tr), H, W, 3], device=DEVICE)
    viewdirs_tr = torch.zeros([len(rgb_tr), H, W, 3], device=DEVICE)
    times_tr = torch.zeros([len(rgb_tr), H, W, 1], device=DEVICE)

    imsz = [1] * len(rgb_tr)
    for i, c2w in enumerate(train_poses):
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
            H=H, W=W, K=K, c2w=c2w, ndc=ndc, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)

        # rays_o_tr[i].copy_(rays_o.to(rgb_tr.device))
        # rays_d_tr[i].copy_(rays_d.to(rgb_tr.device))
        # viewdirs_tr[i].copy_(viewdirs.to(rgb_tr.device))
        # times_id =  torch.ones([H, W, 1], device=rgb_tr.device)*times[i]
        # times_tr[i].copy_(times_id.to(rgb_tr.device))

        rays_o_tr[i].copy_(rays_o.to(DEVICE))
        rays_d_tr[i].copy_(rays_d.to(DEVICE))
        viewdirs_tr[i].copy_(viewdirs.to(DEVICE))
        times_id = torch.ones([H, W, 1], device=DEVICE) * times[i]
        times_tr[i].copy_(times_id.to(DEVICE))

        del rays_o, rays_d, viewdirs, times_id

    eps_time = time.time() - eps_time
    print('get_training_rays: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, times_tr


@torch.no_grad()
def get_training_rays_flatten(rgb_tr_ori, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y, times):
    print('get_training_rays_flatten: start')
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
    eps_time = time.time()
    DEVICE = rgb_tr_ori[0].device
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)

    rgb_tr = torch.zeros([N, 3], device=DEVICE)
    rays_o_tr = torch.zeros_like(rgb_tr)
    rays_d_tr = torch.zeros_like(rgb_tr)
    viewdirs_tr = torch.zeros_like(rgb_tr)
    rays_tr = torch.zeros([N, 1], device=DEVICE)

    imsz = []
    top = 0
    for c2w, img, (H, W), K, time_i in zip(train_poses, rgb_tr_ori, HW, Ks, times):
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
            H=H, W=W, K=K, c2w=c2w, ndc=ndc,
            inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)

        rays_t = torch.ones([H, W, 1], device=DEVICE) * time_i

        n = H * W
        rgb_tr[top:top + n].copy_(img.flatten(0, 1))
        rays_o_tr[top:top + n].copy_(rays_o.flatten(0, 1).to(DEVICE))
        rays_d_tr[top:top + n].copy_(rays_d.flatten(0, 1).to(DEVICE))
        viewdirs_tr[top:top + n].copy_(viewdirs.flatten(0, 1).to(DEVICE))
        rays_tr[top:top + n].copy_(rays_t.flatten(0, 1).to(DEVICE))

        del rays_o, rays_d, viewdirs, rays_t

        imsz.append(n)
        top += n

    assert top == N
    eps_time = time.time() - eps_time
    print('get_training_rays_flatten: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, rays_tr


@torch.no_grad()
def get_training_rays_in_maskcache_sampling(rgb_tr_ori, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y, model,
                                            render_kwargs):
    print('get_training_rays_in_maskcache_sampling: start')
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
    CHUNK = 64
    DEVICE = rgb_tr_ori[0].device
    eps_time = time.time()
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    rgb_tr = torch.zeros([N, 3], device=DEVICE)
    rays_o_tr = torch.zeros_like(rgb_tr)
    rays_d_tr = torch.zeros_like(rgb_tr)
    viewdirs_tr = torch.zeros_like(rgb_tr)
    imsz = []
    top = 0
    for c2w, img, (H, W), K in zip(train_poses, rgb_tr_ori, HW, Ks):
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
            H=H, W=W, K=K, c2w=c2w, ndc=ndc,
            inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        mask = torch.ones(img.shape[:2], device=DEVICE, dtype=torch.bool)
        for i in range(0, img.shape[0], CHUNK):
            rays_pts, mask_outbbox = model.sample_ray(
                rays_o=rays_o[i:i + CHUNK], rays_d=rays_d[i:i + CHUNK], **render_kwargs)
            mask_outbbox[~mask_outbbox] |= (~model.mask_cache(rays_pts[~mask_outbbox]))
            mask[i:i + CHUNK] &= (~mask_outbbox).any(-1).to(DEVICE)
        n = mask.sum()
        rgb_tr[top:top + n].copy_(img[mask])
        rays_o_tr[top:top + n].copy_(rays_o[mask].to(DEVICE))
        rays_d_tr[top:top + n].copy_(rays_d[mask].to(DEVICE))
        viewdirs_tr[top:top + n].copy_(viewdirs[mask].to(DEVICE))
        imsz.append(n)
        top += n

    print('get_training_rays_in_maskcache_sampling: ratio', top / N)
    rgb_tr = rgb_tr[:top]
    rays_o_tr = rays_o_tr[:top]
    rays_d_tr = rays_d_tr[:top]
    viewdirs_tr = viewdirs_tr[:top]
    eps_time = time.time() - eps_time
    print('get_training_rays_in_maskcache_sampling: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz


def batch_indices_generator(N, BS):
    # torch.randperm on cuda produce incorrect results in my machine
    idx, top = torch.LongTensor(np.random.permutation(N)), 0
    while True:
        if top + BS > N:
            idx, top = torch.LongTensor(np.random.permutation(N)), 0
        yield idx[top:top + BS]
        top += BS


