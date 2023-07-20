import pdb
import warnings

warnings.filterwarnings("ignore")

import json
import os
import random

import numpy as np
import torch
from PIL import Image


class Load_hyper_data():
    def __init__(self,
                 datadir,
                 ratio=0.5,
                 use_bg_points=False,
                 add_cam=False,
                 ndc= False,

                 ):
        from .utils import Camera
        datadir = os.path.expanduser(datadir)
        with open(f'{datadir}/scene.json', 'r') as f:
            scene_json = json.load(f)
        with open(f'{datadir}/metadata.json', 'r') as f:
            meta_json = json.load(f)
        with open(f'{datadir}/dataset.json', 'r') as f:
            dataset_json = json.load(f)



        if ndc:
            self.near = 0
            self.far = 1

        else:
            self.near = scene_json['near']
            self.far = scene_json['far']


        self.coord_scale = scene_json['scale']
        self.scene_center = scene_json['center']

        self.all_img = dataset_json['ids']
        self.val_id = dataset_json['val_ids']

        self.add_cam = False
        if len(self.val_id) == 0:
            self.i_train = np.array([i for i in np.arange(len(self.all_img)) if
                                     (i % 4 == 0)])
            self.i_test = self.i_train + 2
            self.i_test = self.i_test[:-1, ]
        else:
            self.add_cam = True
            self.train_id = dataset_json['train_ids']
            self.i_test = []
            self.i_train = []
            for i in range(len(self.all_img)):
                id = self.all_img[i]
                if id in self.val_id:
                    self.i_test.append(i)
                if id in self.train_id:
                    self.i_train.append(i)
        assert self.add_cam == add_cam

        print('self.i_train', self.i_train)
        print('self.i_test', self.i_test)
        self.all_cam = [meta_json[i]['camera_id'] for i in self.all_img]
        self.all_time = [meta_json[i]['time_id'] for i in self.all_img]
        max_time = max(self.all_time)
        self.all_time = [meta_json[i]['time_id'] / max_time for i in self.all_img]
        self.selected_time = set(self.all_time)
        self.ratio = ratio

        # all poses
        self.all_cam_params = []


        # all_orientation = []
        #
        # for im in self.all_img:
        #     # pdb.set_trace()
        #     camera = Camera.from_json(f'{datadir}/camera/{im}.json')
        #     # (34, 3, 5)
        #     all_orientation.append(camera.orientation)
        # pdb.set_trace()
        # all_orientation = recenter_poses(np.array(all_orientation))


        for im in self.all_img:

            # pdb.set_trace()
            camera = Camera.from_json(f'{datadir}/camera/{im}.json')
            camera = camera.scale(ratio)
            camera.position = camera.position - self.scene_center
            camera.position = camera.position * self.coord_scale
            self.all_cam_params.append(camera)


        self.all_img = [f'{datadir}/rgb/{int(1 / ratio)}x/{i}.png' for i in self.all_img]
        self.h, self.w = self.all_cam_params[0].image_shape

        self.use_bg_points = use_bg_points

        if use_bg_points:
            with open(f'{datadir}/points.npy', 'rb') as f:
                points = np.load(f)
            self.bg_points = (points - self.scene_center) * self.coord_scale
            self.bg_points = torch.tensor(self.bg_points).float()

        print(f'total {len(self.all_img)} images ',
              'use cam =', self.add_cam,
              'use bg_point=', self.use_bg_points)

    def load_idx(self, idx, not_dic=False):

        all_data = self.load_raw(idx)
        if not_dic == True:
            rays_o = all_data['rays_ori']
            rays_d = all_data['rays_dir']
            viewdirs = all_data['viewdirs']
            rays_color = all_data['rays_color']
            return rays_o, rays_d, viewdirs, rays_color
        return all_data

    def load_raw(self, idx):
        image = Image.open(self.all_img[idx])
        camera = self.all_cam_params[idx]
        pixels = camera.get_pixel_centers()
        rays_dir = torch.tensor(camera.pixels_to_rays(pixels)).float().view([-1, 3])
        rays_ori = torch.tensor(camera.position[None, :]).float().expand_as(rays_dir)
        rays_color = torch.tensor(np.array(image)).view([-1, 3]) / 255.
        return {'rays_ori': rays_ori,
                'rays_dir': rays_dir,
                'viewdirs': rays_dir / rays_dir.norm(dim=-1, keepdim=True),
                'rays_color': rays_color,
                'near': torch.tensor(self.near).float().view([-1]),
                'far': torch.tensor(self.far).float().view([-1]), }


    def load_idx_1(self, idx, not_dic=False, ndc = False):

        all_data = self.load_raw_1(idx, ndc)
        if not_dic == True:
            rays_o = all_data['rays_ori']
            rays_d = all_data['rays_dir']
            viewdirs = all_data['viewdirs']
            rays_color = all_data['rays_color']
            time = all_data['time']

            return rays_o, rays_d, viewdirs, rays_color, time
        return all_data

    def load_raw_1(self, idx, ndc):
        image = Image.open(self.all_img[idx])
        camera = self.all_cam_params[idx]
        pixels = camera.get_pixel_centers()

        rays_dir = torch.tensor(camera.pixels_to_rays(pixels)).float()
        rays_ori = torch.tensor(camera.position[None, :]).float().expand_as(rays_dir)

        # rays_dir -> normalized or not
        viewdirs = rays_dir / rays_dir.norm(dim=-1, keepdim=True)

        if ndc:
            # pdb.set_trace()
            # rays_ori and rays_dir shape: [960, 536, 3]), camera.focal_length = camera.focal_length
            rays_ori, rays_dir = ndc_rays(self.h, self.w, camera.focal_length, 1., rays_ori, rays_dir)

        near = torch.tensor(self.near).float().view([-1])
        far = torch.tensor(self.far).float().view([-1])

        rays_color = torch.tensor(np.array(image)) / 255.
        time = torch.tensor(np.array(self.all_time[idx]))

        return {'rays_ori': rays_ori,
                'rays_dir': rays_dir,
                'viewdirs': viewdirs,
                'rays_color': rays_color,
                'time': time,
                'near': near,
                'far': far, }


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


def recenter_poses(poses):

    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses


def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w


def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def get_rays(H, W, K, c2w, inverse_y, flip_x, flip_y, mode='center'):

    # pdb.set_trace()
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
