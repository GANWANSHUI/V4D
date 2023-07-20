# -*- encoding: utf-8 -*-
import os
import pdb

import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1):

    splits = ['train', 'val', 'test']

    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip

        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)

    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40+1)[:-1]], 0)

    if half_res:
        # pdb.set_trace()
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    return imgs, poses, render_poses, [H, W, focal], i_split


def load_blender_static_data(basedir, half_res=False, testskip=1, select_category=['ship', 'mic', "chair", 'ficus'], use_guassian_time=False):

    # select_category = ['ship', 'mic', "chair", 'ficus', 'hotdog', 'lego', 'drums', 'materials']
    select_category = ['ship', 'mic', "chair", 'ficus']
    # select_category = ['ship', 'mic']

    total_category = select_category

    splits = ['train', 'val', 'test']

    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_all_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    all_times = []
    counts = [0]

    for s in splits:

        meta = metas[s]
        imgs = []
        poses = []
        times = []
        # if s=='train' or testskip==0:
        #     skip = 2  # if you remove/change this 2, also change the /2 in the times vector
        # else:
        if s == 'train':

            skip = 1

        else:
            skip = testskip

        for c_id, c_name in enumerate(meta):  # c_id��0,1,2,3..., c_name: chair, drums,...

            if c_name not in select_category:
                continue

            # print('c_name:', c_name)
            # print(c_id, c_name, meta[c_name])
            for t, frame in enumerate(meta[c_name]['frames'][::skip]):
                # print('frame', frame['file_path'])

                if 'Users' in frame['file_path']:
                    frame['file_path'] = './' + frame['file_path'].split("\\", -1)[-1].split('//')[-1]

                fname = os.path.join(basedir, frame['file_path'][0:2] + c_name + '/' + frame['file_path'][2:] + '.png')
                imgs.append(imageio.imread(fname))

                pose = np.array(frame['transform_matrix'])
                # print('transform_matrix1:', pose)
                # pose = np.around(pose, 4)
                # print('transform_matrix2:', np.around(pose, 4))

                poses.append(pose)

                # if len(select_category) == 1:
                #     cur_time = total_category.index('{}'.format(c_name))  #
                # else:
                cur_time = total_category.index('{}'.format(c_name)) / (len(select_category))  #

                # print('cur_time:', cur_time)
                times.append(cur_time)
                # exit()

        # print('max, min:', np.max(times), np.min(times))

        # assert times[0] == 0, times[-1] != 1   # "Time must start at 0"

        imgs = (np.array(imgs) / 255.).astype(np.float32)  # keep all 4 channels (RGBA)

        # poses = np.array(poses).astype(np.float32)

        # ֻȡС�������λ
        poses = np.array(poses)
        poses = np.around(poses, 4).astype(np.float32)

        times = np.array(times).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])

        all_imgs.append(imgs)
        all_poses.append(poses)

        if s == 'train' and use_guassian_time != 0:
            times = get_guass_times(times, time_length=len(select_category), normal_factor=use_guassian_time)

        all_times.append(times)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    # print('i_split:', i_split, len(i_split))
    # print('len all_imgs:', len(all_imgs))
    # exit()

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    times = np.concatenate(all_times, 0)

    # print('times min mix:',  np.amin(times) , np.amax(times) )
    # exit()

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta[select_category[0]]['camera_angle_x'])

    focal = .5 * W / np.tan(.5 * camera_angle_x)

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]],
                               0).repeat(len(select_category), 1, 1)

    if len(select_category) == 1:
        render_times = torch.zeros(render_poses.shape[0])
    else:
        render_times = torch.ones(render_poses.shape[0])
        for i in range(0, render_poses.shape[0], 40):
            render_times[i:i + 40] = render_times[i:i + 40] * ((i / 40) / ((len(select_category))))

    render_times = render_times.to('cpu').numpy()

    # print('render_times:', render_times)

    if half_res:

        H = H // 2
        W = W // 2
        focal = focal / 2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (H, W), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()
    # pdb.set_trace()


    return imgs, poses, times, render_poses, render_times, [H, W, focal], i_split


def load_blender_dynamic_data(basedir, half_res=False, testskip=1, revise_t = False):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    all_times = []
    counts = [0]

    revision_delta = 0
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        times = []
        # if s=='train' or testskip==0:
        #     skip = 2  # if you remove/change this 2, also change the /2 in the times vector
        # else:
        skip = testskip

        if s == 'train' and revise_t:
            revision_delta = 1 / (len(meta['frames'][::skip]) - 1) - (1 / len(meta['frames'][::skip]))


        for t, frame in enumerate(meta['frames'][::skip]):
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))

            if 'time' in frame:
                if frame['time'] == 0:
                    cur_time = frame['time']

                else:
                    cur_time = frame['time'] - revision_delta

            else:
                print('no implement on dynamic loading')
                exit()
                cur_time = float(t) / (len(meta['frames'][::skip]) - 1)

            times.append(cur_time)

        assert times[0] == 0, "Time must start at 0"

        imgs = (np.array(imgs) / 255.).astype(np.float32)  # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        times = np.array(times).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
        all_times.append(times)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    times = np.concatenate(all_times, 0)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    # pdb.set_trace()
    if os.path.exists(os.path.join(basedir, 'transforms_{}.json'.format('render'))):
        with open(os.path.join(basedir, 'transforms_{}.json'.format('render')), 'r') as fp:
            meta = json.load(fp)
        render_poses = []
        for frame in meta['frames']:
            render_poses.append(np.array(frame['transform_matrix']))
        render_poses = np.array(render_poses).astype(np.float32)
    else:
        render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 100 + 1)[:-1]], 0)
    # pdb.set_trace()
    render_times = torch.linspace(0., 1., render_poses.shape[0])

    # pdb.set_trace()

    if half_res:

        H = H // 2
        W = W // 2
        focal = focal / 2.

        # H = H // 4
        # W = W // 4
        # focal = focal / 4.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (H, W), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    return imgs, poses, times, render_poses, render_times, [H, W, focal], i_split


