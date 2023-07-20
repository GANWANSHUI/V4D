import math, sys
import pdb
import copy
import numpy as np
import scipy.signal
import pathlib
from typing import List
import torch.nn.functional as F
from torch import Tensor
import torch, time
import json
from pathlib import PurePosixPath as GPath
import torch.nn as nn
from typing import Any, Callable, List, Optional, Text, Tuple, Union
sys.path.append('/home/wsgan/nerf/D4D')


PRNGKey = Any
Shape = Tuple[int]
Dtype = Any  # this could be a real type?
Array = Any
Activation = Callable[[Array], Array]
Initializer = Callable[[PRNGKey, Shape, Dtype], Array]
Normalizer = Callable[[], Callable[[Array], Array]]
PathType = Union[Text, pathlib.PurePosixPath]

''' Misc
'''
mse2psnr = lambda x : -10. * torch.log10(x)
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


'''loss'''
def L1(x, M=None):
    if M == None:
        return torch.mean(torch.abs(x))
    else:
        return torch.sum(torch.abs(x) * M) / (torch.sum(M) + 1e-8) / x.shape[-1]


def L2(x, M=None):
    if M == None:
        return torch.mean(x ** 2)
    else:
        return torch.sum((x ** 2) * M) / (torch.sum(M) + 1e-8) / x.shape[-1]

# Spatial smoothness (adapted from NSFF)
def compute_sf_smooth_s_loss(pts1, pts2, H, W, f):

    N_samples = pts1.shape[1]

    # NDC coordinate to world coordinate
    pts1_world = NDC2world(pts1[..., :int(N_samples * 0.95), :], H, W, f)
    pts2_world = NDC2world(pts2[..., :int(N_samples * 0.95), :], H, W, f)

    # scene flow in world coordinate
    scene_flow_world = pts1_world - pts2_world

    return L1(scene_flow_world[..., :-1, :] - scene_flow_world[..., 1:, :])

# Temporal smoothness
def compute_sf_smooth_loss(pts, pts_f, pts_b, H, W, f):

    N_samples = pts.shape[1]

    pts_world   = NDC2world(pts[..., :int(N_samples * 0.9), :],   H, W, f)
    pts_f_world = NDC2world(pts_f[..., :int(N_samples * 0.9), :], H, W, f)
    pts_b_world = NDC2world(pts_b[..., :int(N_samples * 0.9), :], H, W, f)

    # scene flow in world coordinate
    sceneflow_f = pts_f_world - pts_world
    sceneflow_b = pts_b_world - pts_world

    # For a 3D point, its forward and backward sceneflow should be opposite.
    return L2(sceneflow_f + sceneflow_b)


def _compute_residual_and_jacobian(
    x: np.ndarray,
    y: np.ndarray,
    xd: np.ndarray,
    yd: np.ndarray,
    k1: float = 0.0,
    k2: float = 0.0,
    k3: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray]:
  """Auxiliary function of radial_and_tangential_undistort()."""

  r = x * x + y * y
  d = 1.0 + r * (k1 + r * (k2 + k3 * r))

  fx = d * x + 2 * p1 * x * y + p2 * (r + 2 * x * x) - xd
  fy = d * y + 2 * p2 * x * y + p1 * (r + 2 * y * y) - yd

  # Compute derivative of d over [x, y]
  d_r = (k1 + r * (2.0 * k2 + 3.0 * k3 * r))
  d_x = 2.0 * x * d_r
  d_y = 2.0 * y * d_r

  # Compute derivative of fx over x and y.
  fx_x = d + d_x * x + 2.0 * p1 * y + 6.0 * p2 * x
  fx_y = d_y * x + 2.0 * p1 * x + 2.0 * p2 * y

  # Compute derivative of fy over x and y.
  fy_x = d_x * y + 2.0 * p2 * y + 2.0 * p1 * x
  fy_y = d + d_y * y + 2.0 * p2 * x + 6.0 * p1 * y

  return fx, fy, fx_x, fx_y, fy_x, fy_y


def _radial_and_tangential_undistort(
    xd: np.ndarray,
    yd: np.ndarray,
    k1: float = 0,
    k2: float = 0,
    k3: float = 0,
    p1: float = 0,
    p2: float = 0,
    eps: float = 1e-9,
    max_iterations=10) -> Tuple[np.ndarray, np.ndarray]:
  """Computes undistorted (x, y) from (xd, yd)."""
  # Initialize from the distorted point.
  x = xd.copy()
  y = yd.copy()

  for _ in range(max_iterations):
    fx, fy, fx_x, fx_y, fy_x, fy_y = _compute_residual_and_jacobian(
        x=x, y=y, xd=xd, yd=yd, k1=k1, k2=k2, k3=k3, p1=p1, p2=p2)
    denominator = fy_x * fx_y - fx_x * fy_y
    x_numerator = fx * fy_y - fy * fx_y
    y_numerator = fy * fx_x - fx * fy_x
    step_x = np.where(
        np.abs(denominator) > eps, x_numerator / denominator,
        np.zeros_like(denominator))
    step_y = np.where(
        np.abs(denominator) > eps, y_numerator / denominator,
        np.zeros_like(denominator))

    x = x + step_x
    y = y + step_y

  return x, y




def NDC2world(pts, H, W, f):

    # NDC coordinate to world coordinate
    pts_z = 2 / (torch.clamp(pts[..., 2:], min=-1., max=1-1e-3) - 1)
    pts_x = - pts[..., 0:1] * pts_z * W / 2 / f
    pts_y = - pts[..., 1:2] * pts_z * H / 2 / f
    pts_world = torch.cat([pts_x, pts_y, pts_z], -1)

    return pts_world


''' Extend Adam to support per-voxel learning rate
'''
class Adam(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        self.per_lr = None
        super(Adam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def set_pervoxel_lr(self, count):
        assert self.param_groups[0]['params'][0].shape == count.shape
        self.per_lr = count.float() / count.max()



    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            per_lrs = []
            beta1, beta2 = group['betas']

            for p in group['params']:

                if p.grad is not None:

                    params_with_grad.append(p)

                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)

                    if self.per_lr is not None and p.shape == self.per_lr.shape:
                        per_lrs.append(self.per_lr)
                    else:
                        per_lrs.append(None)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])


            adam(params_with_grad,
                 grads,
                 exp_avgs,
                 exp_avg_sqs,
                 max_exp_avg_sqs,
                 state_steps,
                 amsgrad=group['amsgrad'],
                 beta1=beta1,
                 beta2=beta2,
                 lr=group['lr'],
                 weight_decay=group['weight_decay'],
                 eps=group['eps'],
                 per_lrs=per_lrs)
        return loss


def adam(params: List[Tensor],
         grads: List[Tensor],
         exp_avgs: List[Tensor],
         exp_avg_sqs: List[Tensor],
         max_exp_avg_sqs: List[Tensor],
         state_steps: List[int],
         *,
         amsgrad: bool,
         beta1: float,
         beta2: float,
         lr: float,
         weight_decay: float,
         eps: float,
         per_lrs):

    for i, param in enumerate(params):

        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]
        per_lr = per_lrs[i]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1

        if per_lr is not None:
            param.addcdiv_(exp_avg * per_lr, denom, value=-step_size)
        else:
            param.addcdiv_(exp_avg, denom, value=-step_size)

# global_step = 0


def create_optimizer_or_freeze_model(model, cfg_train, global_step):
    decay_steps = cfg_train.lrate_decay * 1000  # 20000
    decay_factor = 0.1 ** (global_step/decay_steps)

    param_group = []
    for k in cfg_train.keys():

        network_name = k

        if not k.startswith('lrate_'):
            continue
        k = k[len('lrate_'):]

        if not hasattr(model, k):
            continue

        param = getattr(model, k)
        if param is None:
            print(f'create_optimizer_or_freeze_model: param {k} not exist')
            continue

        lr = getattr(cfg_train, f'lrate_{k}') * decay_factor


        if lr > 0:
            print(f'create_optimizer_or_freeze_model: param {k} lr {lr}')
            if isinstance(param, nn.Module):
                param = param.parameters()


            # pdb.set_trace()
            param_group.append({'params': param, 'lr': lr, 'name': network_name})

            # pdb.set_trace()

        else:
            print(f'create_optimizer_or_freeze_model: param {k} freeze')

            param.requires_grad = False

    # pdb.set_trace()
    return Adam(param_group, betas=(0.9, 0.99))


''' Checkpoint utils
'''
def load_checkpoint(model, optimizer, ckpt_path, no_reload_optimizer):
    ckpt = torch.load(ckpt_path)
    start = ckpt['global_step']
    model.load_state_dict(ckpt['model_state_dict'])
    if not no_reload_optimizer:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return model, optimizer, start


def load_model(model_class, ckpt_path):
    ckpt = torch.load(ckpt_path)
    model = model_class(**ckpt['model_kwargs'])
    model.load_state_dict(ckpt['model_state_dict'])
    return model


''' Evaluation metrics (ssim, lpips)
'''
def rgb_ssim(img0, img1, max_val,
             filter_size=11,
             filter_sigma=1.5,
             k1=0.01,
             k2=0.03,
             return_map=False):
    # Modified from https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    assert len(img0.shape) == 3
    assert img0.shape[-1] == 3
    assert img0.shape == img1.shape

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma)**2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode='valid')

    filt_fn = lambda z: np.stack([
        convolve2d(convolve2d(z[...,i], filt[:, None]), filt[None, :])
        for i in range(z.shape[-1])], -1)
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2) - mu00
    sigma11 = filt_fn(img1**2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0., sigma00)
    sigma11 = np.maximum(0., sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(
        np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = np.mean(ssim_map)
    return ssim_map if return_map else ssim


__LPIPS__ = {}
def init_lpips(net_name, device):
    assert net_name in ['alex', 'vgg']
    import lpips
    print(f'init_lpips: lpips_{net_name}')
    return lpips.LPIPS(net=net_name, version='0.1').eval().to(device)

def rgb_lpips(np_gt, np_im, net_name, device):
    if net_name not in __LPIPS__:
        __LPIPS__[net_name] = init_lpips(net_name, device)
    gt = torch.from_numpy(np_gt).permute([2, 0, 1]).contiguous().to(device)
    im = torch.from_numpy(np_im).permute([2, 0, 1]).contiguous().to(device)
    return __LPIPS__[net_name](gt, im, normalize=True).item()


def mse(pred, gt, device):
    gt = torch.from_numpy(gt).permute([2, 0, 1]).contiguous().to(device)
    pred = torch.from_numpy(pred).permute([2, 0, 1]).contiguous().to(device)

    return torch.mean((pred - gt) ** 2)


class Generator3DLUT_identity(nn.Module):
    def __init__(self, dim=33):
        super(Generator3DLUT_identity, self).__init__()
        if dim == 33:
            file = open("./lib/IdentityLUT33.txt", 'r')
        elif dim == 64:
            file = open("./lib/IdentityLUT64.txt", 'r')
        lines = file.readlines()
        buffer = np.zeros((3,dim,dim,dim), dtype=np.float32)

        for i in range(0,dim):
            for j in range(0,dim):
                for k in range(0,dim):
                    n = i * dim*dim + j * dim + k
                    x = lines[n].split()
                    buffer[0,i,j,k] = float(x[0])
                    buffer[1,i,j,k] = float(x[1])
                    buffer[2,i,j,k] = float(x[2])
        self.LUT = nn.Parameter(torch.from_numpy(buffer).requires_grad_(True))


    def forward(self, x=0):
        return self.LUT


class Generator3DLUT_identity_1(nn.Module):
    def __init__(self, dim=33):
        super(Generator3DLUT_identity_1, self).__init__()
        if dim == 33:
            file = open("/home/wsgan/nerf/D4D/lib/IdentityLUT33.txt", 'r')
        elif dim == 64:
            file = open("/home/wsgan/nerf/D4D/lib/IdentityLUT64.txt", 'r')
        lines = file.readlines()
        buffer = np.zeros((3, dim, dim, dim), dtype=np.float32)

        for i in range(0, dim):
            for j in range(0, dim):
                for k in range(0, dim):
                    n = i * dim * dim + j * dim + k
                    x = lines[n].split()
                    buffer[0, i, j, k] = float(x[0])
                    buffer[1, i, j, k] = float(x[1])
                    buffer[2, i, j, k] = float(x[2])
        self.LUT = nn.Parameter(torch.from_numpy(buffer).requires_grad_(True))

    def forward(self, x=0):
        return self.LUT


class Generator3DLUT_zero(nn.Module):
    def __init__(self, dim=33):
        super(Generator3DLUT_zero, self).__init__()

        self.LUT = torch.zeros(3,dim,dim,dim, dtype=torch.float)
        self.LUT = nn.Parameter(torch.tensor(self.LUT))


    def forward(self, x = 0):

        return self.LUT

class TV_3D(nn.Module):
    def __init__(self, dim=33):
        super(TV_3D, self).__init__()

        self.weight_r = torch.ones(3, dim, dim, dim - 1, dtype=torch.float)
        self.weight_r[:, :, :, (0, dim - 2)] *= 2.0
        self.weight_g = torch.ones(3, dim, dim - 1, dim, dtype=torch.float)
        self.weight_g[:, :, (0, dim - 2), :] *= 2.0
        self.weight_b = torch.ones(3, dim - 1, dim, dim, dtype=torch.float)
        self.weight_b[:, (0, dim - 2), :, :] *= 2.0
        self.relu = torch.nn.ReLU()

    def forward(self, LUT):

        tv2 = LUT.LUT.diff(dim=1).abs()
        tv3 = LUT.LUT.diff(dim=2).abs()
        tv4 = LUT.LUT.diff(dim=3).abs()

        tv = (tv2.mean() + tv3.mean() + tv4.mean()) / 3
        mn = 0



        # dif_r = LUT.LUT[:, :, :, :-1] - LUT.LUT[:, :, :, 1:]
        # dif_g = LUT.LUT[:, :, :-1, :] - LUT.LUT[:, :, 1:, :]
        # dif_b = LUT.LUT[:, :-1, :, :] - LUT.LUT[:, 1:, :, :]
        # tv = torch.mean(torch.mul((dif_r ** 2), self.weight_r)) + torch.mean(
        #     torch.mul((dif_g ** 2), self.weight_g)) + torch.mean(torch.mul((dif_b ** 2), self.weight_b))
        #
        # mn = torch.mean(self.relu(dif_r)) + torch.mean(self.relu(dif_g)) + torch.mean(self.relu(dif_b))

        return tv, mn


def grid_sample_3d(image, optical):
    N, C, ID, IH, IW = image.shape
    _, D, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]
    iz = optical[..., 2]

    ix = ((ix + 1) / 2) * (IW - 1);
    iy = ((iy + 1) / 2) * (IH - 1);
    iz = ((iz + 1) / 2) * (ID - 1);
    with torch.no_grad():
        ix_tnw = torch.floor(ix);
        iy_tnw = torch.floor(iy);
        iz_tnw = torch.floor(iz);

        ix_tne = ix_tnw + 1;
        iy_tne = iy_tnw;
        iz_tne = iz_tnw;

        ix_tsw = ix_tnw;
        iy_tsw = iy_tnw + 1;
        iz_tsw = iz_tnw;

        ix_tse = ix_tnw + 1;
        iy_tse = iy_tnw + 1;
        iz_tse = iz_tnw;

        ix_bnw = ix_tnw;
        iy_bnw = iy_tnw;
        iz_bnw = iz_tnw + 1;

        ix_bne = ix_tnw + 1;
        iy_bne = iy_tnw;
        iz_bne = iz_tnw + 1;

        ix_bsw = ix_tnw;
        iy_bsw = iy_tnw + 1;
        iz_bsw = iz_tnw + 1;

        ix_bse = ix_tnw + 1;
        iy_bse = iy_tnw + 1;
        iz_bse = iz_tnw + 1;

    tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz);
    tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz);
    tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz);
    tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz);
    bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse);
    bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw);
    bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne);
    bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw);

    with torch.no_grad():
        torch.clamp(ix_tnw, 0, IW - 1, out=ix_tnw)
        torch.clamp(iy_tnw, 0, IH - 1, out=iy_tnw)
        torch.clamp(iz_tnw, 0, ID - 1, out=iz_tnw)

        torch.clamp(ix_tne, 0, IW - 1, out=ix_tne)
        torch.clamp(iy_tne, 0, IH - 1, out=iy_tne)
        torch.clamp(iz_tne, 0, ID - 1, out=iz_tne)

        torch.clamp(ix_tsw, 0, IW - 1, out=ix_tsw)
        torch.clamp(iy_tsw, 0, IH - 1, out=iy_tsw)
        torch.clamp(iz_tsw, 0, ID - 1, out=iz_tsw)

        torch.clamp(ix_tse, 0, IW - 1, out=ix_tse)
        torch.clamp(iy_tse, 0, IH - 1, out=iy_tse)
        torch.clamp(iz_tse, 0, ID - 1, out=iz_tse)

        torch.clamp(ix_bnw, 0, IW - 1, out=ix_bnw)
        torch.clamp(iy_bnw, 0, IH - 1, out=iy_bnw)
        torch.clamp(iz_bnw, 0, ID - 1, out=iz_bnw)

        torch.clamp(ix_bne, 0, IW - 1, out=ix_bne)
        torch.clamp(iy_bne, 0, IH - 1, out=iy_bne)
        torch.clamp(iz_bne, 0, ID - 1, out=iz_bne)

        torch.clamp(ix_bsw, 0, IW - 1, out=ix_bsw)
        torch.clamp(iy_bsw, 0, IH - 1, out=iy_bsw)
        torch.clamp(iz_bsw, 0, ID - 1, out=iz_bsw)

        torch.clamp(ix_bse, 0, IW - 1, out=ix_bse)
        torch.clamp(iy_bse, 0, IH - 1, out=iy_bse)
        torch.clamp(iz_bse, 0, ID - 1, out=iz_bse)

    image = image.view(N, C, ID * IH * IW)

    tnw_val = torch.gather(image, 2,
                           (iz_tnw * IW * IH + iy_tnw * IW + ix_tnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tne_val = torch.gather(image, 2,
                           (iz_tne * IW * IH + iy_tne * IW + ix_tne).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tsw_val = torch.gather(image, 2,
                           (iz_tsw * IW * IH + iy_tsw * IW + ix_tsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tse_val = torch.gather(image, 2,
                           (iz_tse * IW * IH + iy_tse * IW + ix_tse).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bnw_val = torch.gather(image, 2,
                           (iz_bnw * IW * IH + iy_bnw * IW + ix_bnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bne_val = torch.gather(image, 2,
                           (iz_bne * IW * IH + iy_bne * IW + ix_bne).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bsw_val = torch.gather(image, 2,
                           (iz_bsw * IW * IH + iy_bsw * IW + ix_bsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bse_val = torch.gather(image, 2,
                           (iz_bse * IW * IH + iy_bse * IW + ix_bse).long().view(N, 1, D * H * W).repeat(1, C, 1))

    out_val = (tnw_val.view(N, C, D, H, W) * tnw.view(N, 1, D, H, W) +
               tne_val.view(N, C, D, H, W) * tne.view(N, 1, D, H, W) +
               tsw_val.view(N, C, D, H, W) * tsw.view(N, 1, D, H, W) +
               tse_val.view(N, C, D, H, W) * tse.view(N, 1, D, H, W) +
               bnw_val.view(N, C, D, H, W) * bnw.view(N, 1, D, H, W) +
               bne_val.view(N, C, D, H, W) * bne.view(N, 1, D, H, W) +
               bsw_val.view(N, C, D, H, W) * bsw.view(N, 1, D, H, W) +
               bse_val.view(N, C, D, H, W) * bse.view(N, 1, D, H, W))

    return out_val


class NeRFOriginal(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, skips=[4], use_viewdirs=False, output_weight=5):
        super(NeRFOriginal, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        layers = [nn.Linear(input_ch, W)]
        for i in range(D - 1):
            layer = nn.Linear
            in_channels = W
            if i in self.skips:
                in_channels += input_ch

            layers += [layer(in_channels, W)]

        self.pts_linears = nn.ModuleList(layers)

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            # self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, output_weight)
            nn.init.constant_(self.rgb_linear.bias, 1)
        else:
            self.output_linear = nn.Linear(W, output_weight)
            nn.init.constant_(self.output_linear.bias, 1)


    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            # alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            outputs = self.rgb_linear(h)
            # outputs = rgb
            # outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs




class rgb_net(nn.Module):
    def __init__(self, rgb_input=256, rgbnet_width=256, rgb_out=3,  view_dependent='pre', input_ch_views = 0, rgbnet_depth = 5):
        super(rgb_net, self).__init__()

        self.view_dependent = view_dependent

        self.input_ch_views = input_ch_views

        self.rgb_input = rgb_input

        self.rgbnet_depth = rgbnet_depth

        if self.view_dependent == 'pre':
            self.rgbnet = nn.Sequential(nn.Linear(self.rgb_input, rgbnet_width), nn.ReLU(inplace=True),
                                        *[nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True))
                                          for _ in range(self.rgbnet_depth - 2)],
                                        nn.Linear(rgbnet_width, rgb_out), )

            nn.init.constant_(self.rgbnet[-1].bias, 0)

        elif self.view_dependent == 'post':
            self.pre_rgbnet = nn.Sequential(nn.Linear(self.rgb_input - self.input_ch_views, rgbnet_width), nn.ReLU(inplace=True),
                          *[nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True))
                            for _ in range(self.rgbnet_depth - 3)],)

            self.views_linears = nn.Sequential(nn.Linear(input_ch_views + rgbnet_width, rgbnet_width), nn.ReLU(inplace=True), nn.Linear(rgbnet_width, rgb_out))

            nn.init.constant_(self.views_linears[-1].bias, 0)

        else:
            print('error implementation!')
            exit()

    def forward(self, x):

        if self.view_dependent == 'pre':
            # rgb = self.rgbnet(x)
            return self.rgbnet(x)

        elif self.view_dependent == 'post':

            input_pts, input_views = torch.split(x, [self.rgb_input - self.input_ch_views, self.input_ch_views], dim=-1)

            pre_rgb_feat = self.pre_rgbnet(input_pts)

            post_rgb_feat = torch.cat([pre_rgb_feat, input_views], -1)

            # rgb = self.views_linears(post_rgb_feat)
            return self.views_linears(post_rgb_feat)

        else:
            print('error implementation!')
            exit()




class Camera:
  """Class to handle camera geometry."""

  def __init__(self,
               orientation: np.ndarray,
               position: np.ndarray,
               focal_length: Union[np.ndarray, float],
               principal_point: np.ndarray,
               image_size: np.ndarray,
               skew: Union[np.ndarray, float] = 0.0,
               pixel_aspect_ratio: Union[np.ndarray, float] = 1.0,
               radial_distortion: Optional[np.ndarray] = None,
               tangential_distortion: Optional[np.ndarray] = None,
               dtype=np.float32):
    """Constructor for camera class."""
    if radial_distortion is None:
      radial_distortion = np.array([0.0, 0.0, 0.0], dtype)
    if tangential_distortion is None:
      tangential_distortion = np.array([0.0, 0.0], dtype)

    self.orientation = np.array(orientation, dtype)
    self.position = np.array(position, dtype)
    self.focal_length = np.array(focal_length, dtype)
    self.principal_point = np.array(principal_point, dtype)
    self.skew = np.array(skew, dtype)
    self.pixel_aspect_ratio = np.array(pixel_aspect_ratio, dtype)
    self.radial_distortion = np.array(radial_distortion, dtype)
    self.tangential_distortion = np.array(tangential_distortion, dtype)
    self.image_size = np.array(image_size, np.uint32)
    self.dtype = dtype

  @classmethod
  def from_json(cls, path: PathType):
    """Loads a JSON camera into memory."""
    path = GPath(path)
    # with path.open('r') as fp:
    with open(path, 'r') as fp:
      camera_json = json.load(fp)

    # Fix old camera JSON.
    if 'tangential' in camera_json:
      camera_json['tangential_distortion'] = camera_json['tangential']

    return cls(
        orientation=np.asarray(camera_json['orientation']),
        position=np.asarray(camera_json['position']),
        focal_length=camera_json['focal_length'],
        principal_point=np.asarray(camera_json['principal_point']),
        skew=camera_json['skew'],
        pixel_aspect_ratio=camera_json['pixel_aspect_ratio'],
        radial_distortion=np.asarray(camera_json['radial_distortion']),
        tangential_distortion=np.asarray(camera_json['tangential_distortion']),
        image_size=np.asarray(camera_json['image_size']),
    )

  def to_json(self):
    return {
        k: (v.tolist() if hasattr(v, 'tolist') else v)
        for k, v in self.get_parameters().items()
    }

  def get_parameters(self):
    return {
        'orientation': self.orientation,
        'position': self.position,
        'focal_length': self.focal_length,
        'principal_point': self.principal_point,
        'skew': self.skew,
        'pixel_aspect_ratio': self.pixel_aspect_ratio,
        'radial_distortion': self.radial_distortion,
        'tangential_distortion': self.tangential_distortion,
        'image_size': self.image_size,
    }

  @property
  def scale_factor_x(self):
    return self.focal_length

  @property
  def scale_factor_y(self):
    return self.focal_length * self.pixel_aspect_ratio

  @property
  def principal_point_x(self):
    return self.principal_point[0]

  @property
  def principal_point_y(self):
    return self.principal_point[1]

  @property
  def has_tangential_distortion(self):
    return any(self.tangential_distortion != 0.0)

  @property
  def has_radial_distortion(self):
    return any(self.radial_distortion != 0.0)

  @property
  def image_size_y(self):
    return self.image_size[1]

  @property
  def image_size_x(self):
    return self.image_size[0]

  @property
  def image_shape(self):
    return self.image_size_y, self.image_size_x

  @property
  def optical_axis(self):
    return self.orientation[2, :]

  @property
  def translation(self):
    return -np.matmul(self.orientation, self.position)

  def pixel_to_local_rays(self, pixels: np.ndarray):
    """Returns the local ray directions for the provided pixels."""
    y = ((pixels[..., 1] - self.principal_point_y) / self.scale_factor_y)
    x = ((pixels[..., 0] - self.principal_point_x - y * self.skew) /
         self.scale_factor_x)

    if self.has_radial_distortion or self.has_tangential_distortion:
      x, y = _radial_and_tangential_undistort(
          x,
          y,
          k1=self.radial_distortion[0],
          k2=self.radial_distortion[1],
          k3=self.radial_distortion[2],
          p1=self.tangential_distortion[0],
          p2=self.tangential_distortion[1])

    dirs = np.stack([x, y, np.ones_like(x)], axis=-1)
    return dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)

  def pixels_to_rays(self, pixels: np.ndarray) -> np.ndarray:
    """Returns the rays for the provided pixels.

    Args:
      pixels: [A1, ..., An, 2] tensor or np.array containing 2d pixel positions.

    Returns:
        An array containing the normalized ray directions in world coordinates.
    """
    if pixels.shape[-1] != 2:
      raise ValueError('The last dimension of pixels must be 2.')
    if pixels.dtype != self.dtype:
      raise ValueError(f'pixels dtype ({pixels.dtype!r}) must match camera '
                       f'dtype ({self.dtype!r})')

    batch_shape = pixels.shape[:-1]
    pixels = np.reshape(pixels, (-1, 2))

    local_rays_dir = self.pixel_to_local_rays(pixels)
    rays_dir = np.matmul(self.orientation.T, local_rays_dir[..., np.newaxis])
    rays_dir = np.squeeze(rays_dir, axis=-1)

    # Normalize rays.

    # import pdb
    # pdb.set_trace()
    # rays_dir /= np.linalg.norm(rays_dir, axis=-1, keepdims=True)
    rays_dir = rays_dir.reshape((*batch_shape, 3))


    return rays_dir

  def pixels_to_points(self, pixels: np.ndarray, depth: np.ndarray):
    rays_through_pixels = self.pixels_to_rays(pixels)
    cosa = np.matmul(rays_through_pixels, self.optical_axis)
    points = (
        rays_through_pixels * depth[..., np.newaxis] / cosa[..., np.newaxis] +
        self.position)
    return points

  def points_to_local_points(self, points: np.ndarray):
    translated_points = points - self.position
    local_points = (np.matmul(self.orientation, translated_points.T)).T
    return local_points

  def project(self, points: np.ndarray):
    """Projects a 3D point (x,y,z) to a pixel position (x,y)."""
    batch_shape = points.shape[:-1]
    points = points.reshape((-1, 3))
    local_points = self.points_to_local_points(points)

    # Get normalized local pixel positions.
    x = local_points[..., 0] / local_points[..., 2]
    y = local_points[..., 1] / local_points[..., 2]
    r2 = x**2 + y**2

    # Apply radial distortion.
    distortion = 1.0 + r2 * (
        self.radial_distortion[0] + r2 *
        (self.radial_distortion[1] + self.radial_distortion[2] * r2))

    # Apply tangential distortion.
    x_times_y = x * y
    x = (
        x * distortion + 2.0 * self.tangential_distortion[0] * x_times_y +
        self.tangential_distortion[1] * (r2 + 2.0 * x**2))
    y = (
        y * distortion + 2.0 * self.tangential_distortion[1] * x_times_y +
        self.tangential_distortion[0] * (r2 + 2.0 * y**2))

    # Map the distorted ray to the image plane and return the depth.
    pixel_x = self.focal_length * x + self.skew * y + self.principal_point_x
    pixel_y = (self.focal_length * self.pixel_aspect_ratio * y
               + self.principal_point_y)

    pixels = np.stack([pixel_x, pixel_y], axis=-1)
    return pixels.reshape((*batch_shape, 2))

  def get_pixel_centers(self):
    """Returns the pixel centers."""
    xx, yy = np.meshgrid(np.arange(self.image_size_x, dtype=self.dtype),
                         np.arange(self.image_size_y, dtype=self.dtype))
    return np.stack([xx, yy], axis=-1) + 0.5

  def scale(self, scale: float):
    """Scales the camera."""
    if scale <= 0:
      raise ValueError('scale needs to be positive.')

    new_camera = Camera(
        orientation=self.orientation.copy(),
        position=self.position.copy(),
        focal_length=self.focal_length * scale,
        principal_point=self.principal_point.copy() * scale,
        skew=self.skew,
        pixel_aspect_ratio=self.pixel_aspect_ratio,
        radial_distortion=self.radial_distortion.copy(),
        tangential_distortion=self.tangential_distortion.copy(),
        image_size=np.array((int(round(self.image_size[0] * scale)),
                             int(round(self.image_size[1] * scale)))),
    )
    return new_camera

  def look_at(self, position, look_at, up, eps=1e-6):
    """Creates a copy of the camera which looks at a given point.

    Copies the provided vision_sfm camera and returns a new camera that is
    positioned at `camera_position` while looking at `look_at_position`.
    Camera intrinsics are copied by this method. A common value for the
    up_vector is (0, 1, 0).

    Args:
      position: A (3,) numpy array representing the position of the camera.
      look_at: A (3,) numpy array representing the location the camera
        looks at.
      up: A (3,) numpy array representing the up direction, whose
        projection is parallel to the y-axis of the image plane.
      eps: a small number to prevent divides by zero.

    Returns:
      A new camera that is copied from the original but is positioned and
        looks at the provided coordinates.

    Raises:
      ValueError: If the camera position and look at position are very close
        to each other or if the up-vector is parallel to the requested optical
        axis.
    """

    look_at_camera = self.copy()
    optical_axis = look_at - position
    norm = np.linalg.norm(optical_axis)
    if norm < eps:
      raise ValueError('The camera center and look at position are too close.')
    optical_axis /= norm

    right_vector = np.cross(optical_axis, up)
    norm = np.linalg.norm(right_vector)
    if norm < eps:
      raise ValueError('The up-vector is parallel to the optical axis.')
    right_vector /= norm

    # The three directions here are orthogonal to each other and form a right
    # handed coordinate system.
    camera_rotation = np.identity(3)
    camera_rotation[0, :] = right_vector
    camera_rotation[1, :] = np.cross(optical_axis, right_vector)
    camera_rotation[2, :] = optical_axis

    look_at_camera.position = position
    look_at_camera.orientation = camera_rotation
    return look_at_camera

  def crop_image_domain(
      self, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0):
    """Returns a copy of the camera with adjusted image bounds.

    Args:
      left: number of pixels by which to reduce (or augment, if negative) the
        image domain at the associated boundary.
      right: likewise.
      top: likewise.
      bottom: likewise.

    The crop parameters may not cause the camera image domain dimensions to
    become non-positive.

    Returns:
      A camera with adjusted image dimensions.  The focal length is unchanged,
      and the principal point is updated to preserve the original principal
      axis.
    """

    crop_left_top = np.array([left, top])
    crop_right_bottom = np.array([right, bottom])
    new_resolution = self.image_size - crop_left_top - crop_right_bottom
    new_principal_point = self.principal_point - crop_left_top
    if np.any(new_resolution <= 0):
      raise ValueError('Crop would result in non-positive image dimensions.')

    new_camera = self.copy()
    new_camera.image_size = np.array([int(new_resolution[0]),
                                      int(new_resolution[1])])
    new_camera.principal_point = np.array([new_principal_point[0],
                                           new_principal_point[1]])
    return new_camera

  def copy(self):
    return copy.deepcopy(self)



if __name__ == "__main__":
    if True:
        image = torch.rand(1,1,68,71,103)
        grid = torch.rand(1,1,1,8192,3)

        # image = torch.rand(1, 3, 200, 300, 100)
        # grid = torch.rand(1, 100, 100, 2, 3)

        start = time.time()
        output1 = grid_sample_3d(image, grid)
        print('output1', output1)
        end = time.time()
        print('using {}'.format(end - start))

        start = time.time()
        output2 = F.grid_sample(image, grid, padding_mode='border', align_corners=True)
        print('output2', output2)
        end = time.time()
        print('using {}'.format(end - start))

        '''
         # grids[0].shape = [1,1,68,71,103]
        # ind_norm.shape = [1,1,1,8192,3]
        
        '''