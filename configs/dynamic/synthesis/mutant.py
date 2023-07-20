_base_ = '/home/wsgan/project/nerf/V4D/configs/default_synthesis.py'

data = dict(
    datadir='./data/nerf_synthetic/mutant',
    dataset_type='blender_dynamic',
    white_bkgd=True,)
