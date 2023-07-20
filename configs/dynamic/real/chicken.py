_base_ = '/home/wsgan/project/nerf/V4D/configs/default_real.py'


data = dict(
    datadir='./data/real/faceforward/vrig-chicken',
    dataset_type='hyper_dataset',
    white_bkgd=False,
)