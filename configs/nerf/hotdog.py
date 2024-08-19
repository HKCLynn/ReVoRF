_base_ = '../default.py'

expname = 'hotdog'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='./data/nerf_synthetic/hotdog',
    dataset_type='blender',
    white_bkgd=True,
    train_scene=[35, 37, 57, 88]
)

