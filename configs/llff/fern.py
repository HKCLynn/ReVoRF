_base_ = './llff_default.py'

expname = 'fern'

data = dict(
    datadir='./data/nerf_llff_data/fern',
    train_scene=[ 1, 10, 19]
)

