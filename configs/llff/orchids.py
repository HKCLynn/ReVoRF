_base_ = './llff_default.py'

expname = 'orchids'

data = dict(
    datadir='./data/nerf_llff_data/orchids',
    train_scene=[1, 12, 23]
)

