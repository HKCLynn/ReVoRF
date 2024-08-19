_base_ = '../default.py'

basedir = './logs/llff6v'

data = dict(
    dataset_type='llff',
    ndc=True,
    width=1008,
    height=756,
    rand_bkgd=True,
)

coarse_train = dict(
    N_iters=0,
    weight_entropy_last=0,
)

fine_train = dict(
    N_iters=15000,
    N_rand=4096,
    weight_distortion=0.01,
    pg_scale=[2000,4000,6000,8000],
    ug_scale=[10000,11000,12000,13000],
    ray_sampler='patch',
    tv_before=1e9,
    tv_dense_before=20000,
    weight_tv_density=1e-5,
    weight_tv_k0=1e-6,
    weight_entropy_last=0.0,
)

fine_model_and_render = dict(
    num_voxels=256**3,
    mpi_depth=128,
    rgbnet_dim=9,
    rgbnet_width=64,
    world_bound_scale=1,
    fast_color_thres=1e-3,
)

