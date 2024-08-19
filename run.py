import os
import sys, copy, glob, json, time, random, argparse
from shutil import copyfile
from tqdm import tqdm, trange

import mmengine
import imageio
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import utils, dvgo, dcvgo, dmpigo
from lib.load_data import load_data
from lib.general import *
from lib.warping import *
from lib.models import DPT_model
import matplotlib.pyplot as plt


from torch_efficient_distloss import flatten_eff_distloss



def config_parser():
    '''Define command line arguments
    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True,
                        help='config file path')
    parser.add_argument('--txtfile',type=str, default='',
                        help='config file path')
    parser.add_argument("--seed", type=int, default=777,
                        help='Random seed')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--no_reload_optimizer", action='store_true',
                        help='do not reload optimizer state from saved ckpt')
    parser.add_argument("--ft_path", type=str, default='',
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--export_bbox_and_cams_only", type=str, default='',
                        help='export scene bbox and camera poses for debugging and 3d visualization')
    parser.add_argument("--export_coarse_only", type=str, default='')
    parser.add_argument("--dpt_model_path",type=str,default='')

    # testing options
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true')
    parser.add_argument("--render_train", action='store_true')
    parser.add_argument("--render_video", action='store_true')
    parser.add_argument("--render_video_flipy", action='store_true')
    parser.add_argument("--render_video_rot90", default=0, type=int)
    parser.add_argument("--render_video_factor", type=float, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--dump_images", action='store_true')
    parser.add_argument("--eval_ssim", action='store_true')
    parser.add_argument("--eval_lpips_alex", action='store_true')
    parser.add_argument("--eval_lpips_vgg", action='store_true')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=500,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=100000,
                        help='frequency of weight ckpt saving')
    return parser


@torch.no_grad()
def render_viewpoints(model, render_poses, HW, Ks, ndc, render_kwargs,
                      gt_imgs=None, savedir=None, dump_images=False,
                      render_factor=0, render_video_flipy=False, render_video_rot90=0,
                      eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False):
    '''Render images for the given viewpoints; run evaluation if gt given.
    '''
    assert len(render_poses) == len(HW) and len(HW) == len(Ks)

    if render_factor!=0:
        HW = np.copy(HW)
        Ks = np.copy(Ks)
        HW = (HW/render_factor).astype(int)
        Ks[:, :2, :3] /= render_factor

    rgbs = []
    depths = []
    bgmaps = []
    psnrs = []
    ssims = []
    lpips_alex = []
    lpips_vgg = []

    for i, c2w in enumerate(tqdm(render_poses)):

        H, W = HW[i]
        K = Ks[i]
        c2w = torch.Tensor(c2w)
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H, W, K, c2w, ndc, inverse_y=render_kwargs['inverse_y'],
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        keys = ['rgb_marched', 'depth', 'alphainv_last']
        rays_o = rays_o.flatten(0,-2)
        rays_d = rays_d.flatten(0,-2)
        viewdirs = viewdirs.flatten(0,-2)
        render_result_chunks = [
            {k: v for k, v in model(ro, rd, vd, **render_kwargs).items() if k in keys}
            for ro, rd, vd in zip(rays_o.split(8192, 0), rays_d.split(8192, 0), viewdirs.split(8192, 0))
        ]
        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H,W,-1)
            for k in render_result_chunks[0].keys()
        }
        rgb = render_result['rgb_marched'].cpu().numpy()
        depth = render_result['depth'].cpu().numpy()
        bgmap = render_result['alphainv_last'].cpu().numpy()

        rgbs.append(rgb)
        depths.append(depth)
        bgmaps.append(bgmap)
        if i==0:
            print('Testing', rgb.shape)

        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
            f.write('psnr '+str(p))
            f.write('\n')
            psnrs.append(p)
            if eval_ssim:
                ssims.append(utils.rgb_ssim(rgb, gt_imgs[i], max_val=1))
            if eval_lpips_alex:
                lpips_alex.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='alex', device=c2w.device))
            if eval_lpips_vgg:
                lpips_vgg.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='vgg', device=c2w.device))

    if len(psnrs):
        print('Testing psnr', np.mean(psnrs), '(avg)')
        f.write('psnr '+str(np.mean(psnrs)))
        f.write('\n')
        if eval_ssim: 
            print('Testing ssim', np.mean(ssims), '(avg)')
            f.write('ssim '+str(np.mean(ssims)))
            f.write('\n')
        if eval_lpips_vgg: 
            print('Testing lpips (vgg)', np.mean(lpips_vgg), '(avg)')
            f.write('lpips (vgg) '+str(np.mean(lpips_vgg)))
            f.write('\n')
        if eval_lpips_alex: 
            print('Testing lpips (alex)', np.mean(lpips_alex), '(avg)')
            f.write('lpips (alex) '+str(np.mean(lpips_alex)))
            f.write('\n')

    if render_video_flipy:
        for i in range(len(rgbs)):
            rgbs[i] = np.flip(rgbs[i], axis=0)
            depths[i] = np.flip(depths[i], axis=0)
            bgmaps[i] = np.flip(bgmaps[i], axis=0)

    if render_video_rot90 != 0:
        for i in range(len(rgbs)):
            rgbs[i] = np.rot90(rgbs[i], k=render_video_rot90, axes=(0,1))
            depths[i] = np.rot90(depths[i], k=render_video_rot90, axes=(0,1))
            bgmaps[i] = np.rot90(bgmaps[i], k=render_video_rot90, axes=(0,1))

    if savedir is not None and dump_images:
        for i in trange(len(rgbs)):
            rgb8 = utils.to8b(rgbs[i])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.array(rgbs)
    depths = np.array(depths)
    bgmaps = np.array(bgmaps)

    return rgbs, depths, bgmaps

@torch.no_grad()
def render_noise_view(model, render_poses, HW, Ks, ndc, render_kwargs, render_factor=0):
    '''Render images for the given viewpoints; run evaluation if gt given.
    '''
    # assert len(render_poses) == len(HW) and len(HW) == len(Ks)

    if render_factor!=0:
        HW = np.copy(HW)
        Ks = np.copy(Ks)
        HW = (HW/render_factor).astype(int)
        Ks[:, :2, :3] /= render_factor

    rgbs = []
    depths = []
    bgmaps = []
    H, W = HW
    K = Ks

    for i, c2w in enumerate(tqdm(render_poses)):

        c2w = torch.Tensor(c2w)
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H, W, K, c2w, ndc, inverse_y=render_kwargs['inverse_y'],
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        keys = ['rgb_marched', 'depth', 'alphainv_last']
        rays_o = rays_o.flatten(0,-2)
        rays_d = rays_d.flatten(0,-2)
        viewdirs = viewdirs.flatten(0,-2)
        render_result_chunks = [
            {k: v for k, v in model(ro, rd, vd, **render_kwargs).items() if k in keys}
            for ro, rd, vd in zip(rays_o.split(8192, 0), rays_d.split(8192, 0), viewdirs.split(8192, 0))
        ]
        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H,W,-1)
            for k in render_result_chunks[0].keys()
        }
        rgb = render_result['rgb_marched'].cpu().numpy()
        depth = render_result['depth'].cpu().numpy()
        bgmap = render_result['alphainv_last'].cpu().numpy()

        rgbs.append(rgb)
        depths.append(depth)
        bgmaps.append(bgmap)

    rgbs = np.array(rgbs)
    depths = np.array(depths)
    bgmaps = np.array(bgmaps)

    return rgbs, depths, bgmaps


def seed_everything():
    '''Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    '''
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def load_everything(args, cfg):
    '''Load images / poses / camera settings / data split.
    '''
    data_dict = load_data(cfg.data)

    # remove useless field
    kept_keys = {
            'hwf', 'HW', 'Ks', 'near', 'far', 'near_clip',
            'i_train', 'i_val', 'i_test', 'irregular_shape',
            'poses', 'render_poses', 'images'}
    for k in list(data_dict.keys()):
        if k not in kept_keys:
            data_dict.pop(k)

    # construct data tensor
    if data_dict['irregular_shape']:
        data_dict['images'] = [torch.FloatTensor(im, device='cpu') for im in data_dict['images']]
    else:
        data_dict['images'] = torch.FloatTensor(data_dict['images'], device='cpu')
    data_dict['poses'] = torch.Tensor(data_dict['poses'])
    return data_dict


def _compute_bbox_by_cam_frustrm_bounded(cfg, HW, Ks, poses, i_train, near, far):
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for (H, W), K, c2w in zip(HW[i_train], Ks[i_train], poses[i_train]):
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w,
                ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        if cfg.data.ndc:
            pts_nf = torch.stack([rays_o+rays_d*near, rays_o+rays_d*far])
        else:
            pts_nf = torch.stack([rays_o+viewdirs*near, rays_o+viewdirs*far])
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0,1,2)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0,1,2)))
    return xyz_min, xyz_max

def _compute_bbox_by_cam_frustrm_unbounded(cfg, HW, Ks, poses, i_train, near_clip):
    # Find a tightest cube that cover all camera centers
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for (H, W), K, c2w in zip(HW[i_train], Ks[i_train], poses[i_train]):
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w,
                ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        pts = rays_o + rays_d * near_clip
        xyz_min = torch.minimum(xyz_min, pts.amin((0,1)))
        xyz_max = torch.maximum(xyz_max, pts.amax((0,1)))
    center = (xyz_min + xyz_max) * 0.5
    radius = (center - xyz_min).max() * cfg.data.unbounded_inner_r
    xyz_min = center - radius
    xyz_max = center + radius
    return xyz_min, xyz_max

def compute_bbox_by_cam_frustrm(args, cfg, HW, Ks, poses, i_train, near, far, **kwargs):
    print('compute_bbox_by_cam_frustrm: start')
    if cfg.data.unbounded_inward:
        xyz_min, xyz_max = _compute_bbox_by_cam_frustrm_unbounded(
                cfg, HW, Ks, poses, i_train, kwargs.get('near_clip', None))

    else:
        xyz_min, xyz_max = _compute_bbox_by_cam_frustrm_bounded(
                cfg, HW, Ks, poses, i_train, near, far)
    print('compute_bbox_by_cam_frustrm: xyz_min', xyz_min)
    print('compute_bbox_by_cam_frustrm: xyz_max', xyz_max)
    print('compute_bbox_by_cam_frustrm: finish')
    return xyz_min, xyz_max

@torch.no_grad()
def compute_bbox_by_coarse_geo(model_class, model_path, thres):
    print('compute_bbox_by_coarse_geo: start')
    eps_time = time.time()
    model = utils.load_model(model_class, model_path)
    interp = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, model.world_size[0]),
        torch.linspace(0, 1, model.world_size[1]),
        torch.linspace(0, 1, model.world_size[2]),
    ), -1)
    dense_xyz = model.xyz_min * (1-interp) + model.xyz_max * interp
    density = model.density(dense_xyz)
    alpha = model.activate_density(density)
    mask = (alpha > thres)
    active_xyz = dense_xyz[mask]
    xyz_min = active_xyz.amin(0)
    xyz_max = active_xyz.amax(0)
    print('compute_bbox_by_coarse_geo: xyz_min', xyz_min)
    print('compute_bbox_by_coarse_geo: xyz_max', xyz_max)
    eps_time = time.time() - eps_time
    print('compute_bbox_by_coarse_geo: finish (eps time:', eps_time, 'secs)')
    return xyz_min, xyz_max

def create_new_model(cfg, cfg_model, cfg_train, xyz_min, xyz_max, stage, coarse_ckpt_path):
    model_kwargs = copy.deepcopy(cfg_model)
    num_voxels = model_kwargs.pop('num_voxels')
    if len(cfg_train.pg_scale):
        num_voxels = int(num_voxels / (2**len(cfg_train.pg_scale)))

    if cfg.data.ndc:
        print(f'scene_rep_reconstruction ({stage}): \033[96muse multiplane images\033[0m')
        model = dmpigo.DirectMPIGO(
            xyz_min=xyz_min, xyz_max=xyz_max,
            num_voxels=num_voxels,
            **model_kwargs)
    elif cfg.data.unbounded_inward:
        print(f'scene_rep_reconstruction ({stage}): \033[96muse contraced voxel grid (covering unbounded)\033[0m')
        model = dcvgo.DirectContractedVoxGO(
            xyz_min=xyz_min, xyz_max=xyz_max,
            num_voxels=num_voxels,
            **model_kwargs)
    else:
        print(f'scene_rep_reconstruction ({stage}): \033[96muse dense voxel grid\033[0m')
        model = dvgo.DirectVoxGO(
            xyz_min=xyz_min, xyz_max=xyz_max,
            num_voxels=num_voxels,
            mask_cache_path=coarse_ckpt_path,
            **model_kwargs)
    model = model.to(device)
    optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
    return model, optimizer

def load_existed_model(args, cfg, cfg_train, reload_ckpt_path):
    if cfg.data.ndc:
        model_class = dmpigo.DirectMPIGO
    elif cfg.data.unbounded_inward:
        model_class = dcvgo.DirectContractedVoxGO
    else:
        model_class = dvgo.DirectVoxGO
    model = utils.load_model(model_class, reload_ckpt_path).to(device)
    optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
    model, optimizer, start = utils.load_checkpoint(
            model, optimizer, reload_ckpt_path, args.no_reload_optimizer)
    return model, optimizer, start

def load_sparse_depth_views(cfg,data_dict):
    depths=[]
    if cfg.data['hardcode_train_views']==True:
        data_dict['i_train']=cfg.data['train_scene']
        print('Train views:', data_dict['i_train'])
    else:
        sub = list(set(list(data_dict['i_train']))-set(list(data_dict['i_test'])))
        idx_sub = np.linspace(0,len(sub)-1,cfg.data['max_train_views'])
        # i_train = cfg.data['train_scene']
        idx_sub = [round(i) for i in idx_sub]
        data_dict['i_train']=data_dict['i_train'][idx_sub]
        print('Train views:', data_dict['i_train'])
    if cfg.data['dataset_type'] == 'blender':
        for i in data_dict['i_train']:
            depths.append(torch.tensor(imageio.imread('./depths/nerf_synthetic/'+str(cfg['expname'])+'/r_'+str(i)+'.png')/1.0))
        depths=torch.stack(depths,dim=0)
    elif cfg.data['dataset_type'] == 'llff':
        imgfiles = [os.path.join('/depths/nerf_llff_data/'+str(cfg['expname']), f) for f in sorted(os.listdir('data/nerf_llff_data/'+str(cfg['expname'])+'/depths')) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
        for i in data_dict['i_train']:
            depths.append(torch.tensor(imageio.imread(imgfiles[i])/1.0))
        depths=torch.stack(depths,dim=0)
        
    return depths


def scene_rep_reconstruction(args, cfg, cfg_model, cfg_train, xyz_min, xyz_max, data_dict, stage, depth_model, coarse_ckpt_path=None):
    # init
    i_train_ori = data_dict['i_train']
    print('Original training views:', i_train_ori)
    depths=load_sparse_depth_views(cfg,data_dict)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    if abs(cfg_model.world_bound_scale - 1) > 1e-9:
        xyz_shift = (xyz_max - xyz_min) * (cfg_model.world_bound_scale - 1) / 2
        xyz_min -= xyz_shift
        xyz_max += xyz_shift
    HW, Ks, near, far, i_train, i_val, i_test, poses, render_poses, images = [
        data_dict[k] for k in [
            'HW', 'Ks', 'near', 'far', 'i_train', 'i_val', 'i_test', 'poses', 'render_poses', 'images'
        ]
    ]

    # find whether there is existing checkpoint path
    last_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_last.tar')
    if args.no_reload:
        reload_ckpt_path = None
    elif args.ft_path:
        reload_ckpt_path = args.ft_path
    elif os.path.isfile(last_ckpt_path):
        reload_ckpt_path = last_ckpt_path
    else:
        reload_ckpt_path = None

    # init model and optimizer
    if reload_ckpt_path is None:
        print(f'scene_rep_reconstruction ({stage}): train from scratch')
        model, optimizer = create_new_model(cfg, cfg_model, cfg_train, xyz_min, xyz_max, stage, coarse_ckpt_path)
        start = 0
        if cfg_model.maskout_near_cam_vox:
            model.maskout_near_cam_vox(poses[i_train,:3,3], near)
    else:
        print(f'scene_rep_reconstruction ({stage}): reload from {reload_ckpt_path}')
        model, optimizer, start = load_existed_model(args, cfg, cfg_train, reload_ckpt_path)

    # init rendering setup
    render_kwargs = {
        'near': data_dict['near'],
        'far': data_dict['far'],
        'bg': 1 if cfg.data.white_bkgd else 0,
        'rand_bkgd': cfg.data.rand_bkgd,
        'stepsize': cfg_model.stepsize,
        'inverse_y': cfg.data.inverse_y,
        'flip_x': cfg.data.flip_x,
        'flip_y': cfg.data.flip_y,
    }
    extra_data={}
    
    def gather_random_rays():
        rays_o_rd, rays_d_rd, viewdirs_rd, imsz_rd = dvgo.get_random_rays(
            train_poses=poses[i_train_ori],
            HW=HW[i_train_ori], Ks=Ks[i_train_ori], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
            flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,n_poses=20)
        return rays_o_rd, rays_d_rd, viewdirs_rd, imsz_rd
        
    if cfg_train.N_rand_sample:  
        rays_o_rd_tr, rays_d_rd_tr, viewdirs_rd_tr, imsz_rd_tr = gather_random_rays()


    if cfg_train.ray_sampler in ['flatten','in_maskcache']:
    # init batch rays sampler
        def gather_training_rays():
            if data_dict['irregular_shape']:
                rgb_tr_ori = [images[i].to('cpu' if cfg.data.load2gpu_on_the_fly else device) for i in i_train]
            else:
                rgb_tr_ori = images[i_train].to('cpu' if cfg.data.load2gpu_on_the_fly else device)

            if cfg_train.ray_sampler == 'in_maskcache':
                rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays_in_maskcache_sampling(
                        rgb_tr_ori=rgb_tr_ori,
                        train_poses=poses[i_train],
                        HW=HW[i_train], Ks=Ks[i_train],
                        ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                        flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,
                        model=model, render_kwargs=render_kwargs)
            elif cfg_train.ray_sampler == 'flatten':
                rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays_flatten(
                    rgb_tr_ori=rgb_tr_ori,
                    train_poses=poses[i_train],
                    HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                    flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
            else:
                rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays(
                    rgb_tr=rgb_tr_ori,
                    train_poses=poses[i_train],
                    HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                    flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
            index_generator = dvgo.batch_indices_generator(len(rgb_tr), cfg_train.N_rand)
            batch_index_sampler = lambda: next(index_generator)
            return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler

        rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler = gather_training_rays()

    elif cfg_train.ray_sampler == 'patch':
        rgb_tr = images[i_train].to('cpu' if cfg.data.load2gpu_on_the_fly else device)
        pose_tr,rays_o_tr, rays_d_tr, viewdirs_tr=[],[],[],[]
        for c2w, (H, W), K in zip(poses[i_train], HW[i_train], Ks[i_train]):
            rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                    H, W, K, c2w, cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                    flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,)
            
            pose_tr.append(c2w)
            rays_o_tr.append(rays_o)
            rays_d_tr.append(rays_d)
            viewdirs_tr.append(viewdirs)
            
        rays_o_tr=torch.stack(rays_o_tr,dim=0)
        rays_d_tr=torch.stack(rays_d_tr,dim=0)
        viewdirs_tr=torch.stack(viewdirs_tr,dim=0)

    else:
        raise NotImplementedError
        
    # view-count-based learning rate
    if cfg_train.pervoxel_lr:
        def per_voxel_init():
            cnt = model.voxel_count_views(
                    rays_o_tr=rays_o_rd_tr, rays_d_tr=rays_d_rd_tr, imsz=imsz_rd_tr, near=near, far=far,
                    stepsize=cfg_model.stepsize, downrate=cfg_train.pervoxel_lr_downrate,
                    irregular_shape=data_dict['irregular_shape'])
            optimizer.set_pervoxel_lr(cnt)
            model.mask_cache.mask[cnt.squeeze() <= 2] = False
        per_voxel_init()

    if cfg_train.maskout_lt_nviews > 0:
        model.update_occupancy_cache_lt_nviews(
                rays_o_tr, rays_d_tr, imsz, render_kwargs, cfg_train.maskout_lt_nviews)

    # GOGO
    torch.cuda.empty_cache()
    psnr_lst = []
    time0 = time.time()
    global_step = -1
    
    for global_step in trange(start, cfg_train.N_iters):

        # renew occupancy grid
        if model.mask_cache is not None and (global_step + 500) % 1000 == 0:
            model.update_occupancy_cache()

        # progress scaling checkpoint
        if global_step in cfg_train.pg_scale:
            n_rest_scales = len(cfg_train.pg_scale)-cfg_train.pg_scale.index(global_step)-1
            cur_voxels = int(cfg_model.num_voxels / (2**n_rest_scales))
            if isinstance(model, (dvgo.DirectVoxGO, dcvgo.DirectContractedVoxGO)):
                model.scale_volume_grid(cur_voxels)
            elif isinstance(model, dmpigo.DirectMPIGO):
                model.scale_volume_grid(cur_voxels, model.mpi_depth)
            else:
                raise NotImplementedError
            optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
            model.act_shift -= cfg_train.decay_after_scale
            torch.cuda.empty_cache()

        # random sample rays
        if cfg_train.ray_sampler in ['flatten','in_maskcache']:
            sel_i = batch_index_sampler()             
            target = rgb_tr[sel_i]
            rays_o = rays_o_tr[sel_i]
            rays_d = rays_d_tr[sel_i]
            viewdirs = viewdirs_tr[sel_i]  
        elif cfg_train.ray_sampler == 'random':
            sel_b = torch.randint(rgb_tr.shape[0], [cfg_train.N_rand])
            sel_r = torch.randint(rgb_tr.shape[1], [cfg_train.N_rand])
            sel_c = torch.randint(rgb_tr.shape[2], [cfg_train.N_rand])
            target = rgb_tr[sel_b, sel_r, sel_c]
            rays_o = rays_o_tr[sel_b, sel_r, sel_c]
            rays_d = rays_d_tr[sel_b, sel_r, sel_c]
            viewdirs = viewdirs_tr[sel_b, sel_r, sel_c]
            
        elif cfg_train.ray_sampler == 'patch':
            patch_size=cfg_train.patch_size
            patch_num=cfg_train.normal_patch_num
            H, W = HW[0]
            # Sample pose(s)
            idx_img = np.random.randint(0, rgb_tr.shape[0], size=(patch_num, 1))
            # Sample start locations
            x0 = np.random.randint(0, W - patch_size + 1, size=(patch_num, 1, 1))
            y0 = np.random.randint(0, H - patch_size + 1, size=(patch_num, 1, 1))
            xy0 = np.concatenate([x0, y0], axis=-1)
            patch_idx = xy0 + np.stack(
                np.meshgrid(np.arange(patch_size), np.arange(patch_size), indexing='xy'),
                axis=-1).reshape(1, -1, 2)
            rays_o = rays_o_tr[idx_img, patch_idx[Ellipsis, 1], patch_idx[Ellipsis, 0]].reshape(-1, 3)
            rays_d = rays_d_tr[idx_img, patch_idx[Ellipsis, 1], patch_idx[Ellipsis, 0]].reshape(-1, 3)
            viewdirs = viewdirs_tr[idx_img, patch_idx[Ellipsis, 1], patch_idx[Ellipsis, 0]].reshape(-1, 3)
            target = rgb_tr[idx_img, patch_idx[Ellipsis, 1], patch_idx[Ellipsis, 0]].reshape(-1, 3)
            gt_depths = depths[idx_img, patch_idx[Ellipsis, 1], patch_idx[Ellipsis, 0]].reshape(-1,patch_size,patch_size)
            
            if len(extra_data) != 0:
                warp_patch_size=cfg_train.patch_size
                warp_patch_num=cfg_train.normal_patch_num
                
                H, W = HW[0]
                # Sample pose(s)
                warp_idx_img = np.random.randint(0, warp_rgb_tr.shape[0], size=(warp_patch_num, 1))
                # Sample start locations
                warp_x0 = np.random.randint(0, W - warp_patch_size + 1, size=(warp_patch_num, 1, 1))
                warp_y0 = np.random.randint(0, H - warp_patch_size + 1, size=(warp_patch_num, 1, 1))
                warp_xy0 = np.concatenate([warp_x0, warp_y0], axis=-1)
                warp_patch_idx = warp_xy0 + np.stack(
                    np.meshgrid(np.arange(warp_patch_size), np.arange(warp_patch_size), indexing='xy'),
                    axis=-1).reshape(1, -1, 2)
                warp_rays_o = warp_rays_o_tr[warp_idx_img, warp_patch_idx[Ellipsis, 1], warp_patch_idx[Ellipsis, 0]].reshape(-1, 3)
                warp_rays_d = warp_rays_d_tr[warp_idx_img, warp_patch_idx[Ellipsis, 1], warp_patch_idx[Ellipsis, 0]].reshape(-1, 3)
                warp_viewdirs = warp_viewdirs_tr[warp_idx_img, warp_patch_idx[Ellipsis, 1], warp_patch_idx[Ellipsis, 0]].reshape(-1, 3)
                warp_target = warp_rgb_tr[warp_idx_img, warp_patch_idx[Ellipsis, 1], warp_patch_idx[Ellipsis, 0]].reshape(-1, 3)
                warp_gt_depths = warp_depths_tr[warp_idx_img, warp_patch_idx[Ellipsis, 1], warp_patch_idx[Ellipsis, 0]].reshape(-1,warp_patch_size,warp_patch_size)
                warp_select_masks = warp_masks_tr[warp_idx_img, warp_patch_idx[Ellipsis, 1], warp_patch_idx[Ellipsis, 0]].reshape(-1,warp_patch_size,warp_patch_size)
                

        else:
            raise NotImplementedError

        if cfg_train.N_rand_sample:
            n_patches = cfg_train.N_rand_sample // (cfg_train.patch_size ** 2)
            H, W = HW[0]
            # Sample pose(s)
            idx_img = np.random.randint(0, len(rays_o_rd_tr), size=(n_patches, 1))
            #TODO: Sample patch from one image
            # Sample start locations
            x0 = np.random.randint(0, W - cfg_train.patch_size + 1, size=(n_patches, 1, 1))
            y0 = np.random.randint(0, H - cfg_train.patch_size + 1, size=(n_patches, 1, 1))
            xy0 = np.concatenate([x0, y0], axis=-1)
            patch_idx = xy0 + np.stack(
                np.meshgrid(np.arange(cfg_train.patch_size), np.arange(cfg_train.patch_size), indexing='xy'),
                axis=-1).reshape(1, -1, 2)
            
            rays_o_rd_patch = rays_o_rd_tr[idx_img, patch_idx[Ellipsis, 1], patch_idx[Ellipsis, 0]].reshape(-1, 3)
            rays_d_rd_patch = rays_d_rd_tr[idx_img, patch_idx[Ellipsis, 1], patch_idx[Ellipsis, 0]].reshape(-1, 3)
            viewdirs_rd_patch = viewdirs_rd_tr[idx_img, patch_idx[Ellipsis, 1], patch_idx[Ellipsis, 0]].reshape(-1, 3)

        if cfg.data.load2gpu_on_the_fly:
            target = target.to(device)
            rays_o = rays_o.to(device).reshape(-1,3)
            rays_d = rays_d.to(device).reshape(-1,3)
            viewdirs = viewdirs.to(device).reshape(-1,3)

        # volume rendering
        render_result = model(
            rays_o, rays_d, viewdirs,
            global_step=global_step, is_train=True,
            **render_kwargs)
        
        render_rd_patch_result = model(
            rays_o_rd_patch, rays_d_rd_patch, viewdirs_rd_patch,
            global_step=global_step, is_train=True,
            **render_kwargs
            )
        
        if len(extra_data) != 0:
            render_warp_result = model(
                warp_rays_o, warp_rays_d, warp_viewdirs,
                global_step=global_step, is_train=True,
                **render_kwargs
                )
        

        # gradient descent step
        optimizer.zero_grad(set_to_none=True)
        loss = cfg_train.weight_main * F.mse_loss(render_result['rgb_marched'], target)
        psnr = utils.mse2psnr(loss.detach())
        

        if cfg_train.weight_entropy_last > 0:
            pout = render_result['alphainv_last'].clamp(1e-6, 1-1e-6)
            entropy_last_loss = -(pout*torch.log(pout) + (1-pout)*torch.log(1-pout)).mean()
            loss += cfg_train.weight_entropy_last * entropy_last_loss
        if cfg_train.weight_nearclip > 0:
            near_thres = data_dict['near_clip'] / model.scene_radius[0].item()
            near_mask = (render_result['t'] < near_thres)
            density = render_result['raw_density'][near_mask]
            if len(density):
                nearclip_loss = (density - density.detach()).sum()
                loss += cfg_train.weight_nearclip * nearclip_loss
        if cfg_train.weight_distortion > 0:
            n_max = render_result['n_max']
            s = render_result['s']
            w = render_result['weights']
            ray_id = render_result['ray_id']
            loss_distortion = flatten_eff_distloss(w, s, 1/n_max, ray_id)
            loss += cfg_train.weight_distortion * loss_distortion
        if cfg_train.weight_rgbper > 0:
            rgbper = (render_result['raw_rgb'] - target[render_result['ray_id']]).pow(2).sum(-1)
            rgbper_loss = (rgbper * render_result['weights'].detach()).sum() / len(rays_o)
            loss += cfg_train.weight_rgbper * rgbper_loss
        if cfg_train.weight_depths > 0:
            depth_relative_loss=utils.compute_relative_depth_loss(render_result['depth'],gt_depths,patch_num,patch_size)
            loss+=cfg_train.weight_depths* depth_relative_loss
        if cfg_train.weight_entropy_xyz >0:
            xyz_loss=model.xyz_entropy('density')
            loss+=cfg_train.weight_entropy_xyz*xyz_loss
        if cfg_train.weight_tv_depth >0:
            depth = render_rd_patch_result['depth'].reshape(-1,cfg_train.patch_size,cfg_train.patch_size,1)
            depth_tvnorm_loss=utils.compute_tv_norm(depth).mean()
            loss+=cfg_train.weight_tv_depth*depth_tvnorm_loss
        if len(extra_data) != 0:
            if cfg_train.weight_warp_rgb >0:
                warp_rgb_loss = cfg_train.weight_warp_rgb * F.mse_loss(render_warp_result['rgb_marched']*warp_select_masks.reshape(-1,1), warp_target*warp_select_masks.reshape(-1,1))
                # warp_rgb_loss = cfg_train.weight_warp_rgb * F.mse_loss(render_warp_result['rgb_marched'], warp_target)
                loss+=cfg_train.weight_warp_rgb * warp_rgb_loss
            if cfg_train.weight_warp_depths > 0:
                warp_depth_relative_loss=utils.compute_relative_depth_loss(render_result['depth'],warp_gt_depths,warp_patch_num,patch_size,warp_select_masks)
                loss+=cfg_train.weight_warp_depths* warp_depth_relative_loss
        loss.backward()

        if global_step<cfg_train.tv_before and global_step>cfg_train.tv_after and global_step%cfg_train.tv_every==0:
            if cfg_train.weight_tv_density>0:
                model.reliable_based_density_smooth_add_grad(
                    cfg_train.weight_tv_density/len(rays_o), global_step<cfg_train.tv_dense_before)
            if cfg_train.weight_tv_k0>0:
                model.reliable_based_k0_smooth_add_grad(
                    cfg_train.weight_tv_k0/len(rays_o), global_step<cfg_train.tv_dense_before)

        optimizer.step()
        if not math.isinf(psnr.item()):
            psnr_lst.append(psnr.item())

        # update lr
        decay_steps = cfg_train.lrate_decay * 1000
        decay_factor = 0.1 ** (1/decay_steps)
        for i_opt_g, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = param_group['lr'] * decay_factor

        # check log & save
        if global_step%args.i_print==0:
            eps_time = time.time() - time0
            eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
            tqdm.write(f'scene_rep_reconstruction ({stage}): iter {global_step:6d} / '
                       f'Loss: {loss.item():.9f} / PSNR: {np.mean(psnr_lst):5.2f} / '
                       f'Eps: {eps_time_str}')
            psnr_lst = []

        if global_step%args.i_weights==0:
            path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_{global_step:06d}.tar')
            torch.save({
                'global_step': global_step,
                'model_kwargs': model.get_kwargs(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', path)

    
        if global_step+1 in cfg_train.ug_scale:
            with torch.no_grad():
                extra_data={}
                stepsize = cfg.fine_model_and_render.stepsize
                render_viewpoints_kwargs = {
                    'model': model,
                    'ndc': cfg.data.ndc,
                    'render_kwargs': {
                        'near': data_dict['near'],
                        'far': data_dict['far'],
                        'bg': 1 if cfg.data.white_bkgd else 0,
                        'stepsize': stepsize,
                        'inverse_y': cfg.data.inverse_y,
                        'flip_x': cfg.data.flip_x,
                        'flip_y': cfg.data.flip_y,
                        'render_depth': True,
                    },
                }
                if cfg.data['dataset_type'] == 'blender':

                    gen = generater(4)
                    noise_poses = []
                    for pose in np.array(poses[i_train].cpu().float()):
                        noise_poses.append(torch.tensor(pose).float())
                        for delta_degree in cfg.data['delta_degrees']:
                            noise_pose=gen.general_pose(pose,0,delta_degree)
                            noise_poses.append(noise_pose)
                            noise_pose=gen.general_pose(pose,delta_degree,-delta_degree)
                            noise_poses.append(noise_pose)
 
            
            
                    noise_rgbs, noise_depths, bgmaps = render_noise_view(
                        render_poses=noise_poses,
                        HW=data_dict['HW'][0],
                        Ks=data_dict['Ks'][0],
                        **render_viewpoints_kwargs)
                    noise_rgbs=np.split(noise_rgbs,noise_rgbs.shape[0],axis=0)
                    noise_depths=np.split(noise_depths,noise_depths.shape[0],axis=0)
                    bgmaps=np.split(bgmaps>0.999,bgmaps.shape[0],axis=0)


                        
                    warping_rgb_all,masks,warp_bgmaps,poses_noise=[],[],[],[]
                    for seen_index,index in enumerate(range(0,len(noise_rgbs),2*len(cfg.data['delta_degrees'])+1)):
                        seen_proj=noise_poses[index]
                        unseen_proj=noise_poses[index+1:index+2*len(cfg.data['delta_degrees'])+1]
                        seen_rgb = np.array(images[i_train[seen_index]].cpu().unsqueeze(0))
                        seen_depth = noise_depths[index]
                        unseen_depth = noise_depths[index+1:index+2*len(cfg.data['delta_degrees'])+1]
                        warping_out_rgb, mask = nerf_warping(seen_rgb,seen_proj,unseen_proj,seen_depth,unseen_depth,data_dict['Ks'][0],0.5)
                        warping_rgb_all +=  warping_out_rgb
                        masks += mask
                        warp_bgmaps+=bgmaps[index+1:index+2*len(cfg.data['delta_degrees'])+1]
                        poses_noise+=noise_poses[index+1:index+2*len(cfg.data['delta_degrees'])+1]
                        
                elif cfg.data['dataset_type'] == 'llff':
                    noise_poses=[]
                    for index,pose in enumerate(np.array(poses[i_train].cpu().float())):
                        noise_poses.append(torch.tensor(pose).float())
                        noise_poses.append(torch.tensor(average_extrinsic_matrices(np.array(poses[index].cpu().float()),np.array(poses[(index+1)%3].cpu().float()))).float())
                        noise_poses.append(torch.tensor(average_extrinsic_matrices(np.array(poses[index].cpu().float()),np.array(poses[(index+2)%3].cpu().float()))).float())
                    
                    noise_rgbs, noise_depths, bgmaps = render_noise_view(
                        render_poses=noise_poses,
                        HW=data_dict['HW'][0],
                        Ks=data_dict['Ks'][0],
                        **render_viewpoints_kwargs)
                    noise_rgbs=np.split(noise_rgbs,noise_rgbs.shape[0],axis=0)
                    noise_depths=np.split(noise_depths,noise_depths.shape[0],axis=0)

                    
                    for i in range(len(noise_poses)):
                        noise_poses[i]=torch.cat((noise_poses[i],torch.tensor([[0,0,0,1]])),dim=0)
                        
                        
                    warping_rgb_all,masks,warp_bgmaps,poses_noise=[],[],[],[]
                    for seen_index,index in enumerate(range(0,len(data_dict['i_train'])*3,3)):
                        seen_proj=noise_poses[index]
                        unseen_proj=noise_poses[index+1:index+3]
                        seen_rgb = np.array(images[i_train[seen_index]].cpu().unsqueeze(0))
                        seen_depth = noise_depths[index]
                        unseen_depth = noise_depths[index+1:index+3]
                        warping_out_rgb, mask = nerf_warping(seen_rgb,seen_proj,unseen_proj,seen_depth,unseen_depth,data_dict['Ks'][0],0.5)
                        warping_rgb_all +=  warping_out_rgb
                        masks += mask
                        warp_bgmaps+=bgmaps[index+1:index+3]
                        poses_noise+=noise_poses[index+1:index+3]



                disguise_depths = depth_model.get_relative_depth(warping_rgb_all)
                    
                    
                warp_rgbs = torch.stack(warping_rgb_all,dim=0).squeeze().permute(0,2,3,1)  # [1,3,800,800]
                warp_masks = torch.stack(masks,dim=0)           # [800,800]
                warp_bgmaps = torch.tensor(np.stack(warp_bgmaps,axis=0).squeeze())    # [1,800,800,1] 
                warp_poses = torch.stack(poses_noise,dim=0)     # [4,4]
                warp_depths = torch.tensor(disguise_depths)                  # [16,800,800]

                # warp_depths[warp_bgmaps>0]=warp_depths.min()

                extra_data = {
                    'warp_rgbs':warp_rgbs,
                    'warp_masks':warp_masks,    
                    'warp_poses':warp_poses,
                    'warp_depths':warp_depths
                }
                
                warp_rgb_tr = extra_data['warp_rgbs']
                warp_depths_tr = extra_data['warp_depths']
                warp_masks_tr = extra_data['warp_masks']
                warp_poses_tr = extra_data['warp_poses']
                warp_rays_o_tr, warp_rays_d_tr, warp_viewdirs_tr=[],[],[]

                for c2w in warp_poses_tr:
                    warp_rays_o, warp_rays_d, warp_viewdirs = dvgo.get_rays_of_a_view(
                            HW[0][0], HW[0][1], Ks[0], c2w, cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                            flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,)

                    warp_rays_o_tr.append(warp_rays_o)
                    warp_rays_d_tr.append(warp_rays_d)
                    warp_viewdirs_tr.append(warp_viewdirs)
                
                warp_rays_o_tr=torch.stack(warp_rays_o_tr,dim=0)
                warp_rays_d_tr=torch.stack(warp_rays_d_tr,dim=0)
                warp_viewdirs_tr=torch.stack(warp_viewdirs_tr,dim=0)
                 
            if cfg_train.set_lr:
                cnt = model.reliable_count_views(
                        rays_o_tr=warp_rays_o_tr, rays_d_tr=warp_rays_d_tr,masks=warp_masks_tr, imsz=[1]*warp_rays_o_tr.shape[0],near=near, far=far,
                        stepsize=cfg_model.stepsize, downrate=cfg_train.pervoxel_lr_downrate,
                        irregular_shape=data_dict['irregular_shape'])
                optimizer.update_pervoxel_lr(cnt)
                model.set_reliable_area(cnt)
                    

    if global_step != -1:
        torch.save({
            'global_step': global_step,
            'model_kwargs': model.get_kwargs(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, last_ckpt_path)
        print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', last_ckpt_path)


def train(args, cfg, data_dict):

    # init
    print('train: start')
    eps_time = time.time()
    os.makedirs(os.path.join(cfg.basedir, cfg.expname), exist_ok=True)
    with open(os.path.join(cfg.basedir, cfg.expname, 'args.txt'), 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    cfg.dump(os.path.join(cfg.basedir, cfg.expname, 'config.py'))
    initial_dpt_model=DPT_model(args.dpt_model_path)
    # coarse geometry searching (only works for inward bounded scenes)
    eps_coarse = time.time()
    xyz_min_coarse, xyz_max_coarse = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, **data_dict)
    if cfg.coarse_train.N_iters > 0:
        scene_rep_reconstruction(
                args=args, cfg=cfg,
                cfg_model=cfg.coarse_model_and_render, cfg_train=cfg.coarse_train,
                xyz_min=xyz_min_coarse, xyz_max=xyz_max_coarse,
                data_dict=data_dict, depth_model=initial_dpt_model, stage='coarse')
        eps_coarse = time.time() - eps_coarse
        eps_time_str = f'{eps_coarse//3600:02.0f}:{eps_coarse//60%60:02.0f}:{eps_coarse%60:02.0f}'
        print('train: coarse geometry searching in', eps_time_str)
        coarse_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'coarse_last.tar')
    else:
        print('train: skip coarse geometry searching')
        coarse_ckpt_path = None

    # fine detail reconstruction
    eps_fine = time.time()
    if cfg.coarse_train.N_iters == 0:
        xyz_min_fine, xyz_max_fine = xyz_min_coarse.clone(), xyz_max_coarse.clone()
    else:
        xyz_min_fine, xyz_max_fine = compute_bbox_by_coarse_geo(
                model_class=dvgo.DirectVoxGO, model_path=coarse_ckpt_path,
                thres=cfg.fine_model_and_render.bbox_thres)
    scene_rep_reconstruction(
            args=args, cfg=cfg,
            cfg_model=cfg.fine_model_and_render, cfg_train=cfg.fine_train,
            xyz_min=xyz_min_fine, xyz_max=xyz_max_fine,
            data_dict=data_dict, dpt_model=initial_dpt_model, stage='fine',
            coarse_ckpt_path=coarse_ckpt_path)
    eps_fine = time.time() - eps_fine
    eps_time_str = f'{eps_fine//3600:02.0f}:{eps_fine//60%60:02.0f}:{eps_fine%60:02.0f}'
    print('train: fine detail reconstruction in', eps_time_str)

    eps_time = time.time() - eps_time
    eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
    print('train: finish (eps time', eps_time_str, ')')


if __name__=='__main__':

    # load setup
    parser = config_parser()

    args = parser.parse_args()
    f=open(args.txtfile+'.txt', "a")
    cfg = mmengine.Config.fromfile(args.config)

    # init enviroment
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    seed_everything()

    # load images / poses / camera settings / data split
    data_dict = load_everything(args=args, cfg=cfg)

    # export scene bbox and camera poses in 3d for debugging and visualization
    if args.export_bbox_and_cams_only:
        print('Export bbox and cameras...')
        xyz_min, xyz_max = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, **data_dict)
        poses, HW, Ks, i_train = data_dict['poses'], data_dict['HW'], data_dict['Ks'], data_dict['i_train']
        near, far = data_dict['near'], data_dict['far']
        if data_dict['near_clip'] is not None:
            near = data_dict['near_clip']
        cam_lst = []
        for c2w, (H, W), K in zip(poses[i_train], HW[i_train], Ks[i_train]):
            rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                    H, W, K, c2w, cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                    flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,)
            cam_o = rays_o[0,0].cpu().numpy()
            cam_d = rays_d[[0,0,-1,-1],[0,-1,0,-1]].cpu().numpy()
            cam_lst.append(np.array([cam_o, *(cam_o+cam_d*max(near, far*0.05))]))
        np.savez_compressed(args.export_bbox_and_cams_only,
            xyz_min=xyz_min.cpu().numpy(), xyz_max=xyz_max.cpu().numpy(),
            cam_lst=np.array(cam_lst))
        print('done')
        sys.exit()

    if args.export_coarse_only:
        print('Export coarse visualization...')
        with torch.no_grad():
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'coarse_last.tar')
            model = utils.load_model(dvgo.DirectVoxGO, ckpt_path).to(device)
            alpha = model.activate_density(model.density.get_dense_grid()).squeeze().cpu().numpy()
            rgb = torch.sigmoid(model.k0.get_dense_grid()).squeeze().permute(1,2,3,0).cpu().numpy()
        np.savez_compressed(args.export_coarse_only, alpha=alpha, rgb=rgb)
        print('done')
        sys.exit()

    # train
    if not args.render_only:
        train(args, cfg, data_dict)

    # load model for rendring
    if args.render_test or args.render_train or args.render_video:
        if args.ft_path:
            ckpt_path = args.ft_path
        else:
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_last.tar')
        ckpt_name = ckpt_path.split('/')[-1][:-4]
        if cfg.data.ndc:
            model_class = dmpigo.DirectMPIGO
        elif cfg.data.unbounded_inward:
            model_class = dcvgo.DirectContractedVoxGO
        else:
            model_class = dvgo.DirectVoxGO
        model = utils.load_model(model_class, ckpt_path).to(device)
        stepsize = cfg.fine_model_and_render.stepsize
        render_viewpoints_kwargs = {
            'model': model,
            'ndc': cfg.data.ndc,
            'render_kwargs': {
                'near': data_dict['near'],
                'far': data_dict['far'],
                'bg': 1 if cfg.data.white_bkgd else 0,
                'stepsize': stepsize,
                'inverse_y': cfg.data.inverse_y,
                'flip_x': cfg.data.flip_x,
                'flip_y': cfg.data.flip_y,
                'render_depth': True,
            },
        }

    # render trainset and eval
    if args.render_train:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_train_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        print('All results are dumped into', testsavedir)
        rgbs, depths, bgmaps = render_viewpoints(
                render_poses=data_dict['poses'][data_dict['i_train']],
                HW=data_dict['HW'][data_dict['i_train']],
                Ks=data_dict['Ks'][data_dict['i_train']],
                gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_train']],
                savedir=testsavedir, dump_images=args.dump_images,
                eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.depth.mp4'), utils.to8b(1 - depths / np.max(depths)), fps=30, quality=8)

    # render testset and eval
    if args.render_test:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_test_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        print('All results are dumped into', testsavedir)
        rgbs, depths, bgmaps = render_viewpoints(
                render_poses=data_dict['poses'][data_dict['i_test']],
                HW=data_dict['HW'][data_dict['i_test']],
                Ks=data_dict['Ks'][data_dict['i_test']],
                gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_test']],
                savedir=testsavedir, dump_images=args.dump_images,
                eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                **render_viewpoints_kwargs)
        for i,rgb in enumerate(rgbs):
            imageio.imwrite(os.path.join(testsavedir+'{:03d}.png'.format(i)), utils.to8b(rgb))
            
            
        depths_vis = depths
        dmin, dmax = np.percentile(depths_vis, q=[1, 99])
        depth_vis = plt.get_cmap('rainbow')(1 - np.clip((depths_vis - dmin) / (dmax - dmin), 0, 1)).squeeze()[..., :3]
        # depth_vis[(bgmaps > 0.5).squeeze()]=[1,1,1]
        rgb8=utils.to8b(rgbs)
        depth8=utils.to8b(depth_vis)
        for i,img in enumerate(rgb8):
            imageio.imwrite(os.path.join(testsavedir, '{:03d}.png'.format(i)), img)
            imageio.imwrite(os.path.join(testsavedir, '{:03d}_depth.png'.format(i)), depth8[i].squeeze())

    # render video
    if args.render_video:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_video_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        print('All results are dumped into', testsavedir)
        rgbs, depths, bgmaps = render_viewpoints(
                render_poses=data_dict['render_poses'],
                HW=data_dict['HW'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                Ks=data_dict['Ks'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                render_factor=args.render_video_factor,
                render_video_flipy=args.render_video_flipy,
                render_video_rot90=args.render_video_rot90,
                savedir=testsavedir, dump_images=args.dump_images,
                **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        import matplotlib.pyplot as plt
        depths_vis = depths * (1-bgmaps) + bgmaps
        dmin, dmax = np.percentile(depths_vis[bgmaps < 0.1], q=[5, 95])
        depth_vis = plt.get_cmap('rainbow')(1 - np.clip((depths_vis - dmin) / (dmax - dmin), 0, 1)).squeeze()[..., :3]
        imageio.mimwrite(os.path.join(testsavedir, 'video.depth.mp4'), utils.to8b(depth_vis), fps=30, quality=8)

    print('Done')

