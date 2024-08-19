import torch.nn.functional as F
import torch
import numpy as np

def get_points_world(H, W, K, depth, c2w, inverse_y, flip_x, flip_y, mode='center'):
    i, j = torch.meshgrid(
        torch.linspace(0, W-1, W, device=depth.device),
        torch.linspace(0, H-1, H, device=depth.device))  # pytorch's meshgrid has indexing='ij'
    i = i.t().float()
    j = j.t().float()
    if mode == 'lefttop':
        pass
    elif mode == 'center':
        i, j = i+0.5, j+0.5
    elif mode == 'random':
        i = i+torch.rand_like(i)
        j = j+torch.rand_like(j)
    else:
        raise NotImplementedError

    if flip_x:
        i = i.flip((1,))
    if flip_y:
        j = j.flip((0,))
    if inverse_y:
        dirs = torch.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], torch.ones_like(i)], -1)
    else:
        dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,3].expand(rays_d.shape)
    points_world = rays_o+rays_d*depth.squeeze().unsqueeze(-1)
    return points_world

def get_ref_rays(w2c_ref, intrinsic_ref, point_samples, img, point_samples_unseen,threshold=0):
    '''
        point_samples [N_rays N_sample 3]
    '''

    device = img.device

    B, N_rays, N_samples = point_samples.shape[:3] # [B ray_samples N_samples 3]
    point_samples_seen = point_samples.clone()
    point_samples = point_samples.reshape(B, -1, 3)
    

    N, C, H, W = img.shape
    inv_scale = torch.tensor([W-1, H-1]).to(device)

    # wrap to ref view
    if w2c_ref is not None:
        # R = w2c_ref[:3, :3]  # (3, 3)
        # T = w2c_ref[:3, 3:]  # (3, 1)
        # point_samples = torch.matmul(point_samples, R.t()) + T.reshape(1,3)
        R = w2c_ref[:, :3, :3]  # (B, 3, 3)
        T = w2c_ref[:, :3, 3:]  # (B, 3, 1)
        transform = torch.FloatTensor([[1,0,0],[0,-1,0], [0,0,-1]])[None].cuda()
        point_samples = (point_samples @ R.permute(0,2,1) + T.reshape(B, 1, 3)) @ transform # [B, ray_samples*N_samples, 3] [5, 131072, 3]

    if intrinsic_ref is not None:
        # using projection
        # point_samples_pixel = point_samples @ intrinsic_ref.t()
        point_samples_pixel = point_samples @ intrinsic_ref.permute(0,2,1) # [B, ray_samples*N_samples, 3]

        point_samples_pixel_x = (point_samples_pixel[:, :, 0] / point_samples_pixel[:, :, 2] + 0.0).round().detach()  # [B, ray_samples*N_samples]
        point_samples_pixel_y = (point_samples_pixel[:, :, 1] / point_samples_pixel[:, :, 2] + 0.0).round().detach()  # [B, ray_samples*N_samples]


    point_samples_pixel_y=torch.clamp(point_samples_pixel_y, 0, H-1).clone()
    point_samples_pixel_x=torch.clamp(point_samples_pixel_x, 0, W-1).clone()
    
    point_samples_seen=point_samples_seen.reshape(H,W,3)
    point_samples_unseen=point_samples_unseen.reshape(H,W,3)
    
    point_warp_samples = torch.ones_like(point_samples_unseen)*-100
    
    point_warp_samples[point_samples_pixel_y.type(torch.cuda.LongTensor),point_samples_pixel_x.type(torch.cuda.LongTensor),:]=point_samples_seen.reshape(point_samples_seen.shape[0]*point_samples_seen.shape[1],point_samples_seen.shape[2])
    
    delta_distance = torch.norm(point_warp_samples-point_samples_unseen, p=2, dim=-1)
    mask = (delta_distance<threshold).reshape(H,W)
    
    rgb_ref=torch.ones_like(img)
    rgb_ref[:, :, point_samples_pixel_y.type(torch.cuda.LongTensor), point_samples_pixel_x.type(torch.cuda.LongTensor)]=img.reshape(img.shape[0],img.shape[1],1,img.shape[2]*img.shape[3])


    return rgb_ref,mask

def nerf_warping(seen_fea, seen_proj, unseen_projs, seen_depth,unseen_depths, K, threshold=0):
    """
    src_fea是seen  [H,W,3]
    ref_fea是unseen [batch,H,W,3]
    
    seen_fea [H,W,3]
    seen_proj [4,4]
    unseen_proj [batch,4,4]
    seen_depth [H,W]
    """
    warping_out_rgb=[]
    masks=[]
    seen_fea,seen_proj,seen_depth,unseen_depths,K=torch.tensor(seen_fea).float(),torch.tensor(seen_proj).float(),torch.tensor(seen_depth).float(),torch.tensor(unseen_depths).float(),torch.tensor(K).float()
    for ref_proj,unseen_depth in zip(unseen_projs,unseen_depths):
        ref_proj = torch.tensor(ref_proj)
        H,W=seen_fea.shape[1],seen_fea.shape[2]
        points_world = get_points_world(H,W,K,seen_depth,seen_proj,False,False,False).reshape(-1,3)
        points_world_unseen = get_points_world(H,W,K,unseen_depth,ref_proj,False,False,False).reshape(-1,3)
        warp_rgb,mask = get_ref_rays(ref_proj.inverse().unsqueeze(0),K.unsqueeze(0),points_world[None, :,None, :],seen_fea.permute(0,3,1,2),points_world_unseen[None, :,None, :],threshold)
        warping_out_rgb.append(warp_rgb)
        masks.append(mask)
    return warping_out_rgb,masks


        