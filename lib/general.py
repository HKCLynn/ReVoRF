import torch
import torch.nn.functional as F
import numpy as np
from math import pi

def homo_warping(src_fea, src_proj, ref_proj, ref_depth, patch_start, patch_size):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # ref_depth: [B, H, W]
    # out: [B, C, H, W]
    '''
    将seen的视图转换到unseen的视图中，需要输入unseen视图的Depth map和seen视图，以及二者的project matrix
    '''
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    height, width = ref_depth.shape[1], ref_depth.shape[2]
    total_height, total_width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(patch_start[0], patch_start[0]+patch_size[0], dtype=torch.float32, device=src_fea.device),
                               torch.arange(patch_start[1], patch_start[1]+patch_size[1], dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        # 齐次化
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz*(ref_depth.unsqueeze(1).contiguous().view(batch,1,-1))   #[B, 3, H*W]
        
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1)  # [B, 3, H*W]
        proj_xy = proj_xyz[:, :2, :] / proj_xyz[:, 2:3, :]  # [B, 2, H*W]
        proj_x_normalized = proj_xy[:, 0, :] / ((total_width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :] / ((total_height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=2)  # [B, H*W, 2]
        grid = proj_xy

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, height, width, 2), mode='bilinear',
                                   padding_mode='zeros',align_corners=True)
    warped_src_fea = warped_src_fea.view(batch, channels, height, width)

    return warped_src_fea

def warping_out(seen_fea, seen_proj, unseen_proj, unseen_depth, K, patch_start=None, patch_size=None):
    """
    src_fea是seen  [batch,H,W,3]
    ref_fea是unseen [batch,H,W,3]
    
    seen_fea [batch,H,W,3]
    seen_proj [batch,4,4]
    unseen_proj [batch,4,4]
    unseen_depth [batch,H,W]
    patch_start [batch,2]
    """
    warping_out=[]
    for src_fea,src_proj,ref_proj,ref_depth,patch_st in zip(seen_fea ,seen_proj, unseen_proj, unseen_depth, patch_start):
        src_proj=torch.inverse(src_proj)
        ref_proj=torch.inverse(ref_proj)


        src_proj[:3,:]=K@src_proj[:3,:]
        ref_proj[:3,:]=K@ref_proj[:3,:]
        src_proj=src_proj.unsqueeze(0)
        ref_proj=ref_proj.unsqueeze(0)
        
        src_fea=src_fea.unsqueeze(0)
        ref_depth=ref_depth.unsqueeze(0)
        warping_out.append(homo_warping(src_fea, src_proj, ref_proj, ref_depth, patch_st, patch_size))
    return torch.cat(warping_out,dim=0)

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

def t_matrix_to_spherical_coords(t_matrix):
    # 提取平移向量
    translation = t_matrix[:3, 3]

    # 计算极角（theta）
    theta = np.arccos(translation[2] / np.linalg.norm(translation))

    # 计算方位角（phi）
    phi = np.arctan2(translation[1], translation[0])

    return theta*180/pi, phi*180/pi

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(theta/180.*np.pi) @ c2w
    c2w = rot_theta(phi/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


class generater():
    def __init__ (self,radius):
        self.radius=radius
        
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

    def t_matrix_to_spherical_coords(self,t_matrix):
        # 提取平移向量
        translation = t_matrix[:3, 3]

        # 计算极角（theta）
        theta = np.arccos(translation[2] / np.linalg.norm(translation))

        # 计算方位角（phi）
        phi = np.arctan2(translation[1], translation[0])

        return theta*180/pi, phi*180/pi

    def pose_spherical(self, theta, phi, radius):
        c2w = trans_t(radius)
        c2w = rot_phi(theta/180.*np.pi) @ c2w
        c2w = rot_theta(phi/180.*np.pi) @ c2w
        c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
        return c2w

    def general_pose(self, matrix, delta_theta=0, delta_phi=0):
        theta,phi = t_matrix_to_spherical_coords(matrix)
        theta-=90
        phi-=90
        phi=-phi
        theta+=delta_theta
        phi+=delta_phi
        return pose_spherical(theta,phi,self.radius)

from scipy.spatial.transform import Rotation as R

def average_extrinsic_matrices(matrix1, matrix2):
    """
    计算两个3x4外参矩阵的平均矩阵。

    :param matrix1: 第一个外参矩阵。
    :param matrix2: 第二个外参矩阵。
    :return: 平均外参矩阵。
    """
    # 提取旋转矩阵和平移向量
    rotation1, translation1 = matrix1[:, :3], matrix1[:, 3]
    rotation2, translation2 = matrix2[:, :3], matrix2[:, 3]

    # 将旋转矩阵转换为四元数
    quaternion1 = R.from_matrix(rotation1).as_quat()
    quaternion2 = R.from_matrix(rotation2).as_quat()

    # 计算四元数的平均值
    average_quaternion = (quaternion1 + quaternion2) / 2

    # 将平均四元数转换回旋转矩阵
    average_rotation = R.from_quat(average_quaternion).as_matrix()

    # 计算平均平移向量
    average_translation = (translation1 + translation2) / 2

    # 构造平均外参矩阵
    average_matrix = np.hstack((average_rotation, average_translation.reshape(-1, 1)))

    return average_matrix


if __name__ == "__main__":
    x=np.array(([[ 1.0000e+00,  6.1232e-17, -1.0606e-16, -4.2423e-16],
                [-1.2246e-16,  5.0000e-01, -8.6603e-01, -3.4641e+00],
                [ 0.0000e+00,  8.6603e-01,  5.0000e-01,  2.0000e+00],
                [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]))
    
    # x=np.array([[ 1.0000e+00,  1.2246e-16, -7.4988e-33, -2.9995e-32],
    #             [-1.2246e-16,  1.0000e+00, -6.1232e-17, -2.4493e-16],
    #             [ 0.0000e+00,  6.1232e-17,  1.0000e+00,  4.0000e+00],
    #             [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])
    
    y=np.array([[-9.3054223e-01,  1.1707554e-01, -3.4696460e-01, -1.3986591e+00],
                [-3.6618456e-01, -2.9751042e-01,  8.8170075e-01,  3.5542498e+00],
                [ 7.4505806e-09,  9.4751304e-01,  3.1971723e-01,  1.2888215e+00],
                [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]])
    
    y_s = np.array([[-0.9305, -0.1171,  0.3470,  1.3879],
                    [ 0.3662, -0.2975,  0.8817,  3.5268],
                    [ 0.0000,  0.9475,  0.3197,  1.2789],
                    [ 0.0000,  0.0000,  0.0000,  1.0000]])
    # -y轴是-90度，那么Y轴就是90度，那么x轴是0度，所以相差一个90度
    gen=generater(4)
    theta,phi = t_matrix_to_spherical_coords(y)
    print(theta,phi)
    theta,phi = t_matrix_to_spherical_coords(y_s)
    print(theta,phi)
    print(gen.general_pose(y_s))
    # print(theta,phi)
    
    # # y轴是0度
    # print(pose_spherical(theta-90,phi-90,4))
