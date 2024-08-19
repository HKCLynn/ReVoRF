import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np

from torchvision.transforms import Compose

from .dpt.models import DPTDepthModel
from .dpt.transforms import Resize, NormalizeImage, PrepareForNet
import math
import cv2

class DPT_model():
    def __init__(self,model_path):
        self.net_w = self.net_h = 384
        self.model = DPTDepthModel(
            path=model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        self.normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.transform = Compose(
        [
            Resize(
                self.net_w,
                self.net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            self.normalization,
            PrepareForNet(),
        ])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model=self.model.cuda()
        self.model = self.model.to(memory_format=torch.channels_last)
        self.model = self.model.half()
        self.model.eval()
        
    def out_depth(self, depth, max=1):
        """Write depth map to pfm and png file.

        Args:
            path (str): filepath without extension
            depth (array): depth
        """
        depth_min = np.min(depth,axis=(1,2),keepdims=True)
        depth_max = np.max(depth,axis=(1,2),keepdims=True)

        max_val = max


        out = max_val/2+(max_val/2) * (depth - depth_min) / (depth_max - depth_min)

        return out

    def get_relative_depth(self,images):
        # images list [800,800,3] 0-1
    
        pre_images=[]
        for image in images:
            pre_images.append(self.transform({"image": np.array(image.cpu().squeeze().permute(1,2,0))})["image"])
        pre_images=torch.tensor(np.stack(pre_images,axis=0))
        with torch.no_grad():
            sample = pre_images.to(self.device)
            
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()

            prediction = self.model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=images[0].shape[2:4],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

        return self.out_depth(prediction,20)


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun

class PartialConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, False)
        self.input_conv.apply(weights_init('kaiming'))
        self.slide_winsize = in_channels * kernel_size * kernel_size

        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, mask):
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)
        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(
                output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask == 0

        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

        output_pre = ((output - output_bias) * self.slide_winsize) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)

        return output, new_mask

class PCBActiv(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, sample='none-3', activ='relu',
                 conv_bias=False):
        super().__init__()
        if sample == 'down-5':
            self.conv = PartialConv(in_ch, out_ch, 5, 2, 2, bias=conv_bias)
        elif sample == 'down-7':
            self.conv = PartialConv(in_ch, out_ch, 7, 2, 3, bias=conv_bias)
        elif sample == 'down-3':
            self.conv = PartialConv(in_ch, out_ch, 3, 2, 1, bias=conv_bias)
        else:
            self.conv = PartialConv(in_ch, out_ch, 3, 1, 1, bias=conv_bias)

        if bn:
            self.bn = nn.BatchNorm2d(out_ch)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input, input_mask):
        h, h_mask = self.conv(input, input_mask)
        if hasattr(self, 'bn'):
            h = self.bn(h)
        if hasattr(self, 'activation'):
            h = self.activation(h)
        return h, h_mask

class Inpaint_Depth_Net(nn.Module):
    def __init__(self, layer_size=7, upsampling_mode='nearest'):
        super().__init__()
        in_channels = 4
        out_channels = 1
        self.freeze_enc_bn = False
        self.upsampling_mode = upsampling_mode
        self.layer_size = layer_size
        self.enc_1 = PCBActiv(in_channels, 64, bn=False, sample='down-7', conv_bias=True)
        self.enc_2 = PCBActiv(64, 128, sample='down-5', conv_bias=True)
        self.enc_3 = PCBActiv(128, 256, sample='down-5')
        self.enc_4 = PCBActiv(256, 512, sample='down-3')
        for i in range(4, self.layer_size):
            name = 'enc_{:d}'.format(i + 1)
            setattr(self, name, PCBActiv(512, 512, sample='down-3'))

        for i in range(4, self.layer_size):
            name = 'dec_{:d}'.format(i + 1)
            setattr(self, name, PCBActiv(512 + 512, 512, activ='leaky'))
        self.dec_4 = PCBActiv(512 + 256, 256, activ='leaky')
        self.dec_3 = PCBActiv(256 + 128, 128, activ='leaky')
        self.dec_2 = PCBActiv(128 + 64, 64, activ='leaky')
        self.dec_1 = PCBActiv(64 + in_channels, out_channels,
                              bn=False, activ=None, conv_bias=True)
    def add_border(self, input, mask_flag, PCONV=True):
        with torch.no_grad():
            h = input.shape[-2]
            w = input.shape[-1]
            require_len_unit = 2 ** self.layer_size
            residual_h = int(np.ceil(h / float(require_len_unit)) * require_len_unit - h) # + 2*require_len_unit
            residual_w = int(np.ceil(w / float(require_len_unit)) * require_len_unit - w) # + 2*require_len_unit
            enlarge_input = torch.zeros((input.shape[0], input.shape[1], h + residual_h, w + residual_w)).to(input.device)
            if mask_flag:
                if PCONV is False:
                    enlarge_input += 1.0
                enlarge_input = enlarge_input.clamp(0.0, 1.0)
            else:
                enlarge_input[:, 2, ...] = 0.0
            anchor_h = residual_h//2
            anchor_w = residual_w//2
            enlarge_input[..., anchor_h:anchor_h+h, anchor_w:anchor_w+w] = input

        return enlarge_input, [anchor_h, anchor_h+h, anchor_w, anchor_w+w]

    def forward_3P(self, mask, context, depth, edge, unit_length=128, cuda=None):
        with torch.no_grad():
            input = torch.cat((depth, edge, context, mask), dim=1)
            n, c, h, w = input.shape
            residual_h = int(np.ceil(h / float(unit_length)) * unit_length - h)
            residual_w = int(np.ceil(w / float(unit_length)) * unit_length - w)
            anchor_h = residual_h//2
            anchor_w = residual_w//2
            enlarge_input = torch.zeros((n, c, h + residual_h, w + residual_w)).to(cuda)
            enlarge_input[..., anchor_h:anchor_h+h, anchor_w:anchor_w+w] = input
            # enlarge_input[:, 3] = 1. - enlarge_input[:, 3]
            depth_output = self.forward(enlarge_input)
            depth_output = depth_output[..., anchor_h:anchor_h+h, anchor_w:anchor_w+w]
            # import pdb; pdb.set_trace()

        return depth_output

    def forward(self, input_feat, refine_border=False, sample=False, PCONV=True):
        input = input_feat
        input_mask = (input_feat[:, -2:-1] + input_feat[:, -1:]).clamp(0, 1).repeat(1, input.shape[1], 1, 1)

        vis_input = input.cpu().data.numpy()
        vis_input_mask = input_mask.cpu().data.numpy()
        H, W = input.shape[-2:]
        if refine_border is True:
            input, anchor = self.add_border(input, mask_flag=False)
            input_mask, anchor = self.add_border(input_mask, mask_flag=True, PCONV=PCONV)
        h_dict = {}  # for the output of enc_N
        h_mask_dict = {}  # for the output of enc_N
        h_dict['h_0'], h_mask_dict['h_0'] = input, input_mask

        h_key_prev = 'h_0'
        for i in range(1, self.layer_size + 1):
            l_key = 'enc_{:d}'.format(i)
            h_key = 'h_{:d}'.format(i)
            h_dict[h_key], h_mask_dict[h_key] = getattr(self, l_key)(
                h_dict[h_key_prev], h_mask_dict[h_key_prev])
            h_key_prev = h_key

        h_key = 'h_{:d}'.format(self.layer_size)
        h, h_mask = h_dict[h_key], h_mask_dict[h_key]

        for i in range(self.layer_size, 0, -1):
            enc_h_key = 'h_{:d}'.format(i - 1)
            dec_l_key = 'dec_{:d}'.format(i)

            h = F.interpolate(h, scale_factor=2, mode=self.upsampling_mode)
            h_mask = F.interpolate(h_mask, scale_factor=2, mode='nearest')

            h = torch.cat([h, h_dict[enc_h_key]], dim=1)
            h_mask = torch.cat([h_mask, h_mask_dict[enc_h_key]], dim=1)
            h, h_mask = getattr(self, dec_l_key)(h, h_mask)
        output = h
        if refine_border is True:
            h_mask = h_mask[..., anchor[0]:anchor[1], anchor[2]:anchor[3]]
            output = output[..., anchor[0]:anchor[1], anchor[2]:anchor[3]]

        return output

class inpaint_model():
    def __init__(self,model_path = None):
        self.model=Inpaint_Depth_Net()
        depth_feat_weight = torch.load(model_path, map_location=torch.device('cuda'))
        self.model.load_state_dict(depth_feat_weight, strict=True)
        self.model=self.model.to('cuda')
        self.model.eval()

    def get_inpainting_depth(self,real_depth,mask,bgmap=None):
        
        mask = torch.stack(mask,dim=0).unsqueeze(1)
        bgmap = torch.tensor(np.stack(bgmap,axis=0).squeeze()).unsqueeze(1)
        real_depth = torch.tensor(real_depth).unsqueeze(1)
        
        # if bgmap==None:
        context = mask
        fix_mask = ~mask
        # else:
        #     context = bgmap & mask
        #     fix_mask = bgmap & (~mask)
            
        import imageio
        # imageio.imwrite('./log.png',np.array((x.squeeze().cpu())*255).astype(np.uint8))
        
        log_depth = torch.log(real_depth+1e-8)
        log_depth[fix_mask>0]=0
        log_depth_context = log_depth.masked_fill(~context, 0)
        input_mean_depth = torch.mean(log_depth_context,dim=(1,2,3), keepdim=True)
        input_zero_mean_depth = ((log_depth - input_mean_depth) * context)
        output_depth=self.model.forward_3P(fix_mask,
                                            context,
                                            input_zero_mean_depth,
                                            torch.zeros_like(input_zero_mean_depth),
                                            unit_length=192,
                                            cuda='cuda')
        output_depth = torch.exp(output_depth + input_mean_depth) * fix_mask + real_depth * ~fix_mask

        output_depth=torch.clamp(output_depth,real_depth.min(),real_depth.max())
        output_depth[torch.isnan(output_depth)]=real_depth.max().float()
        output_depth= output_depth.data.cpu().numpy().squeeze()

        return output_depth