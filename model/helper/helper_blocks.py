## code from: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import random
import cv2
from torchpq.clustering import MultiKMeans
# from .swin_transformer import BasicLayer,WindowAttention
# from .swin_transformer import *
# from .Vit_helper import PatchUnembed,VisionTransformer,Attention,Block,PatchEmbed_stride
from .custom_trans_v1 import *

# __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
#            'resnet152', 'ResNet34P','ResNet50S','ResNet50P','ResNet101P']
#
# resnet18_dir = '/local/sda4/yqian3/RoadNets/resnet_model/resnet18-5c106cde.pth'
# resnet34_dir = '/local/sda4/yqian3/RoadNets/resnet_model/resnet34-333f7ec4.pth'
# resnet50_dir = '/local/sda4/yqian3/RoadNets/resnet_model/resnet50-19c8e357.pth'
# resnet101_dir = '/local/sda4/yqian3/RoadNets/resnet_model/resnet101-5d3b4d8f.pth'
#
# model_urls = {
#     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#     'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#     'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#     'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
# }

def setup_seed(seed):
   torch.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.deterministic = True
#    torch.backends.cudnn.benchmark = False
#    torch.backends.cudnn.enabled = False


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d,nn.BatchNorm1d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        # elif isinstance(m, (nn.Sequential, nn.ModuleList, PatchEmbed,PatchEmbed_stride,\
        #      PatchExpand, PatchMerging, PatchUnembed,VisionTransformer,Attention,Block)):
        elif isinstance(m, (nn.Sequential, nn.ModuleList, PatchEmbed,\
             PatchExpand, PatchMerging)):        
            weight_init(m)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.ReLU,nn.AdaptiveAvgPool2d,nn.AdaptiveAvgPool1d,DropPath,nn.AdaptiveMaxPool1d,\
            nn.Softmax,  nn.Identity, Mlp, nn.PixelShuffle,nn.PixelUnshuffle, nn.MaxPool2d,nn.Dropout,nn.GELU,\
            nn.Sigmoid,MultiKMeans,nn.AvgPool2d,BasicLayer,nn.UpsamplingBilinear2d,nn.MSELoss,WindowAttention)):
            pass
        else:
            m.initialize()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, tgroup=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        if tgroup == True:
            self.bn1 = nn.GroupNorm(4,planes)
            self.bn2 = nn.GroupNorm(4,planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
        
    def initialize(self):
        weight_init(self)

class BasicBlockDe(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,tgroup=False):
        super(BasicBlockDe, self).__init__()

        self.convRes = conv3x3(inplanes,planes,stride)
        # self.bnRes = nn.BatchNorm2d(planes)
        self.reluRes = nn.ReLU(inplace=True)

        self.conv1 = conv3x3(inplanes, planes, stride)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        if tgroup == True:
            self.bnRes = nn.GroupNorm(4,planes)
            self.bn1 = nn.GroupNorm(4,planes)
            self.bn2 = nn.GroupNorm(4,planes)
        else:
            self.bnRes = nn.BatchNorm2d(planes)
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = self.convRes(x)
        residual = self.bnRes(residual)
        residual = self.reluRes(residual)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

    def initialize(self):
        weight_init(self)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

    def initialize(self):
        weight_init(self)

class ConvBlock(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size=3,padding=1,dilation=1,stride=1,groups=1,\
                tbn=True,trelu=True,tgroup=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch,out_ch,kernel_size=kernel_size,\
            stride=stride,padding=padding,dilation=dilation,groups=groups)
        if tgroup == True:
            self.bn = nn.GroupNorm(4,out_ch)
        else:
            self.bn = nn.BatchNorm2d(out_ch)
        # self.relu = nn.ReLU(inplace=True)
        # self.relu = nn.LeakyReLU(inplace=True)
        self.relu = nn.GELU()
        self.tbn = tbn
        self.trelu = trelu

    def forward(self, x):
        out = self.conv(x)
        if self.tbn:
            out = self.bn(out)
        if self.trelu:
            out = self.relu(out)
        return out

    def initialize(self):
        weight_init(self)

class ConvBlock1d(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size=3,padding=1,dilation=1,tbn=True,trelu=True):
        super(ConvBlock1d, self).__init__()
        self.conv = nn.Conv1d(in_ch,out_ch,kernel_size=kernel_size,\
            padding=padding,dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.tbn = tbn
        self.trelu = trelu

    def forward(self, x):
        out = self.conv(x)
        if self.tbn:
            out = self.bn(out)
        if self.trelu:
            out = self.relu(out)
        return out

class CAM(nn.Module):
    def __init__(self,in_ch):
        super(CAM, self).__init__()
        self.conv =  nn.Conv2d(in_ch,in_ch,1)
        self.bn = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU(inplace=True)

        self.gap = nn.AdaptiveAvgPool2d((1,1))


    def forward(self,x):
        hg = self.relu(self.bn(self.conv(x)))
        hg = self.gap(hg)
        hg = F.sigmoid(hg)
        hout = x*hg

        return hout

class RAM(nn.Module):
    def __init__(self):
        super(RAM, self).__init__()
        # self.conv = nn.Conv2d(in_ch,in_ch,1)

    def forward(self,x,xmap):
        hx = F.sigmoid(xmap)
        hx = F.interpolate(hx, x.shape[-1], mode='bilinear')
        ra = 1-hx
        hout = ra*x

        return hout

class SCM(nn.Module):
    def __init__(self,n_channels=3,n_classes=1):
        super(SCM,self).__init__()

        self.preconv1 = nn.Conv2d(1,1,5,padding=2).cuda()
        self.preconv2 = nn.Conv2d(1,1,5,padding=2).cuda()
        self.preconv3 = nn.Conv2d(1,1,5,padding=2).cuda()
        self.preconv4 = nn.Conv2d(1,1,5,padding=2).cuda()


        self.conv1 = nn.Conv2d(2,1,5,padding=2).cuda()
        self.conv2 = nn.Conv2d(2,1,5,padding=2).cuda()
        self.conv3 = nn.Conv2d(2,1,5,padding=2).cuda()
        self.conv4 = nn.Conv2d(2,1,5,padding=2).cuda()

    def forward(self,x):
        # assume use the same conv kernals
        # TODO: one of the variables needed for gradient 
        # computation has been modified by an inplace operation
        C,H,W = x.shape[1:4]
        

        #stage 1 top-down
        t1 = x[:,:,0,:].unsqueeze(1)
        t1 = self.preconv1(t1)
        out1 = t1
        for h in range(1,H):
            t2 = x[:,:,h,:].unsqueeze(1)
            t = torch.cat((t1,t2), 1) #上一层加上这一层
            t1 = self.conv1(t)
            out1 = torch.cat((t1,out1), 1)
        #stage 2 down-top
        t1 = out1[:,H-1,:,:].unsqueeze(1)
        out2 = self.preconv2(t1)
        for h in range(H-2, -1, -1):
            t2 = out1[:,h,:,:].unsqueeze(1)
            t = torch.cat((t1,t2), 1) 
            t1 = self.conv2(t)
            out2 = torch.cat((t1,out2), 1)
        #stage 3 left-right
        t1 = out2[:,:,:,0].unsqueeze(1)
        out3 = self.preconv3(t1)
        for w in range(1,W):
            t2 = out2[:,:,:,w].unsqueeze(1)
            t = torch.cat((t1,t2), 1) 
            t1 = self.conv3(t)
            out3 = torch.cat((t1,out3), 1)
        #stage 4 right-left
        t1 = out3[:,W-1,:,:].unsqueeze(1)
        out4 = self.preconv4(t1)
        for w in range(W-2, -1, -1):
            t2 = out3[:,w,:,].unsqueeze(1)
            t = torch.cat((t1,t2), 1) 
            t1 = self.conv4(t)
            out4 = torch.cat((t1,out4), 1)

        out = out4.permute(0,3,2,1)
        return out

class QCO_1d(nn.Module):
    def __init__(self,in_ch=16,level_num=64, width=128):
        super(QCO_1d, self).__init__()
        self.conv1 = ConvBlock(in_ch,in_ch,3,1)
        self.conv2 = ConvBlock(in_ch,width,1,0,tbn=False,trelu=False)
        self.f1 = ConvBlock1d(2,64,1,0,tbn=False)
        self.f2 = ConvBlock1d(64,width,1,0,tbn=False)
        self.out = ConvBlock1d(width*2,width,1,0)
        self.level_num = level_num

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        N, C, H, W = x.shape
        x_ave = F.adaptive_avg_pool2d(x, (1, 1)) # global vec
        cos_sim = (F.normalize(x_ave, dim=1) * F.normalize(x, dim=1)).sum(1) # cos_sim compute, L2 norm; feel like channel attention; out [H,W]
        cos_sim = cos_sim.view(N, -1) # flatten to 1d
        cos_sim_min, _ = cos_sim.min(-1) 
        cos_sim_min = cos_sim_min.unsqueeze(-1) #out [1,1]
        cos_sim_max, _ = cos_sim.max(-1)
        cos_sim_max = cos_sim_max.unsqueeze(-1) 
        q_levels = torch.arange(self.level_num).float().cuda() # arange generte [1:n-1]
        q_levels = q_levels.expand(N, self.level_num) # copy, not allocate mem, to every batch
        q_levels =  (2 * q_levels + 1) / (2 * self.level_num) * (cos_sim_max - cos_sim_min) + cos_sim_min # generate L; n/N*(max-min)+min
        q_levels = q_levels.unsqueeze(1) # out [b,1,level]
        q_levels_inter = q_levels[:, :, 1] - q_levels[:, :, 0] #first level
        q_levels_inter = q_levels_inter.unsqueeze(-1) 
        cos_sim = cos_sim.unsqueeze(-1) 
        quant = 1 - torch.abs(q_levels - cos_sim) # generate E, [1,HW,level]
        quant = quant * (quant > (1 - q_levels_inter)) #??? 1- abs(L-S) > 1 - (0.5/N) ->  filter the nosiy, which less than the 1st level
        sta = quant.sum(1) 
        sta = sta / (sta.sum(-1).unsqueeze(-1)) 
        sta = sta.unsqueeze(1) 
        sta = torch.cat([q_levels, sta], dim=1) # generate C
        sta = self.f1(sta)
        sta = self.f2(sta) 
        x_ave = x_ave.squeeze(-1).squeeze(-1) 
        x_ave = x_ave.expand(self.level_num, N, C).permute(1, 2, 0) 
        sta = torch.cat([sta, x_ave], dim=1) # generate D
        sta = self.out(sta)
        return sta, quant # D,E

class TEM(nn.Module):
    ### output of TEM is statistic infomation of each pixel
    ### It may helps the reconstruction of detail features
    ### however it is not visualiable
    def __init__(self, in_ch,level_num, width=128,qkvwidth=128,outc=256):
        super(TEM, self).__init__()
        self.level_num = level_num
        self.outc = outc
        self.qco = QCO_1d(in_ch,level_num,width)
        
        self.k = ConvBlock1d(width,qkvwidth,1,0, tbn=False,trelu=False)
        self.q = ConvBlock1d(width,qkvwidth,1,0, tbn=False,trelu=False)
        self.v = ConvBlock1d(width,qkvwidth,1,0, tbn=False,trelu=False)
    
        self.out = ConvBlock1d(qkvwidth,outc,1,0)
    def forward(self, x):
        N, C, H, W = x.shape
        sta, quant = self.qco(x) #D,E [1,width,level], [1,HW,level]
        k = self.k(sta)
        q = self.q(sta) 
        v = self.v(sta) 
        k = k.permute(0, 2, 1) # [1,level,qkvwidth]
        w = torch.bmm(k, q) # [1,level,qkvwidth]*[1,qkvwidth,level]=[1,level,level]
        w = F.softmax(w, dim=-1) 
        v = v.permute(0, 2, 1) 
        f = torch.bmm(w, v) # [1,level,level]*[1,level,qkvwidth]=[1,level,qkvwidth]
        f = f.permute(0, 2, 1) # [1,qkvwidth,level]
        f = self.out(f) # [1,outc,level]
        quant = quant.permute(0, 2, 1) #[1,level,HW]
        out = torch.bmm(f, quant) # [1,outc,level]*[1,level,HW]=[1,outc,HW]
        out = out.view(N, self.outc, H, W)
        return out # [1,outc,H,W]

class StripPooling(nn.Module):
    """
    Reference:Strip pooling + PPM
    """
    def __init__(self, in_channels, pool_size, norm_layer):
        super(StripPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(pool_size[0])
        self.pool2 = nn.AdaptiveAvgPool2d(pool_size[1])
        self.pool3 = nn.AdaptiveAvgPool2d((1, None))
        self.pool4 = nn.AdaptiveAvgPool2d((None, 1))
        inter_channels = int(in_channels/4)        
        if norm_layer == nn.GroupNorm:
            self.norm = nn.GroupNorm(4,inter_channels)

        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=True),
                                self.norm,
                                nn.ReLU(True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=True),
                                self.norm,
                                nn.ReLU(True))
        self.conv2_0 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=True),
                                self.norm)
        self.conv2_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=True),
                                self.norm)
        self.conv2_2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=True),
                                self.norm)
        self.conv2_3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=True),
                                self.norm)
        self.conv2_4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=True),
                                self.norm)
        self.conv2_5 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=True),
                                self.norm,
                                nn.ReLU(True))
        self.conv2_6 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=True),
                                self.norm,
                                nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels*2, in_channels, 1, bias=True),
                                nn.GroupNorm(4,in_channels))
        # bilinear interpolate options
        self._up_kwargs = {'mode': 'bilinear', 'align_corners': True}
        
    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x2_1 = self.conv2_0(x1)

        # hx = self.pool1(x1)
        # hx = self.conv2_1(hx)
        # hx = F.interpolate(hx, (h,w),**self._up_kwargs)
        # hd = self.pool3(x1)
        # hd = self.conv2_3(hd)
        # hd = F.interpolate(hd, (h,w),**self._up_kwargs)        

        x2_2 = F.interpolate(self.conv2_1(self.pool1(x1)), (h, w), **self._up_kwargs)
        x2_3 = F.interpolate(self.conv2_2(self.pool2(x1)), (h, w), **self._up_kwargs)
        x2_4 = F.interpolate(self.conv2_3(self.pool3(x2)), (h, w), **self._up_kwargs)
        x2_5 = F.interpolate(self.conv2_4(self.pool4(x2)), (h, w), **self._up_kwargs)
        x1 = self.conv2_5(F.relu_(x2_1 + x2_2 + x2_3))
        x2 = self.conv2_6(F.relu_(x2_5 + x2_4))
        out = self.conv3(torch.cat([x1, x2], dim=1))
        return F.relu_(x + out)


class ConvMix(nn.Module):
    def __init__(self,in_ch,dim, kernel_size=7,patch_size=8):
        super(ConvMix,self).__init__()
        self.inconv = nn.Sequential(
            nn.Conv2d(in_ch,dim,kernel_size=kernel_size,stride=patch_size),
            nn.ReLU(),
            nn.BatchNorm2d(dim)
        )
        self.conv_d = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, groups=dim, \
                padding=kernel_size//2), # floor
            nn.ReLU(),
            nn.BatchNorm2d(dim)
        )
        self.conv_p = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(dim)
        )
    def forward(self,x):
        ht = self.inconv(x)
        hx = self.conv_d(ht)
        out = self.conv_p(ht+hx)

        return out

class AttentionRegularization(nn.Module):
    def __init__(self, num_part=8):
        super(AttentionRegularization, self).__init__()
        self.num_part = num_part

    def forward(self, mask):
        # mask: bt* num_part*h*w
        bt = mask.size(0)
        num_part = mask.size(1)
        mask = mask.view(bt, num_part, -1)
        mask = F.normalize(mask, p=1, dim=2)
        mask = F.normalize(mask, p=2, dim=-1)
        temp = torch.matmul(mask, mask.transpose(2, 1)).view(bt, -1)  # bt*num_part*num_part.to(mask.device)
        I = torch.eye(num_part).unsqueeze(dim=0).expand(bt, num_part, num_part).cuda().view(bt, -1)
        orthogonal_loss = (temp - I).pow(2).sum(dim=1).pow(0.5).mean()

        return orthogonal_loss



class CannyFilter(nn.Module):
    def __init__(self,
                 k_gaussian=3,
                 mu=0,
                 sigma=1,
                 k_sobel=3,
                 use_cuda=False):
        super(CannyFilter, self).__init__()
        # device
        self.device = 'cuda' if use_cuda else 'cpu'
 
        # gaussian
 
        gaussian_2D = self.get_gaussian_kernel(k_gaussian, mu, sigma)
        self.gaussian_filter = nn.Conv2d(in_channels=1,
                                         out_channels=1,
                                         kernel_size=k_gaussian,
                                         padding=k_gaussian // 2,
                                         bias=False)
        self.gaussian_filter.weight[:] = torch.from_numpy(gaussian_2D)
 
        # sobel
 
        sobel_2D = self.get_sobel_kernel(k_sobel)
        self.sobel_filter_x = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_sobel,
                                        padding=k_sobel // 2,
                                        bias=False)
        self.sobel_filter_x.weight[:] = torch.from_numpy(sobel_2D)
 
 
        self.sobel_filter_y = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_sobel,
                                        padding=k_sobel // 2,
                                        bias=False)
        self.sobel_filter_y.weight[:] = torch.from_numpy(sobel_2D.T)
 
 
        # thin
 
        thin_kernels = self.get_thin_kernels()
        directional_kernels = np.stack(thin_kernels)
 
        self.directional_filter = nn.Conv2d(in_channels=1,
                                            out_channels=8,
                                            kernel_size=thin_kernels[0].shape,
                                            padding=thin_kernels[0].shape[-1] // 2,
                                            bias=False)
        self.directional_filter.weight[:, 0] = torch.from_numpy(directional_kernels)
 
        # hysteresis
 
        hysteresis = np.ones((3, 3)) + 0.25
        self.hysteresis = nn.Conv2d(in_channels=1,
                                    out_channels=1,
                                    kernel_size=3,
                                    padding=1,
                                    bias=False)
        self.hysteresis.weight[:] = torch.from_numpy(hysteresis)
 
 
    def forward(self, img, low_threshold=None, high_threshold=None, hysteresis=False):
        # set the setps tensors
        B, C, H, W = img.shape
        blurred = torch.zeros((B, C, H, W)).to(self.device)
        grad_x = torch.zeros((B, 1, H, W)).to(self.device)
        grad_y = torch.zeros((B, 1, H, W)).to(self.device)
        grad_magnitude = torch.zeros((B, 1, H, W)).to(self.device)
        grad_orientation = torch.zeros((B, 1, H, W)).to(self.device)
 
        # gaussian
 
        for c in range(C):
            blurred[:, c:c+1] = self.gaussian_filter(img[:, c:c+1])
 
            grad_x = grad_x + self.sobel_filter_x(blurred[:, c:c+1])
            grad_y = grad_y + self.sobel_filter_y(blurred[:, c:c+1])
 
        # thick edges
 
        grad_x, grad_y = grad_x / C, grad_y / C
        grad_magnitude = (grad_x ** 2 + grad_y ** 2) ** 0.5
        grad_orientation = torch.atan(grad_y / grad_x)
        grad_orientation = grad_orientation * (360 / np.pi) + 180 # convert to degree
        grad_orientation = torch.round(grad_orientation / 45) * 45  # keep a split by 45
 
        # thin edges
 
        directional = self.directional_filter(grad_magnitude)
        # get indices of positive and negative directions
        positive_idx = (grad_orientation / 45) % 8
        negative_idx = ((grad_orientation / 45) + 4) % 8
        thin_edges = grad_magnitude.clone()
        # non maximum suppression direction by direction
        for pos_i in range(4):
            neg_i = pos_i + 4
            # get the oriented grad for the angle
            is_oriented_i = (positive_idx == pos_i) * 1
            is_oriented_i = is_oriented_i + (positive_idx == neg_i) * 1
            pos_directional = directional[:, pos_i]
            neg_directional = directional[:, neg_i]
            selected_direction = torch.stack([pos_directional, neg_directional])
 
            # get the local maximum pixels for the angle
            is_max = selected_direction.min(dim=0)[0] > 0.0
            is_max = torch.unsqueeze(is_max, dim=1)
 
            # apply non maximum suppression
            to_remove = (is_max == 0) * 1 * (is_oriented_i) > 0
            thin_edges[to_remove] = 0.0
 
        # thresholds
 
        if low_threshold is not None:
            low = thin_edges > low_threshold
 
            if high_threshold is not None:
                high = thin_edges > high_threshold
                # get black/gray/white only
                thin_edges = low * 0.5 + high * 0.5
 
                if hysteresis:
                    # get weaks and check if they are high or not
                    weak = (thin_edges == 0.5) * 1
                    weak_is_high = (self.hysteresis(thin_edges) > 1) * weak
                    thin_edges = high * 1 + weak_is_high * 1
            else:
                thin_edges = low * 1
 
 
        return blurred, grad_x, grad_y, grad_magnitude, grad_orientation, thin_edges

    def get_thin_kernels(self,start=0, end=360, step=45):
        k_thin = 3  # actual size of the directional kernel
        # increase for a while to avoid interpolation when rotating
        k_increased = k_thin + 2
 
        # get 0° angle directional kernel
        thin_kernel_0 = np.zeros((k_increased, k_increased))
        thin_kernel_0[k_increased // 2, k_increased // 2] = 1
        thin_kernel_0[k_increased // 2, k_increased // 2 + 1:] = -1
 
        # rotate the 0° angle directional kernel to get the other ones
        thin_kernels = []
        for angle in range(start, end, step):
            (h, w) = thin_kernel_0.shape
            # get the center to not rotate around the (0, 0) coord point
            center = (w // 2, h // 2)
            # apply rotation
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
            kernel_angle_increased = cv2.warpAffine(thin_kernel_0, rotation_matrix, (w, h), cv2.INTER_NEAREST)
 
            # get the k=3 kerne
            kernel_angle = kernel_angle_increased[1:-1, 1:-1]
            is_diag = (abs(kernel_angle) == 1)      # because of the interpolation
            kernel_angle = kernel_angle * is_diag   # because of the interpolation
            thin_kernels.append(kernel_angle)
        return thin_kernels

    def get_gaussian_kernel(self,k=3, mu=0, sigma=1, normalize=True):
        # compute 1 dimension gaussian
        gaussian_1D = np.linspace(-1, 1, k)
        # compute a grid distance from center
        x, y = np.meshgrid(gaussian_1D, gaussian_1D)
        distance = (x ** 2 + y ** 2) ** 0.5
    
        # compute the 2 dimension gaussian
        gaussian_2D = np.exp(-(distance - mu) ** 2 / (2 * sigma ** 2))
        gaussian_2D = gaussian_2D / (2 * np.pi *sigma **2)
    
        # normalize part (mathematically)
        if normalize:
            gaussian_2D = gaussian_2D / np.sum(gaussian_2D)
        return gaussian_2D

    def get_sobel_kernel(self,k=3):
        # get range
        range = np.linspace(-(k // 2), k // 2, k)
        # compute a grid the numerator and the axis-distances
        x, y = np.meshgrid(range, range)
        sobel_2D_numerator = x
        sobel_2D_denominator = (x ** 2 + y ** 2)
        sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
        sobel_2D = sobel_2D_numerator / sobel_2D_denominator
        return sobel_2D
