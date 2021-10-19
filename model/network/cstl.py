import torch 
import torch.nn as nn
import torch.nn.init as init
import numpy as np 
import torch.nn.functional as F 
import random

def conv1d(in_planes, out_planes, kernel_size, has_bias=False, **kwargs):
    return nn.Conv1d(in_planes, out_planes, kernel_size, bias=has_bias, **kwargs)

def mlp_sigmoid(in_planes, out_planes, kernel_size, **kwargs):
    return nn.Sequential(conv1d(in_planes, in_planes//16, kernel_size, **kwargs),
                            nn.BatchNorm1d(in_planes//16),
                            nn.LeakyReLU(inplace=True),
                            conv1d(in_planes//16, out_planes, kernel_size, **kwargs),
                            nn.Sigmoid())

def conv_bn(in_planes, out_planes, kernel_size, **kwargs):
    return nn.Sequential(conv1d(in_planes, out_planes, kernel_size, **kwargs),
                            nn.BatchNorm1d(out_planes))


class MSTE(nn.Module):
    def __init__(self, in_planes, out_planes, part_num):
        super(MSTE, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.part_num = part_num

        self.score = mlp_sigmoid(in_planes*part_num, in_planes*part_num, 1, groups=part_num)

        self.short_term = nn.ModuleList([conv_bn(in_planes*part_num, out_planes*part_num, 3, padding=1, groups=part_num), 
                                conv_bn(in_planes*part_num, out_planes*part_num, 3, padding=1, groups=part_num)])

    def get_frame_level(self, x):
        return x 

    def get_short_term(self, x):
        n, s, c, h = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(n, -1, s)
        temp = self.short_term[0](x)
        short_term_feature = temp + self.short_term[1](temp)
        return short_term_feature.view(n, h, c, s).permute(0, 3, 2, 1).contiguous() 

    def get_long_term(self, x):
        n, s, c, h = x.size()
        x = x.permute(0, 3, 2, 1).contiguous()
        pred_score = self.score(x.view(n, h * c, s)).view(n, h, c, s)
        long_term_feature = x.mul(pred_score).sum(-1).div(pred_score.sum(-1))
        long_term_feature = long_term_feature.unsqueeze(1).repeat(1, s, 1, 1)
        return long_term_feature.permute(0, 1, 3, 2).contiguous()

    def forward(self, x):
        multi_scale_feature = [self.get_frame_level(x), self.get_short_term(x), self.get_long_term(x)]
        return multi_scale_feature


class ATA(nn.Module):
    def __init__(self, in_planes, part_num, div):
        super(ATA, self).__init__()
        self.in_planes = in_planes
        self.part_num = part_num
        self.div = div

        self.in_conv = conv1d(part_num*3*in_planes, part_num*3*in_planes // div, 1, False, groups=part_num)

        self.out_conv = conv1d(part_num*3*in_planes // div, part_num*3*in_planes, 1, False, groups=part_num)
        
        self.leaky_relu = nn.LeakyReLU(inplace=True)
    
    def forward(self, t_f, t_s, t_l):
        n, s, c, h = t_f.size()
        t_f = t_f.unsqueeze(-1)
        t_s = t_s.unsqueeze(-1) + t_f
        t_l = t_l.unsqueeze(-1) + t_s

        t = torch.cat([t_f, t_s, t_l], -1)
        t = t.permute(0, 3, 4, 2, 1).contiguous().view(n, h*3*c, s)

        t_inter = self.leaky_relu(self.in_conv(t))
        t_attention = self.out_conv(t_inter).sigmoid()
        weighted_sum = (t_attention * t).view(n, h, 3, c, s).sum(2).sum(-1) / t_attention.view(n, h, 3, c, s).sum(2).sum(-1)
        weighted_sum = self.leaky_relu(weighted_sum).permute(1, 0, 2).contiguous()

        return weighted_sum


class SSFL(nn.Module):
    def __init__(self, in_planes, out_planes, part_num, class_num):
        super(SSFL, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.part_num = part_num
        self.class_num = class_num

        self.part_score = mlp_sigmoid(in_planes*part_num*3, part_num, 1, groups=part_num)

        self.decay_channel = conv1d(in_planes*part_num*3, out_planes*part_num, 1, groups=part_num)

        self.bn = nn.ModuleList()
        for i in range(part_num):
            self.bn.append(nn.BatchNorm1d(in_planes))

        self.fc = nn.Parameter(
            init.xavier_uniform_(
                torch.zeros(1, in_planes, class_num)))

    def forward(self, t_f, t_s, t_l):
        n, s, c, p = t_f.size()
        cat_feature = torch.cat([t_f, t_s, t_l], 2)
        part_score = self.part_score(cat_feature.permute(0, 3, 2, 1).contiguous().view(n, -1, s)).view(n, p, 1, s)

        cat_feature = self.decay_channel(cat_feature.permute(0, 3, 2, 1).contiguous().view(n, -1, s)).view(n, p, c, s)

        weighted_part_vector = (cat_feature * part_score).sum(-1) / part_score.sum(-1) #nxpxc

        # part classification
        part_feature = []
        for idx, block in enumerate(self.bn):
            part_feature.append(block(weighted_part_vector[:, idx, :]).unsqueeze(0))
        part_feature = torch.cat(part_feature, 0)

        part_classification = part_feature.matmul(self.fc)
        
        # part selection
        max_part_idx = part_score.max(-1)[1].squeeze(-1)

        selected_part_feature = []
        for i in range(self.part_num):
            s_idx = max_part_idx[:, i]
            selected_part_feature.append(t_f[range(0, n), s_idx, :, i].view(n, c, -1))
        
        selected_part_feature = torch.cat(selected_part_feature, 2).permute(2, 0, 1).contiguous()

        weighted_part_vector = weighted_part_vector.permute(1, 0, 2).contiguous()

        return part_classification, weighted_part_vector, selected_part_feature