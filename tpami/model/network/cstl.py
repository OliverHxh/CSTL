import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np 
import torch.nn.functional as F

from .vit import ViT, Attention


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
        n, p, c, s = x.size()
        temp = self.short_term[0](x.view(n, -1, s))
        short_term_feature = temp + self.short_term[1](temp)

        return short_term_feature.view(n, p, c, s)


    def get_long_term(self, x):
        n, p, c, s = x.size()
        pred_score = self.score(x.view(n, -1, s)).view(n, p, c, s)
        long_term_feature = x.mul(pred_score).sum(-1).div(pred_score.sum(-1))
        long_term_feature = long_term_feature.unsqueeze(3).repeat(1, 1, 1, s)

        return long_term_feature


    def forward(self, x):

        return self.get_frame_level(x), self.get_short_term(x), self.get_long_term(x)



class ATA(nn.Module):
    def __init__(self, in_planes, out_planes, groups, depth, num_head, decay, kernel_size, stride):
        super(ATA, self).__init__()
        self.groups = groups
        self.t = ViT(in_planes, out_planes, groups, depth, num_head, decay, kernel_size, stride)
        self.fc = conv_bn(in_planes*groups*3, in_planes*groups, 1, groups=groups)
        self.activate = nn.LeakyReLU(inplace=True)


    def forward(self, t_f, t_s, t_l):
        n, p, c, s = t_f.size()
        temporal_feature = self.fc(torch.cat([t_f, t_s, t_l], 2).view(n, -1, s))
        temporal_feature = self.t(temporal_feature.view(n, p, -1, s)) + temporal_feature.view(n, p, -1, s)
        weighted_feature = self.activate(temporal_feature.max(-1)[0]).permute(1, 0, 2).contiguous()

        return weighted_feature


class SSFL(nn.Module):
    def __init__(self, in_planes, out_planes, num_head, part_num, class_num, topk_num):
        super(SSFL, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.num_head = num_head
        self.part_num = part_num
        self.class_num = class_num
        self.topk_num = topk_num

        self.decay_c = nn.Conv1d(in_planes*part_num*3, in_planes*part_num, 1, bias=False, groups=part_num)

        self.frame_corr = Attention(in_planes, part_num, num_head)

        self.bn = nn.ModuleList()
        for i in range(part_num):
            self.bn.append(nn.BatchNorm1d(in_planes))

        self.fc = nn.Parameter(
            init.xavier_uniform_(
                torch.zeros(1, in_planes, class_num)))

    def frame_correlation(self, t_all):
        weighted_part_vector, corrleation = self.frame_corr(t_all)
        weighted_part_vector = (weighted_part_vector + t_all).max(-1)[0]

        return weighted_part_vector, corrleation


    def select_topk_part(self, corr_matrix, t_f):
        n, p, c, s = t_f.shape
        
        topk_part_index = torch.topk(corr_matrix.sum(-2), self.topk_num, dim=-1, largest=True)[1].unsqueeze(4).repeat(1, 1, 1, 1, c)

        selected_topk_part = torch.zeros_like(t_f[..., 0])

        for i in range(self.num_head):
            selected_topk_part += torch.gather(t_f.transpose(2, 3), dim=2, index=topk_part_index[:, i]).squeeze(2)
        
        return selected_topk_part


    def forward(self, t_f, t_s=None, t_l=None):
        n, p, c, s = t_f.shape
        t_all = self.decay_c(torch.cat([t_f, t_s, t_l], 2).view(n, -1, s)).view(n, p, c, s)

        weighted_part_vector, attention = self.frame_correlation(t_all)
        selected_topk_part = self.select_topk_part(attention, t_f)

        part_feature = []
        for idx, block in enumerate(self.bn):
            part_feature.append(block(weighted_part_vector[:, idx, :]).unsqueeze(0))
        part_feature = torch.cat(part_feature, 0)

        part_classification = part_feature.matmul(self.fc)

        return  part_classification.transpose(0, 1), weighted_part_vector.transpose(0, 1), selected_topk_part.transpose(0, 1)
