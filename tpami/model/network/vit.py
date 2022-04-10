import torch 
import torch.nn as nn 
import numpy as np 


class Attention(nn.Module):
    def __init__(self, in_dims, part_num, num_head):
        super(Attention, self).__init__()
        self.part_num = part_num
        self.num_head = num_head
        self.dim_head = in_dims // num_head

        self.scale = self.dim_head ** (-0.5)
        self.softmax = nn.Softmax(dim=-1)
        self.to_qkv = nn.Conv1d(in_dims*part_num, in_dims*3*part_num, 1, bias=False, groups=part_num)
        self.to_out = nn.Conv1d(in_dims*part_num, in_dims*part_num, 1, bias=False, groups=part_num)


    def forward(self, x):
        n, p, c, d = x.shape

        qkv = self.to_qkv(x.view(n, p*c, d))
        qkv = qkv.view(n, p, 3, self.num_head, self.dim_head, d).permute(2, 0, 3, 1, 4, 5).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2] #[n, num_head, p, dim_head, d]

        dots = torch.matmul(q.transpose(-2, -1), k) * self.scale
        attn = self.softmax(dots)

        out = torch.matmul(attn, v.transpose(-2, -1)).transpose(-2, -1) #[n, num_head, p, dim_head, d]

        out = self.to_out(out.permute(0, 2, 1, 3, 4).contiguous().view(n, -1, d)).view(n, p, c, d)

        return out, attn

class PreNorm(nn.Module):
    def __init__(self, part_num, in_dims, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(in_dims)
        self.fn = fn
    
    def forward(self, x):
        n, p, c, d = x.shape

        return self.fn(self.norm(x.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous())


class FeedForward(nn.Module):
    def __init__(self, in_dims, part_num, decay=16):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dims*part_num, in_dims*part_num//decay, 1, bias=False, groups=part_num),
            nn.BatchNorm1d(in_dims*part_num//decay),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(in_dims*part_num//decay, in_dims*part_num, 1, bias=False, groups=part_num)
        )

    def forward(self, x):
        n, p, c, d = x.size()
        out = self.net(x.view(n, -1, d)).view(n, p, c, d)
        return out


class Transformer(nn.Module):
    def __init__(self, in_dims, depth, num_head, decay, part_num):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(part_num, in_dims, Attention(in_dims, part_num, num_head)),
                PreNorm(part_num, in_dims, FeedForward(in_dims, part_num, decay))
            ]))

    def forward(self, x): # nxpxcxd
        for attn, ff in self.layers:
            x = attn(x)[0] + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, in_dims, out_dims, part_num, depth, num_head, decay, kernel_size=1, stride=1):
        super(ViT, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.stride = stride

        self.raw_embedding = nn.Conv1d(in_dims*part_num, out_dims*part_num, 1, bias=False, groups=part_num)
        self.proj = nn.Conv1d(in_dims, out_dims, kernel_size, stride, padding=kernel_size//2, bias=False, groups=in_dims)
        
        self.transformer = Transformer(in_dims, depth, num_head, decay, part_num)
        
        self.activate = nn.LeakyReLU(inplace=True)


    def seq_embedding(self, x):
        n, p, c, d = x.size()
        seq_embedded = self.raw_embedding(x.view(n, -1, d)).view(n, p, -1, d)

        return seq_embedded

    def pos_embedding(self, x):
        n, p, c, d = x.size()
        feat_token = self.proj(x.view(-1, c, d)).view(n, p, -1, d)
        if self.stride == 1:
            feat_token += x
        return feat_token

    def forward(self, x):
        embedded_feature = self.seq_embedding(x) + self.pos_embedding(x)
        trans_feature = self.transformer(embedded_feature)
        return self.activate(trans_feature)
