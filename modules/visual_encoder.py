from typing import Tuple
from collections import OrderedDict
import math
import functools
import clip

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from einops import rearrange


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

        

class ResidualAttentionBlock(nn.Module): 
    def __init__(self, d_model:int, num_heads:int, n_frames:int,  attn_mask: torch.Tensor=None, scale=1., drop_path=0.):
        super().__init__()
        self.attn=nn.MultiheadAttention(d_model, num_heads)
        self.ln_1=LayerNorm(d_model)
        self.mlp=nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2=LayerNorm(d_model)
        self.attn_mask=attn_mask
        self.num_heads=num_heads
        self.n_frames = n_frames

        self.scale=scale
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def attention(self,  x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        #x shape [N+1, b*t, d]
        n, bt, d = x.shape
        x=x+self.attention(self.ln_1(x))
        x=x+self.drop_path(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, vision_width:int, vision_layers:int, vision_heads:int, n_frames:int, attn_mask: torch.Tensor=None, drop_path=0.1):
        super().__init__()
        self.width=vision_width
        self.layers=vision_layers
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.layers)]
        self.resblocks=nn.Sequential(*[ResidualAttentionBlock(vision_width, vision_heads, n_frames, attn_mask,dpr[i]) for i in range(vision_layers)])

    def forward(self,x: torch.Tensor):
        return self.resblocks(x)


class VisualEncoder(nn.Module):
    def __init__(self, input_resolution:int, patch_size:int, vision_width:int, vision_layers:int, vision_heads:int, output_dim:int, n_frames:int, drop_path_rate, joint=True, pretrained=None):
        super().__init__()
        self.input_resolution=input_resolution
        self.n_frames = n_frames
        self.pretrained=pretrained
        self.conv1=nn.Conv2d(in_channels=3,out_channels=vision_width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale=vision_width**-0.5
        self.layers=vision_layers
        self.class_embedding=nn.Parameter(scale*torch.randn(vision_width))
        self.positional_embedding=nn.Parameter(scale*torch .randn((input_resolution //patch_size)**2+1, vision_width))
        self.ln_pre=LayerNorm(vision_width)

        # self.temporal_embedding = nn.Parameter(torch.zeros(1, n_frames, vision_width))
        self.transformer=Transformer(vision_width, vision_layers, vision_heads, n_frames, drop_path=drop_path_rate)

        self.ln_post=LayerNorm(vision_width)
        self.proj = nn.Parameter(scale * torch.randn(vision_width, output_dim))

        self.apply(self._init_weights_)

    

    def _init_weights_(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        


    def forward(self, x: torch.Tensor):
        ##输入x(b*t, c, h, w)
        x=self.conv1(x)  # b*t,d,H/h,W/w
        x = x.reshape(x.shape[0], x.shape[1], -1)  # b*t,d,N
        x = x.permute(0, 2, 1)  # b*t,N,d
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # b*t,N+1,d
        x = x + self.positional_embedding.to(x.dtype)  # b*t,N+1,d

        x = self.ln_pre(x)  # b*t,N+1,d
        x = x.permute(1, 0, 2)  # N+1, b*t, d
        x = self.transformer(x) # N+1, b*t, d
        x = x.permute(1, 0, 2)  # b*t, N+1, d
        x=self.ln_post(x) # b*t, N+1, d                  768


        if self.proj is not None:
            img_embs = x[:,0,:] @ self.proj    #512
        return img_embs, x[:,1:,:]  # b*t,d   b*t,N,768
