import torch
import torch.nn as nn
import torch.nn.functional as F
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
# from clip.model import CLIP, Transformer
from timm.models.layers import trunc_normal_
from collections import OrderedDict


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class MulitHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)


        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        # q:cls_embs (b,a,d)
        # k,v:vid_embs (b,t,d)
        B, N, C = q.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0,2,1,3)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads).permute(0,2,1,3)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads).permute(0,2,1,3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class PromptGeneratorLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.,
    ):
        super().__init__()
        self.cross_attn = MulitHeadAttention(d_model, nhead, proj_drop=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            QuickGELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, cls_embs, vid_embs):
        q = self.norm1(cls_embs)
        k = v = self.norm2(vid_embs)
        x = cls_embs + self.cross_attn(q, k, v)
        x = x + self.dropout(self.mlp(self.norm3(x)))
        return x



class PostVisualToTextual(nn.Module):
    def __init__(self, emb_dim:int):
        super().__init__()
        self.emb_dim = emb_dim
        # self.mlp = nn.Sequential(OrderedDict([
        #     ("c_fc", nn.Linear(emb_dim, emb_dim * 4)),
        #     ("gelu", QuickGELU()),
        #     ("c_proj", nn.Linear(emb_dim * 4, emb_dim))
        # ]))
        layers=2
        self.prompt_generator = nn.ModuleList([PromptGeneratorLayer(d_model = emb_dim, nhead=8) for _ in range(layers)])
        alpha = 0.1
        self.alpha = nn.Parameter(torch.ones(emb_dim) * alpha)
        self.apply(self._init_weights_)

    

    def _init_weights_(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, cls_embs, img_embs_p):
        # cls_embs (b,a,d)
        # img_embs_p (b,N,d)
        for layer in self.prompt_generator:
            cls_embs = layer(cls_embs, img_embs_p)
            
        cls_embs = cls_embs * self.alpha
        return cls_embs
