import torch
import torch.nn as nn
import torch.nn.functional as F
from clip.model import CLIP

from collections import OrderedDict
from timm.models.layers import trunc_normal_


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, emb_dim:int, n_head:int, attn_mask: torch.Tensor=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(emb_dim, n_head)
        self.ln_1 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(emb_dim, emb_dim*4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(emb_dim*4, emb_dim))
        ]))
        self.ln_2 = nn.LayerNorm(emb_dim)
        self.attn_mask = attn_mask
    
    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
    
    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TemporalTransformer(nn.Module):
    def __init__(self, emb_dim:int, layers:int, n_heads:int, attn_mask: torch.Tensor=None):
        super().__init__()
        self.emb_dim = emb_dim
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(emb_dim, n_heads, attn_mask) for _ in range(layers)])
        self.grad_checkpointing = False
    
    def forward(self, x: torch.Tensor):
        for r in self.resblocks:
            if  self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x)
            else:
                x = r(x)
        return x

class VideoHead(nn.Module):
    def __init__(self,emb_dim:int, layers:int, n_frames:int):
        super().__init__()
        n_heads = emb_dim//64
        self.n_frames = n_frames
        self.frame_position_embedding = nn.Embedding(n_frames, emb_dim)
        self.temporal_transformer = TemporalTransformer(emb_dim=emb_dim, layers=layers, n_heads=n_heads)
        self.apply(self._init_weights)

    # def init_weights(self, module):
    #     """ Initialize the weights.
    #     """
    #     if isinstance(module, (nn.Linear, nn.Embedding)):
    #         # Slightly different from the TF version which uses truncated_normal for initialization
    #         # cf https://github.com/pytorch/pytorch/pull/5617
    #         module.weight.data.normal_(mean=0.0, std=0.02)
    #     elif isinstance(module, LayerNorm):
    #         if 'beta' in dir(module) and 'gamma' in dir(module):
    #             module.beta.data.zero_()
    #             module.gamma.data.fill_(1.0)
    #         else:
    #             module.bias.data.zero_()
    #             module.weight.data.fill_(1.0)
    #     if isinstance(module, nn.Linear) and module.bias is not None:
    #         module.bias.data.zero_() 

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, img_embs):
        _, d =img_embs.shape
        x = img_embs.view(-1, self.n_frames, d) #b,t,d
        x = x.contiguous()

        x_original = x
        seq_length = self.n_frames
        position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device) # 0,1,...,t-1
        position_ids = position_ids.unsqueeze(0).expand(x.shape[0], -1)
        frame_position_embeddings = self.frame_position_embedding(position_ids)
        x = x + frame_position_embeddings

        x = x.permute(1,0,2) # t,b,d
        x = self.temporal_transformer(x)
        x = x.permute(1,0,2) # b,t,d
        

        return x




