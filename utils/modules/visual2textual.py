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

class Adapter(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, emb_dim//4)
        self.act = QuickGELU()
        self.fc2 = nn.Linear(emb_dim//4, emb_dim)
    def forward(self, x):
        xs = self.fc1(x)
        xs = self.act(xs)
        xs = self.fc2(xs)
        x = xs + x
        return x

class Attention(nn.Module):
    def __init__(self, emb_dim:int):
        super().__init__()
        self.n_heads = emb_dim //64
        self.scale = emb_dim ** -0.5
        self.query = nn.Linear(emb_dim, emb_dim)
        self.key = nn.Linear(emb_dim, emb_dim)
        self.value = nn.Linear(emb_dim, emb_dim)
    def forward(self, q, k, v):
        # q.shape = a, m, d
        # kv.shape = a, t, d
        a, m, _ = q.shape
        _, t, _ = k.shape
        h = self.n_heads
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        q = q.view(a, m, h, -1).transpose(1,2) # a, n_heads, m, dim/n_heads
        k = k.view(a, t, h, -1).transpose(1,2) # b, n_heads, t, dim/n_heads
        v = v.view(a, t, h, -1).transpose(1,2) # b, n_heads, t, dim/n_heads

        dots = torch.einsum('ahmc,ahtc->ahmt', [q,k]) # a, n_heads, m, t
        attn = dots.softmax(dim=-1)
        out = torch.einsum('ahmt,ahtc->ahmc', [attn,v]) # a, n_heads, m ,d/n_heads
        out = out.transpose(1,2).contiguous().view(a,m,-1) # a, m, d

        return out


class VisualBasedPrompt(nn.Module):
    def __init__(self, emb_dim:int, n_frames:int, dropout):
        super().__init__()
        self.attn = Attention(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            QuickGELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 4, emb_dim)
        )
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.norm3 = nn.LayerNorm(emb_dim)

    def forward(self, x, y):
        q = self.norm1(x)
        k = self.norm2(y)
        v = self.norm2(y)
        prompts = x + self.attn(q,k,v)
        prompts = prompts + self.mlp(self.norm3(prompts))
        return prompts




class VisualToTextual(nn.Module):
    def __init__(self, n_frames:int, num_prompt:int, emb_dim:int, 
                # clip_model, 
                dtype,
                dropout,
                pretrained,
                token_embedding):
        super().__init__()
        self.token_embedding = token_embedding
        self.n_frames = n_frames
        self.num_prompt = num_prompt
        self.emb_dim = emb_dim
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(self.num_prompt, self.emb_dim) # m,d
        nn.init.normal_(self.embedding.weight, std=0.01)

        self.visual_based_prompt = VisualBasedPrompt(emb_dim, n_frames, dropout)
        self.meta_token = nn.Parameter(torch.zeros([self.emb_dim]))
        nn.init.normal_(self.meta_token, std=0.02)
        

        self.adapter = Adapter(emb_dim)

        self.apply(self._init_weights)
        


    def construct_prompts(self, learnable_prompts, prefix, suffix):
        prompts = torch.cat(
                        [
                            prefix,
                            learnable_prompts,
                            suffix
                        ],
                        dim = 1
        )
        return prompts

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
        for n,m in self.named_modules():
            if 'adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)
                  
    

    def forward(self, img_embs_p, texts):
        # vid_embs (b,d)
        # _, d = img_embs.shape
        # img_embs = img_embs.reshape(-1, self.n_frames, d) # b,t,d

        # meta token
        meta_token = self.meta_token.view(1,1,-1).repeat(img_embs_p.shape[0], 1, 1) # b,1,d
        meta_token = self.visual_based_prompt(meta_token, img_embs_p) # b,1,d
        meta_token = self.adapter(meta_token)
       

        # meta_token = self.adapter(img_embs_p) #b,d
        # meta_token = meta_token.unsqueeze(1) # b,1,d
       

        # learnable prompt
        prompt_prefix = " ".join(["X"] * (self.num_prompt+1)) # placeholder
        prompts = [prompt_prefix + " " + name for name in texts] 
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]) # a,77

        tokenized_prompts = tokenized_prompts.to(img_embs_p.device)

        with torch.no_grad():
            embedding = self.token_embedding(tokenized_prompts) #a,77,d
        
        token_prefix = embedding[:, :1, :]  # SOS 
        token_suffix = embedding[:, 1 + self.num_prompt+1 :, :] # CLS EOS

        learnable_prompt = self.embedding(torch.arange(self.num_prompt).to(img_embs_p.device)) # m, d
        learnable_prompt = learnable_prompt.unsqueeze(0) # 1,m,d
        learnable_prompt = learnable_prompt.expand(img_embs_p.shape[0], -1, -1) # b,m,d

        # token_embs = self.construct_prompts(learnable_prompt, token_prefix, token_suffix)  # b,77,d

        # concat
        visual_based_prompt = torch.cat([learnable_prompt, meta_token], dim=1) # b,m+1,d
    
        token_embs = []
        for visual_based_prompt_i in visual_based_prompt: # m+1,d
            prompt_i = visual_based_prompt_i.unsqueeze(0).expand(embedding.shape[0], -1, -1) # a,m+1,d
            token_embs_i = self.construct_prompts(prompt_i, token_prefix, token_suffix) # a, 77, d
            token_embs.append(token_embs_i)
        token_embs = torch.stack(token_embs) # 


        return token_embs, tokenized_prompts #a,77,d a,77

