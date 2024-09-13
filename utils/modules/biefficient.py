from .visual_encoder import VisualEncoder
from .videohead import VideoHead
from .visual2textual import VisualToTextual
from .post_visual2textual import PostVisualToTextual

from typing import Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from pathlib import Path
import sys
# sys.path.append("../")
sys.path.append(str(Path(__file__).resolve().parents[1]))

from clip import clip
from clip.clip import _download, _MODELS
from clip.model import CLIP, Transformer, LayerNorm




class BiEfficient(CLIP):
    def __init__(self,device,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_patch_size: int,
                 vision_width: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 # video
                 n_frames: int,
                 num_prompt:int,

                 droppath=0.,
                 pretrained=None,
                 ):
        super().__init__(
            embed_dim,
            image_resolution, vision_layers, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
        )
        vision_heads = vision_width//64
        self.n_frames = n_frames
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.emb_dropout = droppath
        self.dropout = nn.Dropout(droppath)
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.visual = VisualEncoder(
            input_resolution=image_resolution, patch_size=vision_patch_size, vision_width=vision_width, 
            vision_layers=vision_layers, vision_heads=vision_heads, output_dim=embed_dim, n_frames=n_frames, drop_path_rate=droppath,  pretrained=None
        )
        self.transformer=Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )
        self.VisualToTextual = VisualToTextual(
            n_frames=n_frames, num_prompt=num_prompt,  emb_dim=embed_dim, dtype=self.dtype, dropout=droppath, pretrained=pretrained, token_embedding=self.token_embedding
        )
        self.VideoHead = VideoHead(
            emb_dim=embed_dim, layers=2, n_frames=n_frames
        )
        self.PostVisualToTextual = PostVisualToTextual(emb_dim=embed_dim)
        self.vcs = True
        self.prompts_visual_ln = LayerNorm(vision_width)
        self.word_ln = LayerNorm(embed_dim)
        self.cls_ln = LayerNorm(embed_dim)
        self.prompts_visual_proj = nn.Parameter(torch.randn(vision_width, embed_dim))
        self.alpha = nn.Parameter(torch.ones(embed_dim) * 0.01)

        

    def no_weight_decay_keywords(self):
        return {'positional_embedding', 'temporal_embedding'}

    def encode_image(self, images):
        return self.visual(images)  # b*t, d

    def encode_text(self,  texts, img_embs_p):
        token_embs, tokenized_prompts = self.VisualToTextual(img_embs_p, texts) 
        cls_embs = []
        word_embs =[]
        for token_emb in token_embs: # token_emb (a,77,d)
            x = token_emb + self.positional_embedding.type(self.dtype)
            if self.emb_dropout > 0:
                x = self.dropout(x)
            x = x.permute(1, 0, 2) 
            x = self.transformer(x)
            x = x.permute(1, 0, 2) 
            x = self.ln_final(x).type(self.dtype) 
            word_emb = x @ self.text_projection
            cls_emb= x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
            word_embs.append(word_emb)
            cls_embs.append(cls_emb)
        cls_embs = torch.stack(cls_embs) # b,a,d
        word_embs = torch.stack(word_embs) # b,a,77,d

        cls_embs = cls_embs + self.PostVisualToTextual(cls_embs, img_embs_p) # b,a,d        
        return cls_embs, word_embs

        
        # eos_indx = tokenized_prompts.argmax(dim=-1)
        # x = token_embs + self.positional_embedding.type(self.dtype)
        # x = x.permute(1,0,2)
        # x  = self.transformer(x)
        # x = x.permute(1,0,2)
        # x = self.ln_final(x).type(self.dtype)
        # word_embs = x@self.text_projection
        # cls_embs = x[torch.arange(x.shape[0]), eos_indx] @ self.text_projection # a,d

        # cls_embs = cls_embs.unsqueeze(0).expand(img_embs_p.shape[0], -1,-1) # b,a,d
        # cls_embs = cls_embs + self.PostVisualToTextual(cls_embs, img_embs_p) # b,a,d
        # return cls_embs, word_embs # (b,a,d) (a,77,d)


    def encode_video(self, img_embs):
        vid_embs = self.VideoHead(img_embs) # b,t,d
        # _, d = img_embs.shape
        # img_embs = img_embs.view(-1, self.n_frames, d) # b,t,d
        # vid_embs = vid_embs.mean(dim=1, keepdim=False) # b,d
        return vid_embs

    def get_logits(self, vid_embs, cls_embs):
        # vid_embs = vid_embs/vid_embs.norm(dim=-1, keepdim=True)
        # cls_embs = cls_embs/cls_embs.norm(dim=-1, keepdim=True)
        # logits = vid_embs@cls_embs.t()
        # logit_scale = self.logit_scale.exp()
        # logits = logits * logit_scale
        # vid_embs = vid_embs/vid_embs.norm(dim=-1, keepdim=True)

        vid_embs = vid_embs.mean(dim=1, keepdim=False) # b,d
        logits = []
        for cls_emb, vid_emb in zip(cls_embs, vid_embs):
            vid_emb = vid_emb/vid_emb.norm(dim=-1, keepdim=True)
            cls_emb = cls_emb/cls_emb.norm(dim=-1, keepdim=True)
            logit = vid_emb@cls_emb.t()
            logit_scale = self.logit_scale.exp()
            logit = logit * logit_scale
            logits.append(logit)
        logits = torch.stack(logits)
        return logits
    
    def get_logits_vcs(self, word_embs, cls_embs, vid_embs):
        # cls_embs(b,a,d)
        # word_embs(a,77,d)
        # vid_embs(b,t,d)


        # word_embs (a,77,d)
        # cls_embs (a,d)
        # vid_embs (b,t,d)
        # cls_embs = cls_embs.unsqueeze(0).expand(vid_embs.shape[0],-1,-1)  #(b,a,d)
        # word_embs = word_embs.unsqueeze(0).expand(vid_embs.shape[0],-1,-1,-1) #(b,a,77,d)

        # word_embs (b,a,77,d)
        # cls_embs (b,a,d)
        # vid_embs (b,t,d)
        vid_embs_attn = []
        for cls_emb, word_emb, vid_emb in zip(cls_embs, word_embs, vid_embs):
            # cls_emb (a,d) 
            # word_emb (a,77,d)
            # vid_emb (t,d)

            word_emb=self.word_ln(word_emb)
            cls_emb=self.cls_ln(cls_emb)

            word_emb = word_emb/word_emb.norm(dim=-1, keepdim=True)
            vid_emb = vid_emb/vid_emb.norm(dim=-1, keepdim=True)
            cls_emb = cls_emb/cls_emb.norm(dim=-1, keepdim=True)

            vid_emb_mean = vid_emb.mean(dim=0, keepdim=False) # d
            similarity = torch.einsum('ad,d->a', [cls_emb, vid_emb_mean]) # a
            eos_indx = similarity.argmax(dim=0)
            sims = torch.einsum('wd,td->wt', [word_emb[eos_indx], vid_emb]) # w,t
            attn_weight_v = F.softmax(sims/0.01, dim=-1) # w,t
            attn_weight_v = attn_weight_v.mean(dim=0, keepdim=False) # t
            v_attn = torch.einsum('t,td->d', [attn_weight_v, vid_emb]) #d
            v_attn = vid_emb_mean + self.alpha* v_attn
            vid_embs_attn.append(v_attn)
        vid_embs_attn = torch.stack(vid_embs_attn) # b,d

        cls_embs = cls_embs/cls_embs.norm(dim=-1, keepdim=True) #b,a,d
        logits = torch.einsum('bd,bad->ba', [vid_embs_attn, cls_embs])
        logit_scale = self.logit_scale.exp()
        logits = logits * logit_scale
        return logits



    def forward(self, images, texts):
        # images shape (b*t,c,h,w)  texts shape (a,77)
        img_embs, img_embs_p = self.encode_image(images)  # b*t,d    b*t,N,768
        vid_embs = self.encode_video(img_embs) # b,t,d

        img_embs_p = self.prompts_visual_ln(img_embs_p) #b*t,N,768
        img_embs_p = img_embs_p @ self.prompts_visual_proj # b*t,N,512
        img_embs_p = img_embs_p.view(vid_embs.shape[0], vid_embs.shape[1], -1, img_embs_p.shape[-1]) # b,t,N,d
        img_embs_p = img_embs_p.mean(dim=1, keepdim=False) # b,N,d

        cls_embs, word_embs = self.encode_text(texts, img_embs_p) # b,a,d  a,77,d
        if self.vcs:
            logits = self.get_logits_vcs(word_embs, cls_embs, vid_embs)
        else:
            logits = self.get_logits(vid_embs, cls_embs)
        return logits


def build_model(device, state_dict: dict, T: int, num_prompt:int, droppath=0., logger=None, pretrained=None):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round(
            (state_dict["visual.positional_embedding"].shape[0]-1)**0.5)
        image_resolution = vision_patch_size*grid_size
    else:
        count: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(
            f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)

        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round(
            (state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + \
            1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(
        k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))


    model = BiEfficient(
        device,
        embed_dim,
        image_resolution, vision_patch_size, vision_width, vision_layers,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
        n_frames=T, num_prompt=num_prompt, droppath=droppath, pretrained=pretrained)

    # for key in ["input_resolution", "context_length", "vocab_size"]:
    #     if key in state_dict:
    #         del state_dict[key]

    msg = model.load_state_dict(state_dict, strict=False)
    # logger.info(f"load pretrained CLIP: {msg}")

    return model.eval()


def load(device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
         T=8, num_prompt=16, droppath=0., logger=None, pretrained=None, jit=True):

    model_path = _download(_MODELS[pretrained]) # '../CLIP-models/ViT-B-16.pt'
    try:
        # loading JIT archive
        model = torch.jit.load( # VisualTransformer Transformer
            model_path, map_location=device if jit else "cpu").eval()

        trained_state_dict = model.state_dict().copy()
        state_dict=None
    except RuntimeError:
        # loading saved state dict
        if jit:
            warnings.warn(
                f"File {model_path} is not a JIT archive. Loading as a state dict instead")
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")

    model = build_model(device, state_dict or model.state_dict(), T=T, num_prompt=num_prompt,
                        droppath=droppath, logger=logger, pretrained=pretrained)

    if str(device) == "cpu":
        model.float()
    return model , trained_state_dict
