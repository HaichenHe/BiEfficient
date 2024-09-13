import torch
# import clip
import sys
from pathlib import Path
# sys.path.append("../")
sys.path.append(str(Path(__file__).resolve().parents[1]))
from clip.clip import tokenize


def text_prompt(data):
    # text_aug = ['{}']
    text_aug = 'This is a video about {}'
    # classes = torch.cat([tokenize(c) for i, c in data.classes])
    classes = torch.cat([tokenize(text_aug.format(c)) for i, c in data.classes])
    return classes