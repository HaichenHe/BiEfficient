import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
import time
from utils.utils import init_distributed_mode, AverageMeter, reduce_tensor, accuracy
# from clip.clip import _download, _MODELS
from clip import clip

import yaml
from dotmap import DotMap
from datasets.video import Video_dataset
from datasets.transforms import GroupScale, GroupCenterCrop, Stack, ToTorchFormatTensor, GroupNormalize, GroupOverSample, GroupFullResSample
from modules.text_prompt import text_prompt


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='global config file')
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local-rank", type=int,
                        help='local rank for DistributedDataParallel')
    parser.add_argument(
        "--precision",
        choices=["amp", "fp16", "fp32"],
        default="amp",
        help="Floating point precition."
    )
    parser.add_argument('--test_crops', type=int, default=1)   
    parser.add_argument('--test_clips', type=int, default=3) 
    parser.add_argument('--dense', default=False, action="store_true",
                    help='use dense sample for test as in Non-local I3D')
    args = parser.parse_args()
    return args

def main(args):
    init_distributed_mode(args)

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = DotMap(config)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        cudnn.benchmark = True
    
    # get model
    clip_model, _ = clip.load(config.network.arch, device=device, jit=False, internal_modeling=False, joint_st=True, T=config.data.num_segments, dropout=config.network.drop_out, emb_dropout=config.network.emb_dropout, pretrain=True)
    clip_model = clip_model.eval()

    # model_path = _download(_MODELS['ViT-B/16'])
    # jit=True
    # model = torch.jit.load( 
    #         model_path, map_location=device if jit else "cpu").eval()
    num_param = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
    num_total_param = sum(p.numel() for p in clip_model.parameters())

    print("===================num_param================", num_param//1024//1024)
    
    
    if args.precision == "amp" or args.precision == "fp32":
        clip_model = clip_model.float()
    
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]

    # rescale size
    if 'something' in config.data.dataset:
        scale_size = (240, 320)
    else:
        scale_size = 256 if config.data.input_size == 224 else config.data.input_size

    # crop size
    input_size = config.data.input_size

    # control the spatial crop
    if args.test_crops == 1: # one crop
        cropping = torchvision.transforms.Compose([
            GroupScale(scale_size),
            GroupCenterCrop(input_size),
        ])
    elif args.test_crops == 3:  # do not flip, so only 3 crops (left right center)
        cropping = torchvision.transforms.Compose([
            GroupFullResSample(
                crop_size=input_size,
                scale_size=scale_size,
                flip=False)
        ])
    elif args.test_crops == 5:  # do not flip, so only 5 crops
        cropping = torchvision.transforms.Compose([
            GroupOverSample(
                crop_size=input_size,
                scale_size=scale_size,
                flip=False)
        ])
    elif args.test_crops == 10:
        cropping = torchvision.transforms.Compose([
            GroupOverSample(
                crop_size=input_size,
                scale_size=scale_size,
            )
        ])
    else:
        raise ValueError("Only 1, 3, 5, 10 crops are supported while we got {}".format(args.test_crops))
    
    val_data = Video_dataset(
        config.data.val_root, config.data.val_list, config.data.label_list,
        random_shift=False, num_segments=config.data.num_segments,
        modality=config.data.modality,
        image_tmpl=config.data.image_tmpl,
        test_mode=True,
        transform=torchvision.transforms.Compose([
            cropping,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            GroupNormalize(input_mean, input_std),
        ]),
        dense_sample=args.dense,
        test_clips=args.test_clips)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
    val_loader = DataLoader(
        val_data,
        batch_size=config.data.batch_size,
        num_workers=config.data.workers,
        sampler=val_sampler,
        pin_memory=True, drop_last=False)
    

    clip_model = DistributedDataParallel(clip_model.cuda(), device_ids=[args.gpu], find_unused_parameters=True)

    classes = text_prompt(val_data) # n_cls,77
    n_class = classes.size(0)

    prec1 = validate(
        val_loader, classes, device, clip_model, config, args.test_crops, args.test_clips)
    
    return

def validate(val_loader, classes, device, model, config, test_crops, test_clips):
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()


    with torch.no_grad():
        texts_input = classes.to(device) # 400,77
        n_class = classes.shape[0]
        for i, (image, list_id) in enumerate(val_loader):
            batch_size = list_id.numel()
            num_crop = test_clips*test_crops
            num_segments = config.data.num_segments
            images = image.view(
                    (-1, num_segments, 3)+image.size()[-2:])
            b, t, c, h, w = images.size()

            images_input = images.to(device).view(-1, c, h, w)

            list_id = list_id.to(device)

            # image_feats = model.encode_image(images_input) #b*t,d
            # cls_feat = model.encode_text(texts_input, reduce_token=False) #a,d
            image_feats, cls_feat, _, logit_scale = model(images_input, texts_input, return_token=False)

            image_feats = image_feats.view(b,t,-1) #b,t,d
            video_feats = image_feats.mean(dim=1, keepdim=False) #b,d
            video_feats = video_feats/video_feats.norm(dim=1, keepdim=True)
            cls_feat = cls_feat/cls_feat.norm(dim=1, keepdim=True)

            logits = video_feats@cls_feat.t() #b,a
            logits = logits * logit_scale


            # logits = model(images_input, texts_input)
            similarity = logits.softmax(dim=-1)
            similarity = similarity.reshape(batch_size, num_crop, -1).mean(dim=1)
            similarity = similarity.view(batch_size, -1, n_class).softmax(dim=-1)
            similarity = similarity.mean(dim=1, keepdim=False)

            prec = accuracy(similarity, list_id, topk=(1, 5))
            prec1 = reduce_tensor(prec[0])
            prec5 = reduce_tensor(prec[1])

            top1.update(prec1.item(), list_id.size(0))
            top5.update(prec5.item(), list_id.size(0))

            if i % config.logging.print_freq == 0 and dist.get_rank() == 0:
                print(
                    ('Test: [{0}/{1}] \t'
                     'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                     'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), top1=top1, top5=top5)))

    if dist.get_rank() == 0:
        print('-----Evaluation is finished------')
        print('Overall Prec@1 {:.03f}% Prec@5 {:.03f}%'.format(top1.avg, top5.avg))

    return top1.avg

if __name__ == '__main__':
    args = get_parser()
    main(args)




