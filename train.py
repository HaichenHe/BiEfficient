import os
import sys
import time
import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
import torchvision
# from livelossplot import PlotLosses
from pathlib import Path
import yaml
import pprint
from dotmap import DotMap
import numpy as np
import datetime
import shutil
from contextlib import suppress

from utils.NCELoss import NCELoss, DualLoss
from utils.Augmentation import get_augmentation
from utils.solver import _optimizer, _lr_scheduler
from utils.utils import init_distributed_mode, epoch_saving, best_saving, AverageMeter, reduce_tensor, accuracy, create_logits, gen_label, gather_labels
from utils.logger import setup_logger

from datasets.video import Video_dataset

import clip

from modules import biefficient
from modules.text_prompt import text_prompt


def update_dict(dict):
    new_dict = {}
    for k, v in dict.items():
        new_dict[k.replace('module.', '')] = v
    return new_dict


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', type=str,
                        default='clip.yaml', help='global config file')
    parser.add_argument('--log_time', default='001')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local-rank", type=int,
                        help='local rank for DistributedDataParallel')
    parser.add_argument("--precision",
                        choices=["amp", "fp16", "fp32"],
                        default="amp",
                        help="Floating point precition."
                        )
    parser.add_argument('--auto_augment',
                        type=str,
                        help='enable RandAugment of a certain configuration. see the examples in the SSv2 training scripts.'
                        )
    parser.add_argument('--sampling_rate',
                        type=int,
                        default=16,
                        help='interval between sampled frames. 0 means frames evenly covers the whole video '
                        '(i.e., with variable frame interval depending on the video length).)'
                        )
    parser.add_argument('--resize_type',
                        type=str,
                        default='random_short_side_scale_jitter',
                        choices=['random_resized_crop',
                                 'random_short_side_scale_jitter'],
                        help='spatial resize type. supported modes are "random_resized_crop" and "random_short_side_scale_jitter".'
                        'see implementation in video_dataset/transform.py for the details.'
                        )
    parser.add_argument('--scale_range',
                        type=float,
                        nargs=2,
                        default=[1.0, 1.15],
                        help='range of spatial random resize. for random_resized_crop, the range limits the portion of the cropped area; '
                        'for random_short_side_scale_jitter, the range limits the target short side (as the multiple of --spatial_size).'
                        )
    args = parser.parse_args()

    return args


def main(args):
    global best_prec1
    """ Training Program """
    init_distributed_mode(args)
    if args.distributed:
        print('[INFO] turn on distributed train', flush=True)
    else:
        print('[INFO] turn off distributed train', flush=True)

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    working_dir = os.path.join(config['data']['output_path'], config['data']
                               ['dataset'], config['network']['arch'], args.log_time)

    if dist.get_rank() == 0:
        Path(working_dir).mkdir(parents=True, exist_ok=True)
        shutil.copy(args.config, working_dir)
        shutil.copy('train.py', working_dir)

    # build logger, print env and config
    logger = setup_logger(output=working_dir,
                          distributed_rank=dist.get_rank(),
                          name=f'BiEfficient')
    logger.info("------------------------------------")
    logger.info("Environment Versions:")
    logger.info("- Python: {}".format(sys.version))
    logger.info("- PyTorch: {}".format(torch.__version__))
    logger.info("- TorchVison: {}".format(torchvision.__version__))
    logger.info("------------------------------------")
    pp = pprint.PrettyPrinter(indent=4)
    logger.info(pp.pformat(config))
    logger.info("------------------------------------")
    logger.info("storing name: {}".format(working_dir))

    config = DotMap(config)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        cudnn.benchmark = True

    seed = config.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

 
    model, clip_state_dict = biefficient.load(device="cpu", T=config.data.num_segments, num_prompt=config.data.num_prompt, droppath=config.network.drop_out,
                                              logger=logger, pretrained=config.network.arch)

    model = model.cuda()

    transform_train = get_augmentation(True, config)
    transform_val = get_augmentation(False, config)

    logger.info('train transforms: {}'.format(transform_train.transforms))
    logger.info('val transforms: {}'.format(transform_val.transforms))


    train_data = Video_dataset(
            config.data.train_root, config.data.train_list,
            config.data.label_list, num_segments=config.data.num_segments,
            modality=config.data.modality,
            image_tmpl=config.data.image_tmpl, random_shift=config.data.random_shift,
            transform=transform_train, dense_sample=config.data.dense)
    
    val_data = Video_dataset(
            config.data.val_root, config.data.val_list, config.data.label_list,
            random_shift=False, num_segments=config.data.num_segments,
            modality=config.data.modality,
            image_tmpl=config.data.image_tmpl,
            transform=transform_val, dense_sample=config.data.dense)


    ################ Few shot data for training ###########
    if config.data.shot:
        cls_dict = {}
        for item in train_data.video_list:
            if item.label not in cls_dict:
                cls_dict[item.label] = [item]
            else:
                cls_dict[item.label].append(item)
        import random
        select_vids = []
        K = config.data.shot
        for category, v in cls_dict.items():
            slice = random.sample(v, K)
            select_vids.extend(slice)
        n_repeat = len(train_data.video_list) // len(select_vids)
        train_data.video_list = select_vids * n_repeat
        # print('########### number of videos: {} #########'.format(len(select_vids)))
    ########################################################

    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    train_loader = DataLoader(train_data,
                              batch_size=config.data.batch_size, num_workers=config.data.workers,
                              sampler=train_sampler, drop_last=True, pin_memory=True, persistent_workers=True)

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_data, shuffle=False)
    val_loader = DataLoader(val_data,
                            batch_size=config.data.batch_size, num_workers=config.data.workers,
                            sampler=val_sampler, drop_last=False, pin_memory=True, persistent_workers=True)




    criterion = nn.CrossEntropyLoss()

    start_epoch = config.solver.start_epoch

    if config.pretrain:
        if os.path.isfile(config.pretrain):
            logger.info("=> loading checkpoint '{}'".format(config.pretrain))
            checkpoint = torch.load(config.pretrain, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            del checkpoint
        else:
            logger.info("=> no checkpoint found at '{}'".format(config.resume))

    if config.resume:
        if os.path.isfile(config.resume):
            logger.info("=> loading checkpoint '{}'".format(config.resume))
            checkpoint = torch.load(config.resume, map_location='cpu')
            model.load_state_dict(update_dict(checkpoint['model_state_dict']))
            start_epoch = checkpoint['epoch'] + 1
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(config.evaluate, checkpoint['epoch']))
            del checkpoint
        else:
            logger.info(
                "=> no checkpoint found at '{}'".format(config.pretrain))



    ## freezing some parameters##
    for name, param in model.visual.named_parameters():
        param.requires_grad = False
        if 'S_Adapter' in name:
            param.requires_grad = True
    for name, param in model.transformer.named_parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if 'positional_embedding' in name and 'visual' not in name:
            param.requires_grad = False
        if 'text_projection' in name or 'token_embedding' in name or 'ln_final' in name:
            param.requires_grad = False


    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total_param = sum(p.numel() for p in model.parameters())

    logger.info('Number of total parameters: {}M, tunable parameters: {}M'.format(
        (num_total_param/1024/1024), (num_param/1024/1024)))

    

    optimizer = _optimizer(config, model)
    lr_scheduler = _lr_scheduler(config, optimizer)

    if args.distributed:
        model = DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)

    scaler = GradScaler() if args.precision == "amp" else None

    # classes = text_prompt(train_data)
    # n_class = classes.size(0)


    classes = []
    for i, c in train_data.classes:
        classes.append(c) # 400

    best_prec1 = 0.0

    # if config.solver.evaluate:
    #     logger.info(("===========evaluate==========="))
    #     prec1 = validate(
    #         start_epoch, val_loader, classes, device, model, config, logger)
    #     return

  

    for epoch in range(start_epoch, config.solver.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        train(model, train_loader, optimizer, criterion, scaler,
              epoch, device, lr_scheduler, config, classes, logger)

        if (epoch+1) % config.logging.eval_freq == 0:
            prec1 = validate(
                epoch, val_loader, classes, device, model, config, logger)

            if dist.get_rank() == 0:
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
                logger.info('Testing: {}/{}'.format(prec1, best_prec1))
                logger.info('Saving:')
                filename = "{}/last_model.pt".format(working_dir)

                epoch_saving(epoch, model.module, optimizer, filename)
                if is_best:
                    best_saving(working_dir, epoch, model.module, optimizer)


def train(model, train_loader, optimizer, criterion, scaler, epoch, device, lr_scheduler, config, classes, logger):
    """ train an epoch"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress  
    end = time.time()

    for i, (images, list_id) in enumerate(train_loader):
        # print("\n======================start iteration!=====================")
        # images b,t*c,h,w
        if config.solver.type != 'monitor':
            if (i + 1) == 1 or (i + 1) % 10 == 0:
                lr_scheduler.step(epoch + i / len(train_loader))

        data_time.update(time.time() - end)

        images = images.view(
            (-1, config.data.num_segments, 3)+images.size()[-2:])
        b, t, c, h, w = images.size()
        images = images.view(-1, c, h, w)     # images重塑为(b*t,c,h,w)
        # texts = classes  # tokenize之后，还需在video2text里进行embedding
        texts = []
        for id in list_id:
            texts.append(classes[id]) 


        with autocast():
            # texts = texts[list_id]  # b, 77
            logits = model(images, texts)

            list_id = list_id.to(device)

            ground_truth = torch.tensor(gen_label(list_id)).to(device)

            loss_img = criterion(logits, ground_truth)
            loss_text = criterion(logits.T, ground_truth)
            loss = (loss_img+loss_text)/2
            loss = loss/config.solver.grad_accumulation_steps
        
        

        if scaler is not None:
            # back propagation
            scaler.scale(loss).backward()
            if (i+1) % config.solver.grad_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            # back propagation
            loss.backward()
            if (i + 1) % config.solver.grad_accumulation_steps == 0:
                optimizer.step()  # update param
                optimizer.zero_grad()  # reset gradient

        losses.update(loss.item(), logits.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        cur_iter = epoch * len(train_loader) + i
        max_iter = config.solver.epochs * len(train_loader)
        eta_sec = batch_time.avg * (max_iter - cur_iter + 1)
        eta_sec = str(datetime.timedelta(seconds=int(eta_sec)))

        if i % config.logging.print_freq == 0:
            logger.info(('Epoch: [{0}][{1}/{2}], lr: {lr:.2e}, eta: {3}\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                             epoch, i, len(train_loader), eta_sec, batch_time=batch_time, data_time=data_time, loss=losses,
                             lr=optimizer.param_groups[-1]['lr'])))


def validate(epoch, val_loader, classes, device, model, config, logger):
    top1 = AverageMeter()
    top5 = AverageMeter()
    sims_list = []
    labels_list = []

    model.eval()

    with torch.no_grad():
        # text_input = classes.to(device)  # 400,77
        texts = []
        for i in range(len(classes)):
            texts.append(classes[i])
        for i, (image, class_id) in enumerate(val_loader):
            #b, t*c, h, w
            image = image.view(
                (-1, config.data.num_segments, 3) + image.size()[-2:])
            b, t, c, h, w = image.size()
            image_input = image.to(device).view(-1, c, h, w)

            class_id = class_id.to(device)

            similarity = model(image_input, texts)
            similarity = similarity.softmax(dim=-1)

            prec = accuracy(similarity, class_id, topk=(1, 5))
            prec1 = reduce_tensor(prec[0])
            prec5 = reduce_tensor(prec[1])

            top1.update(prec1.item(), class_id.size(0))
            top5.update(prec5.item(), class_id.size(0))

            if i % config.logging.print_freq == 0:
                logger.info(
                    ('Test: [{0}/{1}]\t'
                     'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                     'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                         i, len(val_loader), top1=top1, top5=top5)))
    logger.info(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                 .format(top1=top1, top5=top5)))

    return top1.avg


if __name__ == '__main__':
    args = get_parser()
    main(args)
