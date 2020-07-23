from __future__ import print_function

from miscc.utils import mkdir_p
from miscc.utils import build_super_images
from miscc.losses import sent_loss, words_loss
from cfg.config import cfg, cfg_from_file

from datasets import TextDataset, TextDatasetCOCO
from datasets import prepare_data

from model import CAPTION_RNN, CAPTION_CNN

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms


dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


UPDATE_INTERVAL = 200
def parse_args():
    parser = argparse.ArgumentParser(description='Train a STREAM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/STREAM/train_bird.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id == -1:
        cfg.CUDA = False
    else:
        cfg.GPU_ID = args.gpu_id

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    ##########################################################################
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '%s/output/%s_%s_%s' % \
        (cfg.OUTPUT_PATH, cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    model_dir = os.path.join(output_dir, 'Model')
    image_dir = os.path.join(output_dir, 'Image')
    mkdir_p(model_dir)
    mkdir_p(image_dir)

    torch.cuda.set_device(cfg.GPU_ID)
    cudnn.benchmark = True

    # Get data loader ##################################################
    batch_size = cfg.TRAIN.BATCH_SIZE
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))  
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip(), 
    ])
    norm = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        )
    ])

    dataset = TextDataset(cfg.DATA_DIR, 'train',
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=image_transform,
                          norm=norm)
    """
    dataset = TextDatasetCOCO(
        cfg.DATA_DIR, 
        'train', 
        base_size=cfg.TREE.BASE_SIZE,
        transform=image_transform,
        norm=norm
    )
    """

    print(dataset.n_words, dataset.embeddings_num)
    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, drop_last=True,
        shuffle=True, num_workers=int(cfg.WORKERS))

    # # validation data #
    dataset_val = TextDataset(cfg.DATA_DIR, 'test',
                              base_size=cfg.TREE.BASE_SIZE,
                              transform=image_transform, 
                              norm=norm)
    """
    dataset_val = TextDatasetCOCO(
        cfg.DATA_DIR, 
        'test', 
        base_size=cfg.TREE.BASE_SIZE,
        transform=image_transform,
        norm=norm
    )
    """
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=batch_size, drop_last=True,
        shuffle=True, num_workers=int(cfg.WORKERS))

    # Train ##############################################################
    encoder = CAPTION_CNN(cfg.CAP.embed_size).cuda()
    decoder = CAPTION_RNN(
        cfg.CAP.embed_size,
        cfg.CAP.hidden_size * 2,
        dataset.n_words,
        cfg.CAP.num_layers
    ).cuda()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params, lr=cfg.CAP.learning_rate)

    log_step = 10
    save_step = 10 
    num_epochs = 50 

    for epoch in range(num_epochs):
        total_step = len(dataloader)
        for i, data in enumerate(dataloader):
            imgs, captions, cap_lens, class_ids, keys = prepare_data(data)
            targets = pack_padded_sequence(captions, cap_lens, batch_first=True)[0]
            features = encoder(imgs[-1])
            outputs = decoder(features, captions, cap_lens)
            loss = criterion(outputs, targets)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            # Print log info
            if i % log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                .format(epoch, num_epochs, i, total_step, loss.item(), np.exp(loss.item()))) 
                                                
            # Save the model checkpoints
        if (epoch+1) % save_step == 0:
            torch.save(decoder.state_dict(), os.path.join(
                model_dir, 'decoder-{}.ckpt'.format(epoch+1)))
            torch.save(encoder.state_dict(), os.path.join(
                model_dir, 'encoder-{}.ckpt'.format(epoch+1)))

