import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import random
import torchvision.transforms as transforms
import pprint
from PIL import Image
from torch.nn.utils.rnn import pack_padded_sequence
import torch.backends.cudnn as cudnn  

from cfg.config import cfg, cfg_from_file                                                                           

from datasets import TextDataset
from datasets import prepare_data

from model import CAPTION_RNN, CAPTION_CNN 

def parse_args():                                                                                                      
    parser = argparse.ArgumentParser(description='Train a STREAM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/STREAM/train_bird.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    parser.add_argument('--model_dir', dest='model_dir', type=str, default='')
    parser.add_argument('--output_dir', dest='output_dir', type=str, default='')
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
    
    torch.cuda.set_device(cfg.GPU_ID)
    cudnn.benchmark = True
    batch_size = 1

    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
    ])

    norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        )
    ])
    unnorm = transforms.Normalize(
        (-0.485/0.229, -0.456/0.229, -0.406/0.229),
        (1/0.229, 1/0.229, 1/0.229)
    )

    dataset = TextDataset(cfg.DATA_DIR, 'test',
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=image_transform,
                          norm=norm)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, drop_last=True,
        shuffle=False, num_workers=int(cfg.WORKERS))

    encoder_path = os.path.join(args.model_dir, 'encoder-9.ckpt')
    decoder_path = os.path.join(args.model_dir, 'decoder-9.ckpt')
    encoder = CAPTION_CNN(cfg.CAP.embed_size).cuda()
    decoder = CAPTION_RNN(
        cfg.CAP.embed_size,
        cfg.CAP.hidden_size * 2,
        dataset.n_words,
        cfg.CAP.num_layers
    ).cuda()
    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))
    encoder.eval()
    decoder.eval()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for i, data in enumerate(dataloader):
        imgs, captions, cap_lens, class_ids, keys = prepare_data(data)
        #targets = pack_padded_sequence(captions.unsqueeze(0), cap_lens, batch_first=True)[0]
        with torch.no_grad():
            features = encoder(imgs[-1])
            sampled_ids = decoder.sample(features)
        sampled_ids = sampled_ids[0].cpu().numpy()
        sampled_captions = []
        for word_id in sampled_ids:
            word = dataset.ixtoword[word_id]
            sampled_captions.append(word)
            if word == '<end>':
                break
        sentence = ' '.join(sampled_captions)
        original_img = unnorm(imgs[-1].squeeze()).cpu().numpy().transpose((1, 2, 0))
        fig = plt.figure()
        plt.imshow(original_img)
        plt.title(sentence)
        #ax1 = fig.add_axes((0.1,0.4,0.8,0.5))
        #ax1.set_title(sentence)
        directory, filename = keys[0].split('/')
        if not os.path.exists(os.path.join(args.output_dir, directory)):
            os.makedirs(os.path.join(args.output_dir, directory))
        plt.savefig(os.path.join(args.output_dir, directory, filename))

