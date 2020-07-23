import os
import numpy as np
import pickle
from PIL import Image

import miscc.inception_score_tf as inception_score
from miscc.utils import compute_inception_score, negative_log_posterior_probability
from miscc.utils import calculate_activation_statistics, calculate_frechet_distance

split = 'unseen'
gen_images_dir = f'/data/seth/MirrorGAN/output/comp_coco_MirrorGAN_2020_07_14_23_30_40/Model/netG_epoch_100/test_{split}/single'
gen_img_filenames = os.listdir(gen_images_dir)

gt_images_dir = '/home/seth/code/MirrorGAN/data/coco/images/val2014'
gt_images_pickle = f'/home/seth/code/MirrorGAN/data/coco/comp_test/{split}_filenames.pickle'
gt_image_ids = list(pickle.load(open(gt_images_pickle, 'rb')))

imgs_np = []
for img_filename in gen_img_filenames:
    img_path = os.path.join(gen_images_dir, img_filename)
    img = np.array(Image.open(img_path).convert("RGB")).transpose((2, 0, 1))
    imgs_np.append(img)
imgs_np = np.stack(imgs_np)

# compute inception score
pred = inception_score.get_inception_pred(imgs_np)
mean, std = compute_inception_score(pred, min(10, 64))
mean_conf, std_conf = negative_log_posterior_probability(pred, min(10, 64))

# gather activations for fake images and compute statistics
fake_acts = inception_score.get_fid_pred(imgs_np)
fake_mu, fake_sigma = calculate_activation_statistics(fake_acts)

img_size = imgs_np.shape[2:]

# load saved statistics or gather activations for real images
real_stats_dir = '/home/seth/code/MirrorGAN/data/coco/fid_score/'
real_mean_path = os.path.join(real_stats_dir, f'{split}_real_mean.npy')
real_sigma_path = os.path.join(real_stats_dir, f'{split}_real_sigma.npy')
if os.path.exists(real_mean_path) and os.path.exists(real_sigma_path):
    real_mu = np.load(real_mean_path)
    real_sigma = np.load(real_sigma_path)
else:
    imgs_np = []
    for img_id in gt_image_ids:
        img_path = os.path.join(gt_images_dir, img_id + '.jpg')
        pil_img = Image.open(img_path).convert("RGB").resize(img_size)
        img = np.array(pil_img).transpose((2, 0, 1))
        imgs_np.append(img)
    imgs_np = np.stack(imgs_np)
    real_acts = inception_score.get_fid_pred(imgs_np)
    real_mu, real_sigma = calculate_activation_statistics(real_acts)
    np.save(real_mean_path, real_mu)
    np.save(real_sigma_path, real_sigma)

# compute FID score
fid_score = calculate_frechet_distance(real_mu, real_sigma, fake_mu, fake_sigma)

print(f'Result for {gen_images_dir} on split {split}')
print(mean, std, mean_conf, std_conf)
print(fid_score)

