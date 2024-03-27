import numpy as np
import os
from PIL import Image
from torchvision import transforms
import argparse
import torch
from modeling.net import DRA
import cv2
import imgaug.augmenters as iaa
from perlin import rand_perlin_2d_np
import glob
import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def randAugmenter(augmenters):
    aug_ind = np.random.choice(np.arange(len(augmenters)), 3, replace=False)
    aug = iaa.Sequential([augmenters[aug_ind[0]],
                          augmenters[aug_ind[1]],
                          augmenters[aug_ind[2]]]
                         )
    return aug

def augment_image(image, anomaly_source_path):
    augmenters = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                  iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
                  iaa.pillike.EnhanceSharpness(),
                  iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
                  iaa.Solarize(0.5, threshold=(32, 128)),
                  iaa.Posterize(),
                  iaa.Invert(),
                  iaa.pillike.Autocontrast(),
                  iaa.pillike.Equalize(),
                  iaa.Affine(rotate=(-45, 45))
                  ]
    aug = randAugmenter(augmenters)
    perlin_scale = 6
    min_perlin_scale = 0
    anomaly_source_img = cv2.imread(anomaly_source_path)
    anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(448, 448))

    anomaly_img_augmented = aug(image=anomaly_source_img)
    perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
    perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

    perlin_noise = rand_perlin_2d_np((448, 448), (perlin_scalex, perlin_scaley))
    rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

    perlin_noise = rot(image=perlin_noise)
    threshold = 0.5
    perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
    perlin_thr = np.expand_dims(perlin_thr, axis=2)

    img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

    beta = torch.rand(1).numpy()[0] * 0.8

    augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
        perlin_thr)

    augmented_image = augmented_image.astype(np.float32)
    msk = (perlin_thr).astype(np.float32)
    augmented_image = msk * augmented_image + (1 - msk) * image
    has_anomaly = 1.0
    if np.sum(msk) == 0:
        has_anomaly = 0.0
    return augmented_image, msk, np.array([has_anomaly], dtype=np.float32)


def transform_train(image_path, anomaly_source_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, dsize=(448, 448))
    rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

    do_aug_orig = torch.rand(1).numpy()[0] > 0.7
    if do_aug_orig:
        image = rot(image=image)

    image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
    augmented_image, anomaly_mask, has_anomaly = augment_image(image, anomaly_source_path)
    augmented_image = np.transpose(augmented_image, (2, 0, 1))
    image = np.transpose(image, (2, 0, 1))
    anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
    return image, augmented_image, anomaly_mask, has_anomaly

def load_image(path):
    if 'npy' in path[-3:]:
        img = np.load(path).astype(np.uint8)
        img = img[:, :, :3]
        return Image.fromarray(img)
    return Image.open(path).convert('RGB')

def save_feature(args, model):
    root = os.path.join(args.dataset_root, args.classname)
    anomaly_source_paths = sorted(glob.glob("./data/dtd/images/" + "/*/*.jpg"))
    anomaly_source_idx = torch.randint(0, len(anomaly_source_paths), (1,)).item()

    o_train_dir = os.path.join(root, "train/good")
    train_dir = os.path.join(root, "aug_dream/train/good")
    train_dir_scale = os.path.join(root, "aug_dream_scale/train/good")
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(train_dir_scale):
        os.makedirs(train_dir_scale)
    normal_files = os.listdir(o_train_dir)

    for file in normal_files:
        f = os.path.join(o_train_dir, file)
        file_name = os.path.splitext(file)[0]
        image, augmented_image, anomaly_mask, has_anomaly = transform_train(f, anomaly_source_paths[anomaly_source_idx])

        augmented_image = torch.Tensor(augmented_image)
        feature, feature_scale = model(image = augmented_image.unsqueeze(0), extracted = True)
        feature = [item.cpu().detach().numpy() for item in feature]
        feature_scale = [item.cpu().detach().numpy() for item in feature_scale]

        feature = np.array(feature).squeeze()
        feature_scale = np.array(feature_scale).squeeze()


        file_root_1 = os.path.join(train_dir, file_name+".npy")
        file_root_2 = os.path.join(train_dir_scale, file_name+".npy")
        np.save(file_root_1, feature)
        np.save(file_root_2, feature_scale)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=48, help="batch size used in SGD")
    parser.add_argument("--steps_per_epoch", type=int, default=20, help="the number of batches per epoch")
    parser.add_argument("--epochs", type=int, default=20, help="the number of epochs")
    parser.add_argument("--cont_rate", type=float, default=0.0,
                        help="the outlier contamination rate in the training data")
    parser.add_argument("--test_threshold", type=int, default=0,
                        help="the outlier contamination rate in the training data")
    parser.add_argument("--test_rate", type=float, default=0.0,
                        help="the outlier contamination rate in the training data")
    parser.add_argument("--dataset", type=str, default='mvtecad', help="a list of data set names")
    parser.add_argument("--ramdn_seed", type=int, default=42, help="the random seed number")
    parser.add_argument('--workers', type=int, default=4, metavar='N', help='dataloader threads')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--savename', type=str, default='model.pkl', help="save modeling")
    parser.add_argument('--dataset_root', type=str, default='./data/mvtecad', help="dataset root")
    parser.add_argument('--experiment_dir', type=str, default='./experiment/experiment_14', help="dataset root")
    parser.add_argument('--classname', type=str, default='carpet', help="dataset class")
    parser.add_argument('--img_size', type=int, default=448, help="dataset root")
    parser.add_argument("--nAnomaly", type=int, default=10, help="the number of anomaly data in training set")
    parser.add_argument("--n_scales", type=int, default=2, help="number of scales at which features are extracted")
    parser.add_argument('--backbone', type=str, default='resnet18', help="backbone")
    parser.add_argument('--criterion', type=str, default='deviation', help="loss")
    parser.add_argument("--topk", type=float, default=0.1, help="topk in MIL")
    parser.add_argument('--know_class', type=str, default=None, help="set the know class for hard setting")
    parser.add_argument('--pretrain_dir', type=str, default=None, help="root of pretrain weight")
    parser.add_argument("--total_heads", type=int, default=4, help="number of head in training")
    parser.add_argument("--nRef", type=int, default=5, help="number of reference set")
    parser.add_argument('--outlier_root', type=str, default=None, help="OOD dataset root")
    parser.add_argument('--cluster_num', type=int, default=5, help="")
    parser.add_argument('--AHL', type=bool, default=False, help="")
    parser.add_argument('--episode_num', type=int, default=2, help="number of episodes")
    parser.add_argument('--extract', type=bool, default=False, help="")
    parser.add_argument('--save_feature', type=bool, default=True, help="")
    parser.add_argument('--update_step', type=int, default=3, help="")
    parser.add_argument('--ta', type=bool, default=False, help="")
    args = parser.parse_args()

    model = DRA(args, backbone=args.backbone, flag=True)
    path = args.experiment_dir +'/'+args.classname +".pkl"
    model.load_state_dict(torch.load(path))

    save_feature(args, model)

