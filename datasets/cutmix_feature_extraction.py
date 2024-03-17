import random

import numpy as np
import os
from PIL import Image
from torchvision import transforms
from datasets.cutmix import CutMix
import argparse
import torch
from modeling.net import SemiADNet
import cv2

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def transform_pseudo(tem_raw_image):
    composed_transforms = transforms.Compose([
        transforms.Resize((448, 448)),
        CutMix(a_img=tem_raw_image),
        transforms.RandomRotation(180),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return composed_transforms

def load_image(path):
    if 'npy' in path[-3:]:
        img = np.load(path).astype(np.uint8)
        img = img[:, :, :3]
        return Image.fromarray(img)
    return Image.open(path).convert('RGB')

def save_feature(args, model):
    root = os.path.join(args.dataset_root, args.classname)

    o_train_dir = os.path.join(root, "train/good")
    train_dir = os.path.join(root, "aug_mix/train/good")
    train_dir_scale = os.path.join(root, "aug_mix_scale/train/good")
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(train_dir_scale):
        os.makedirs(train_dir_scale)
    normal_files = os.listdir(o_train_dir)

    for file in normal_files:
        t = random.sample(normal_files, 1)
        tem = os.path.join(o_train_dir, t[0])
        tem_file_name = os.path.splitext(tem)[0]
        tem_raw_image = load_image(tem)
        transform2 = transform_pseudo(tem_raw_image)

        f = os.path.join(o_train_dir, file)
        file_name = os.path.splitext(file)[0]
        raw_image = load_image(f)
        image = transform2(raw_image)

        feature, feature_scale = model(image = image.unsqueeze(0), flag = True)
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
    parser.add_argument('--meta', type=bool, default=False, help="")
    parser.add_argument('--episode_num', type=int, default=2, help="number of episodes")
    parser.add_argument('--extract', type=bool, default=False, help="")
    parser.add_argument('--save_feature', type=bool, default=True, help="")
    parser.add_argument('--update_step', type=int, default=3, help="")
    parser.add_argument('--ta', type=bool, default=False, help="")
    args = parser.parse_args()

    model = SemiADNet(args)
    path = args.experiment_dir +'/'+args.classname+ str(args.ramdn_seed)+".pkl"
    model.load_state_dict(torch.load(path))

    save_feature(args, model)

