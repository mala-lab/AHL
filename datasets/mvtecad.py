import numpy as np
import os, sys
from datasets.base_dataset import BaseADDataset
from PIL import Image
from torchvision import transforms
from datasets.cutpaste import CutPaste
import random
import math

class MVTecAD(BaseADDataset):
    def __init__(self, args, train = True, AHL = False, sp = False, k = 0, cluster_index = None):
        super(MVTecAD).__init__()
        self.args = args
        self.train = train
        self.classname = self.args.classname
        self.feat_classname = self.args.feat_classname
        self.know_class = self.args.know_class
        self.exp_root = self.args.experiment_dir
        self.pollution_rate = self.args.cont_rate
        if self.args.test_threshold == 0 and self.args.test_rate == 0:
            self.test_threshold = self.args.nAnomaly
        else:
            self.test_threshold = self.args.test_threshold

        self.root = os.path.join(self.args.dataset_root, self.classname)
        self.feat_root = os.path.join(self.args.dataset_root, self.feat_classname)
        self.transform = self.transform_train() if self.train else self.transform_test()
        self.transform_pseudo = self.transform_pseudo()

        self.AHL = AHL
        self.sp = sp    # construct support sets or query set
        self.k = k
        self.cluster_index = cluster_index

        normal_data = list()
        split = 'train'
        normal_files = os.listdir(os.path.join(self.root, split, 'good'))
        for file in normal_files:
            if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
                normal_data.append(split + '/good/' + file)

        self.nPollution = int((len(normal_data)/(1-self.pollution_rate)) * self.pollution_rate)
        if self.test_threshold==0 and self.args.test_rate>0:
            self.test_threshold = int((len(normal_data)/(1-self.args.test_rate)) * self.args.test_rate) + self.args.nAnomaly
        self.ood_data = self.get_ood_data()

        if self.train is False:
            normal_data = list()
            split = 'test'
            normal_files = os.listdir(os.path.join(self.root, split, 'good'))
            for file in normal_files:
                if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
                    normal_data.append(split + '/good/' + file)

        outlier_data, pollution_data = self.split_outlier()
        outlier_data.sort()

        normal_data = normal_data + pollution_data

        if self.train is True and self.AHL is True:
            cluster = list()
            for i in range(len(self.cluster_index)):
                item = list()
                if len(self.cluster_index[i]) <= 2:
                    continue
                for j in range(len(self.cluster_index[i])):
                    item.append(normal_data[j])
                cluster.append(item)

            self.args.cluster_num=len(cluster)

            item1 = list()
            item2 = list()
            for i in range(self.args.cluster_num-1):
                for j in range(i+1, self.args.cluster_num):
                    item1.append(cluster[i])
                    item2.append(cluster[j])

            sp_normal = item1 + item2
            qp_normal = item2 + item1

            np.random.RandomState(self.args.ramdn_seed).shuffle(normal_data)
            s_t = normal_data[0:int(len(normal_data)/2)]
            q_t = normal_data[int(len(normal_data)/2):-1]
            sp_normal.append(s_t)
            qp_normal.append(q_t)

            if self.args.nAnomaly > 1:
                sp_outlier = []
                qp_outlier = []

                for i in range(len(sp_normal)):
                    item1 = random.sample(outlier_data, int(len(outlier_data) / 2))
                    item2 = list()
                    for k in outlier_data:
                        if k not in item1:
                            item2.append(k)
                    m = item1
                    n = random.sample(item1, 2)
                    n = n + item2

                    sp_outlier.append(m)
                    qp_outlier.append(n)
            else:
                sp_outlier = []
                qp_outlier = []
                for i in range(len(sp_normal)):
                    sp_outlier.append(outlier_data)
                    qp_outlier.append(outlier_data)

            if self.sp is True:
                normal_data = sp_normal[self.k]
                outlier_data = sp_outlier[self.k]
            else:
                normal_data = qp_normal[self.k]
                outlier_data = qp_outlier[self.k]

        normal_label = np.zeros(len(normal_data)).tolist()
        outlier_label = np.ones(len(outlier_data)).tolist()

        self.images = normal_data + outlier_data
        self.labels = np.array(normal_label + outlier_label)
        self.normal_idx = np.argwhere(self.labels == 0).flatten()
        self.outlier_idx = np.argwhere(self.labels == 1).flatten()

    def get_ood_data(self):
        ood_data = list()
        if self.args.outlier_root is None:
            return None
        dataset_classes = os.listdir(self.args.outlier_root)
        for cl in dataset_classes:
            if cl == self.args.classname:
                continue
            cl_root = os.path.join(self.args.outlier_root, cl, 'train', 'good')
            ood_file = os.listdir(cl_root)
            for file in ood_file:
                if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
                    ood_data.append(os.path.join(cl_root, file))
        return ood_data

    def split_outlier(self):
        outlier_data_dir = os.path.join(self.root, 'test')
        outlier_classes = os.listdir(outlier_data_dir)
        if self.know_class in outlier_classes:
            print("Know outlier class: " + self.know_class)
            outlier_data = list()
            know_class_data = list()
            for cl in outlier_classes:
                if cl == 'good':
                    continue
                outlier_file = os.listdir(os.path.join(outlier_data_dir, cl))
                for file in outlier_file:
                    if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
                        if cl == self.know_class:
                            know_class_data.append('test/' + cl + '/' + file)
                        else:
                            outlier_data.append('test/' + cl + '/' + file)
            np.random.RandomState(self.args.ramdn_seed).shuffle(know_class_data)
            know_outlier = know_class_data[0:self.args.nAnomaly]
            unknow_outlier = outlier_data
            if self.train:
                return know_outlier, list()
            else:
                return unknow_outlier, list()

        outlier_data = list()
        for cl in outlier_classes:
            if cl == 'good':
                continue
            outlier_file = os.listdir(os.path.join(outlier_data_dir, cl))
            for file in outlier_file:
                if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
                    outlier_data.append('test/' + cl + '/' + file)
        np.random.RandomState(self.args.ramdn_seed).shuffle(outlier_data)
        seen_outliers = outlier_data[0:self.args.nAnomaly]
        unseen_outliers = outlier_data[self.test_threshold:]

        if self.train:
            return seen_outliers, outlier_data[self.args.nAnomaly:self.args.nAnomaly + self.nPollution]
        else:
            return unseen_outliers, list()

    def load_image(self, path):
        if 'npy' in path[-3:]:
            if self.args.extract == False:
                img = np.load(path).astype(np.uint8)
                img = img[:, :, :3]
                return Image.fromarray(img)
            else:
                img = np.load(path)
                return img
        return Image.open(path).convert('RGB')

    def transform_train(self):
        composed_transforms = transforms.Compose([
            transforms.Resize((self.args.img_size, self.args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        return composed_transforms

    def transform_pseudo(self):
        composed_transforms = transforms.Compose([
            transforms.Resize((self.args.img_size,self.args.img_size)),
            CutPaste(),
            transforms.RandomRotation(180),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        return composed_transforms

    def transform_test(self):
        composed_transforms = transforms.Compose([
            transforms.Resize((self.args.img_size, self.args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        return composed_transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if self.args.model_name == "DRA":
            rnd = random.randint(0, 1)
            file_name = os.path.splitext(self.images[index])[0]

            if index in self.normal_idx and rnd == 0 and self.train:
                if self.ood_data is None:
                    index = random.choice(self.normal_idx)
                    if self.args.extract == False:
                        image = self.load_image(os.path.join(self.root, self.images[index]))
                        transform = self.transform_pseudo
                        label = 2
                    else:
                        if self.args.aug_task == True:
                            image = self.load_image(os.path.join(self.feat_root, "feature/" + file_name + ".npy"))
                            image_scale = self.load_image(os.path.join(self.feat_root, "feature_scale/" + file_name + ".npy"))
                            label = self.labels[index]
                        else:
                            image = self.load_image(os.path.join(self.feat_root, "aug/" + file_name + ".npy"))
                            image_scale = self.load_image(os.path.join(self.feat_root, "aug_scale/" + file_name + ".npy"))
                            label = 2
                else:
                    if self.args.extract == False:
                        image = self.load_image(random.choice(self.ood_data))
                        transform = self.transform
                    else:
                        image = self.load_image(os.path.join(self.feat_root, "feature/" + file_name + ".npy"))
                        image_scale = self.load_image(os.path.join(self.feat_root, "feature_scale/" + file_name + ".npy"))
                    label = 2
            else:
                if self.args.extract == False:
                    image = self.load_image(os.path.join(self.root, self.images[index]))
                    transform = self.transform
                else:
                    image = self.load_image(os.path.join(self.feat_root, "feature/" + file_name + ".npy"))
                    image_scale = self.load_image(os.path.join(self.feat_root, "feature_scale/" + file_name + ".npy"))
                label = self.labels[index]

            if self.args.extract == False:
                sample = {'image': transform(image), 'label': label}
            else:
                sample = {'image': image, 'image_scale': image_scale, 'label': label, "fn": file_name}
            return sample

        else:
            file_name = os.path.splitext(self.images[index])[0]
            if self.args.extract is False:
                transform = self.transform
                image = self.load_image(os.path.join(self.root, self.images[index]))
                sample = {'image': transform(image), 'label': self.labels[index]}
            else:
                image = self.load_image(os.path.join(self.feat_root, "dev/feature/" + file_name + ".npy"))
                image_scale = self.load_image(os.path.join(self.feat_root, "dev/feature_scale/" + file_name + ".npy"))
                sample = {'image': image, 'image_scale': image_scale, 'label': self.labels[index], "fn": file_name}
            return sample