from datasets import mvtecad
from torch.utils.data import DataLoader
from dataloaders.utlis import worker_init_fn_seed, BalancedBatchSampler
from dataloaders.utlis import worker_init_fn_seed_ref, BalancedBatchSampler_ref
import torch
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
import random
import os
import math

class initDataloader():
    def __init__(self, args):
        self.args = args

    def build(self, ref = None, **kwargs):
        if self.args.dataset == "mvtecad":
            train_set = mvtecad.MVTecAD(self.args, train=True, meta=False)
            test_set = mvtecad.MVTecAD(self.args, train=False, meta=False)

            train_loader = DataLoader(train_set,
                                      worker_init_fn = worker_init_fn_seed_ref,
                                      batch_sampler = BalancedBatchSampler_ref(self.args, train_set),
                                      **kwargs)
            test_loader = DataLoader(test_set,
                                     batch_size=1,
                                     shuffle=False,
                                     worker_init_fn= worker_init_fn_seed_ref,
                                     **kwargs)

            support_loader, query_loader = [], []
            if self.args.meta == True and ref == False:
                cluster_index, aux_set = self.get_cluster(train_set, test_set)
                if self.args.auxiliary == True:
                    train_loader = aux_set

                support_loader = []
                query_loader = []

                m = 0
                for i in range(len(cluster_index)):
                    if len(cluster_index[i]) <= 2:
                        continue
                    else:
                        m=m+1

                self.args.cluster_num=m

                self.args.episode_num = (2 * math.factorial(m) // (
                            math.factorial(2) * math.factorial(m - 2)))

                for j in range(self.args.episode_num):
                    torch.cuda.empty_cache()
                    item1 = mvtecad.MVTecAD(self.args, train=True, meta=True, sp=True, k=j, cluster_index = cluster_index)
                    item2 = mvtecad.MVTecAD(self.args, train=True, meta=True, sp=False, k=j, cluster_index = cluster_index)

                    if self.args.aug_task == True:
                        n_sample, a_sample, n_sample2, a_sample2 = self.generate_task_aug(item1, item2)
                        item1 = n_sample + a_sample
                        item2 = n_sample2 + a_sample2

                    img, lb, img_s, s_item = [], [], [], []
                    for sample in item1:
                        image = sample["image"]
                        img_scale = sample["image_scale"]
                        label = sample["label"]
                        img.append(image)
                        img_s.append(img_scale)
                        lb.append(label)

                    s_item.append(img)
                    s_item.append(img_s)
                    s_item.append(lb)

                    img, lb, img_q, q_item = [], [], [], []
                    for sample in item2:
                        image = sample["image"]
                        img_scale = sample["image_scale"]
                        label = sample["label"]
                        img.append(image)
                        img_q.append(img_scale)
                        lb.append(label)

                    q_item.append(img)
                    q_item.append(img_q)
                    q_item.append(lb)

                    support_loader.append(s_item)
                    query_loader.append(q_item)

            return train_loader, test_loader, support_loader, query_loader
        else:
            raise NotImplementedError

    def get_cluster(self, train_set, test_set):
        normal_feature = []
        outlier_feature = []
        aux_set = []

        if self.args.auxiliary == False:
            for sample in train_set:
                lb = sample["label"]
                if lb == 1:
                    outlier_feature.append(sample["image"])
                elif lb == 0:
                    normal_feature.append(sample["image"])
                else:
                    continue
        else:
            img, all_lb, img_s, train_aux_set = [], [], [], []
            for sample in train_set:
                lb = sample["label"]
                image = sample["image"]
                img_scale = sample["image_scale"]
                img.append(image)
                img_s.append(img_scale)
                all_lb.append(lb)
                if lb == 1:
                    outlier_feature.append(sample["image"])
                elif lb == 0:
                    normal_feature.append(sample["image"])
                else:
                    continue
            train_aux_set.append(img)
            train_aux_set.append(img_s)
            train_aux_set.append(all_lb)

            img, all_lb, img_s, test_aux_set = [], [], [], []
            for sample in test_set:
                lb = sample["label"]
                image = sample["image"]
                img_scale = sample["image_scale"]
                img.append(image)
                img_s.append(img_scale)
                all_lb.append(lb)
            test_aux_set.append(img)
            test_aux_set.append(img_s)
            test_aux_set.append(all_lb)

            aux_set.append(train_aux_set)
            aux_set.append(test_aux_set)

        if self.args.extract == False:
            normal_array = np.array(torch.stack(normal_feature))
        else:
            normal_array = np.array(normal_feature)
        normal_array = normal_array.reshape(normal_array.shape[0], -1)

        kmeans = KMeans(n_clusters=self.args.cluster_num, random_state=self.args.ramdn_seed).fit(normal_array)
        normal_clusters = kmeans.labels_

        cluster_index = list()
        for i in range(self.args.cluster_num):
            index_list = np.argwhere(normal_clusters == i).flatten()
            cluster_index.append(index_list)

        return cluster_index, aux_set

    def generate_task_aug(self, item1, item2):
        n_sample, a_sample, n_img, n_img_scale = [], [], [], []
        n_sample2, a_sample2, n_img2, n_img_scale2 = [], [], [], []
        root = os.path.join(self.args.dataset_root, self.args.feat_classname)

        if self.args.model_name == "DRA":
            lb = 2
            img_dir = ""
        else:
            lb = 2
            img_dir = "dev/"

        for sample in item1:
            if sample["label"] == 0:
                n_sample.append(sample)
                n_img.append(sample["image"])
                n_img_scale.append(sample["image_scale"])
            else:
                a_sample.append(sample)

        for sample in item2:
            if sample["label"] == 0:
                n_sample2.append(sample)
                n_img2.append(sample["image"])
                n_img_scale2.append(sample["image_scale"])
            else:
                a_sample2.append(sample)

        rnd = random.randint(1, self.args.aug_type_num)

        # seen anomaly samples and DREAM augmentation
        if rnd == 1:
            for i in n_sample:
                cm = random.randint(0, 1)
                if cm == 0:
                    fn = i["fn"]
                    i["image"] = np.load(os.path.join(root, img_dir + "aug_dream/" + fn + ".npy"))
                    i["image_scale"] = np.load(os.path.join(root, img_dir + "aug_dream_scale/" + fn + ".npy"))
                    i["label"] = lb
            for j in n_sample2:
                cm = random.randint(0, 1)
                if cm == 0:
                    fn = j["fn"]
                    j["image"] = np.load(os.path.join(root, img_dir + "aug_dream/" + fn + ".npy"))
                    j["image_scale"] = np.load(os.path.join(root, img_dir + "aug_dream_scale/" + fn + ".npy"))
                    j["label"] = lb

        # seen anomaly samples and CutPaste augmentation
        elif rnd == 2:
            for i in n_sample:
                cm = random.randint(0, 1)
                if cm == 0:
                    fn = i["fn"]
                    i["image"] = np.load(os.path.join(root, img_dir + "aug_paste/" + fn + ".npy"))
                    i["image_scale"] = np.load(os.path.join(root, img_dir + "aug_paste_scale/" + fn + ".npy"))
                    i["label"] = lb

            for j in n_sample2:
                cm = random.randint(0, 1)
                if cm == 0:
                    fn = j["fn"]
                    j["image"] = np.load(os.path.join(root, img_dir + "aug_paste/" + fn + ".npy"))
                    j["image_scale"] = np.load(os.path.join(root, img_dir + "aug_paste_scale/" + fn + ".npy"))
                    j["label"] = lb

        # seen anomaly samples and CutMix augmentation
        elif rnd == 3:
            for i in n_sample:
                cm = random.randint(0, 1)
                if cm == 0:
                    fn = i["fn"]
                    i["image"] = np.load(os.path.join(root, img_dir + "aug_mix/" + fn + ".npy"))
                    i["image_scale"] = np.load(os.path.join(root, img_dir + "aug_mix_scale/" + fn + ".npy"))
                    i["label"] = lb

            for j in n_sample2:
                cm = random.randint(0, 1)
                if cm == 0:
                    fn = j["fn"]
                    j["image"] = np.load(os.path.join(root, img_dir + "aug_mix/" + fn + ".npy"))
                    j["image_scale"] = np.load(os.path.join(root, img_dir + "aug_mix_scale/" + fn + ".npy"))
                    j["label"] = lb

        return n_sample, a_sample, n_sample2, a_sample2