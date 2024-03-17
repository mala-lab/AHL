import numpy as np
from torch.utils.data import Sampler
from datasets.base_dataset import Task_Dataset
from datasets.base_dataset import BaseADDataset

def worker_init_fn_seed(worker_id):
    seed = 10
    seed = seed + worker_id
    np.random.seed(seed)

class BalancedBatchSampler(Sampler):
    def __init__(self,
                 cfg,
                 dataset: Task_Dataset):
        super(BalancedBatchSampler, self).__init__(dataset)
        self.cfg = cfg
        self.dataset = dataset

        label_list = list()
        for i in self.dataset:
            label_list.append(int(i["label"]))

        label_list = np.array(label_list)
        normal_idx = np.argwhere(label_list == 0).flatten()
        outlier_idx = np.argwhere(label_list == 1).flatten()
        aug_idx = np.argwhere(label_list == 2).flatten()

        if len(aug_idx) != 0:
            self.aug_generator = self.randomGenerator(aug_idx)

        self.normal_generator = self.randomGenerator(normal_idx)

        if len(outlier_idx) != 0:
            self.outlier_generator = self.randomGenerator(outlier_idx)

        if self.cfg.nAnomaly != 0:
            if len(aug_idx) != 0 and len(outlier_idx) != 0:
                self.n_normal = 1 * self.cfg.batch_size // 3
                self.n_aug = 1 * self.cfg.batch_size // 3
                self.n_outlier = self.cfg.batch_size - self.n_normal - self.n_aug
            elif len(aug_idx) == 0:
                self.n_normal = 1 * self.cfg.batch_size // 2
                self.n_aug = 0
                self.n_outlier = self.cfg.batch_size - self.n_normal - self.n_aug
            elif len(outlier_idx) == 0:
                self.n_normal = 1 * self.cfg.batch_size // 2
                self.n_outlier = 0
                self.n_aug = self.cfg.batch_size - self.n_normal - self.n_outlier

        else:
            self.n_normal = 1 * self.cfg.batch_size // 2
            self.n_outlier = 0
            self.n_aug = self.cfg.batch_size - self.n_normal - self.n_outlier

    def randomGenerator(self, list):
        while True:
            random_list = np.random.permutation(list)
            for i in random_list:
                yield i

    def __len__(self):
        return self.cfg.steps_per_epoch

    
    def __iter__(self):
        for _ in range(self.cfg.steps_per_epoch):
            batch = []

            for _ in range(self.n_normal):
                batch.append(next(self.normal_generator))

            if self.n_outlier != 0:
                for _ in range(self.n_outlier):
                    batch.append(next(self.outlier_generator))

            if self.n_aug != 0:
                for _ in range(self.n_aug):
                    batch.append(next(self.aug_generator))

            yield batch


def worker_init_fn_seed_ref(worker_id):
    seed = 10
    seed += worker_id
    np.random.seed(seed)


class BalancedBatchSampler_ref(Sampler):
    def __init__(self,
                 cfg,
                 dataset: BaseADDataset):
        super(BalancedBatchSampler_ref, self).__init__(dataset)
        self.cfg = cfg
        self.dataset = dataset

        self.normal_generator = self.random_generator(self.dataset.normal_idx)
        self.outlier_generator = self.random_generator(self.dataset.outlier_idx)
        if self.cfg.nAnomaly != 0:
            self.n_normal = self.cfg.batch_size // 2
            self.n_outlier = self.cfg.batch_size - self.n_normal
        else:
            self.n_normal = self.cfg.batch_size
            self.n_outlier = 0

    @staticmethod
    def random_generator(idx_list):
        while True:
            random_list = np.random.permutation(idx_list)
            for i in random_list:
                yield i

    def __len__(self):
        return self.cfg.steps_per_epoch

    def __iter__(self):
        for _ in range(self.cfg.steps_per_epoch):
            batch = []

            for _ in range(self.n_normal):
                batch.append(next(self.normal_generator))

            for _ in range(self.n_outlier):
                batch.append(next(self.outlier_generator))
            yield batch