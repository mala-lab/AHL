import argparse
import copy
import math
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import copy
import torch.nn.functional as F
from einops import rearrange
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloaders.dataloader import initDataloader
from dataloaders.utlis import worker_init_fn_seed, BalancedBatchSampler
from datasets.base_dataset import Task_Dataset
from modeling.DRA_AHL import DRA
from modeling.Plain_AHL import Plain_Net
from modeling.aux_net import AUX_Model
from modeling.layers import build_criterion

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.autograd.set_detect_anomaly(True)

class Trainer(object):
    def __init__(self, args):
        self.args = args
        kwargs = {'num_workers': self.args.workers}
        builder = initDataloader(self.args)
        self.train_loader, self.test_loader, self.support_loader, self.query_loader = builder.build(ref = False,**kwargs)
        if self.args.model_name == "DevNet":
            self.model = Plain_Net(self.args)
        elif self.args.model_name == "DRA":
            if self.args.total_heads == 4:
                temp_args = copy.deepcopy(self.args)
                temp_args.batch_size = self.args.nRef
                temp_args.nAnomaly = 0
                temp_builder = initDataloader(temp_args)
                self.ref_loader, _, _, _= temp_builder.build(ref=True, **kwargs)
                self.ref = iter(self.ref_loader)

            self.model = DRA(self.args, backbone=self.args.backbone)
        else:
            print("model_name error!")

        self.max_auroc = 0
        self.max_pr = 0

        if self.args.pretrain_dir != None:
            self.aux_model.load_state_dict(torch.load(self.args.pretrain_dir))
            print('Load pretrain weight from: ' + self.args.pretrain_dir)

        self.criterion = build_criterion(self.args.criterion)
        self.mse_loss = torch.nn.MSELoss()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.002, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

        if self.args.auxiliary == True:
            self.aux_model = AUX_Model()
            self.aux_optimizer = torch.optim.Adam(self.aux_model.parameters(), lr=0.002, weight_decay=0.001)

            max_len_s = 0
            max_len_q = 0
            self.support_len = []
            self.query_len = []
            for episode in range(self.args.episode_num):
                l_s = len(self.support_loader[episode][2])
                self.support_len.append(l_s)
                if l_s > max_len_s:
                    max_len_s = l_s

                l_q = len(self.query_loader[episode][2])
                self.query_len.append(l_q)
                if l_q > max_len_q:
                    max_len_q = l_q

            self.aux_support_feature = torch.zeros(self.args.episode_num, max_len_s, self.args.sequence_len, self.args.episode_num)
            self.aux_query_feature = torch.zeros(self.args.episode_num, max_len_q, self.args.sequence_len, self.args.episode_num)

            self.aux_support_current = torch.zeros(self.args.episode_num, max_len_s, self.args.sequence_len, self.args.episode_num)
            self.aux_query_current = torch.zeros(self.args.episode_num, max_len_q, self.args.sequence_len, self.args.episode_num)

            self.support_real_score = torch.zeros(self.args.episode_num, max_len_s, self.args.episode_num)
            self.query_real_score = torch.zeros(self.args.episode_num, max_len_q, self.args.episode_num)

            self.support_score = torch.zeros(self.args.episode_num, max_len_s, self.args.episode_num)
            self.query_score = torch.zeros(self.args.episode_num, max_len_q, self.args.episode_num)

            self.pred_score_support = torch.zeros(self.args.episode_num, max_len_s, self.args.episode_num)
            self.pred_score_query = torch.zeros(self.args.episode_num, max_len_q, self.args.episode_num)

    def generate_target(self, target, eval=False):
        targets = list()
        if eval:
            targets.append(target == 0)
            targets.append(target)
            targets.append(target)
            targets.append(target)
            return targets
        else:
            temp_t = target != 0
            targets.append(target == 0)
            targets.append(temp_t[target != 2])
            targets.append(temp_t[target != 1])
            targets.append(target != 0)
        return targets

    def save_weights(self, filename):
        torch.save(self.model.state_dict(), os.path.join(self.args.experiment_dir, filename))

    def normalization(self, data):
        return data

    def get_reward(self, pred_score, ground_score, k=None):
        ground_score = np.array(ground_score)

        normal_idx = np.argwhere(ground_score == 0).flatten()
        outlier_idx = np.argwhere(ground_score == 1).flatten()
        aug_idx = np.argwhere(ground_score == 2).flatten()

        normal_label = torch.zeros(len(normal_idx))
        outlier_label = torch.ones(len(outlier_idx))
        pesudo_label = torch.ones(len(aug_idx))

        pred_normal = np.array([])
        pred_outlier = np.array([])
        pred_pesudo = np.array([])
        for i in normal_idx:
            pred_normal = np.append(pred_normal, pred_score[i][k].detach().numpy())
        for i in outlier_idx:
            pred_outlier = np.append(pred_outlier, pred_score[i][k].detach().numpy())
        for i in aug_idx:
            pred_pesudo = np.append(pred_pesudo, pred_score[i][k].detach().numpy())

        pred_normal = torch.Tensor(pred_normal)
        pred_outlier = torch.Tensor(pred_outlier)
        pred_pesudo = torch.Tensor(pred_pesudo)

        re_normal = self.mse_loss(pred_normal, normal_label)
        if len(pred_pesudo) > 0:
            re_pseudo = self.mse_loss(pred_pesudo, pesudo_label)
        else:
            re_pseudo = 0
        re_abnormal = self.mse_loss(pred_outlier, outlier_label)

        num_n = len(pred_normal)
        num_a = len(pred_outlier)
        num_p = len(pred_pesudo)
        return re_normal, re_pseudo, re_abnormal, num_n, num_a, num_p

    def train_unit(self, image, image_scale, image_targets, var = None, st = False):
        if self.args.model_name == "DRA":
            if self.args.total_heads == 4:
                try:
                    ref_image = next(self.ref)['image']
                    ref_image_scale = next(self.ref)['image_scale']
                except StopIteration:
                    self.ref = iter(self.ref_loader)
                    ref_image = next(self.ref)['image']
                    ref_image_scale = next(self.ref)['image_scale']
                ref_image = ref_image.cuda()
                ref_image_scale = ref_image_scale.cuda()
                image = torch.cat([ref_image, image], dim=0)
                image_scale = torch.cat([ref_image_scale, image_scale], dim=0)
            if st is True:
                targets = self.generate_target(image_targets, eval=True)
            else:
                targets = self.generate_target(image_targets)
            outputs = self.model.forward(image=image, image_scale=image_scale, label=image_targets,
                                           var=var, st=st)

            losses = list()
            for i in range(self.args.total_heads):
                if self.args.criterion == 'CE':
                    prob = F.softmax(targets[i], dim=1)
                    losses.append(self.criterion(prob, targets[i].long()).view(-1, 1))
                else:
                    losses.append(self.criterion(outputs[i], targets[i].float()).view(-1, 1))
            loss = torch.cat(losses)
            loss = torch.sum(loss)
        else:
            image_targets = image_targets.cpu()
            aug_index = np.argwhere(image_targets == 2).flatten()
            seen_index = np.argwhere(image_targets == 1).flatten()
            for i in aug_index:
                image_targets[i] = 1

            targets = image_targets.clone()
            outputs = self.model.forward(image=image, image_scale=image_scale, var=var)
            outputs_aug = outputs.clone()
            outputs_seen = outputs.clone()
            for j in range(len(image_targets)):
                if j in aug_index:
                    outputs_seen[j] = targets[j]
                elif j in seen_index:
                    outputs_aug[j] = targets[j]
            outputs_aug = outputs_aug.cuda()
            outputs_seen = outputs_seen.cuda()
            targets = targets.cuda()
            losses_aug = self.criterion(outputs_aug, targets.unsqueeze(1).float())
            losses_seen = self.criterion(outputs_seen, targets.unsqueeze(1).float())
            loss = losses_aug + losses_seen
        return outputs, loss

    def training(self, epoch):
        print("AHL training...")
        print("Epoch: ", epoch)
        self.model.train()
        self.optimizer.step()
        self.scheduler.step()
        torch.cuda.empty_cache()

        if self.args.auxiliary == True:
            real_score_q = []
            self.aux_query_feature = self.aux_query_feature.cuda()
            if epoch > self.args.sequence_len + 1:
                for episode in range(self.args.episode_num):
                    item_q = self.aux_model.forward(self.aux_query_feature[episode])
                    self.pred_score_query[episode] = item_q.clone()

                reward = []
                for e in range(self.args.episode_num):
                    r_n = 0.0
                    r_a = 0.0
                    r_p = 0.0
                    n_n = 0
                    n_a = 0
                    n_p = 0
                    for episode in range(self.args.episode_num):
                        re_normal, re_pseudo, re_abnormal, num_n, num_a, num_p = self.get_reward(self.pred_score_query[episode], self.query_loader[episode][2], k=e)  #6*N*1
                        r_n = r_n + re_normal
                        r_a = r_a + re_abnormal
                        r_p = r_p + re_pseudo
                        n_n = n_n + num_n
                        n_a = n_a + num_a
                        n_p = n_p + num_p
                    r = -(0.5*(r_n/n_n) + 0.5*(r_p/n_p) + 1*(r_a/n_a))
                    reward.append(r)
                s = 0
                for i in reward:
                    s = s + math.exp(i)
                reward = [math.exp(i)/s for i in reward]

        loss_q = [0 for _ in range(self.args.update_step + 1)]
        for episode in range(0, len(self.support_loader)):
            print("task: ", episode)
            torch.cuda.empty_cache()

            image_s, image_scale_s, targets_s = self.support_loader[episode][0], self.support_loader[episode][1], \
                                                self.support_loader[episode][2]
            image_s, image_scale_s, targets_s = torch.Tensor(image_s), torch.Tensor(image_scale_s), torch.Tensor(targets_s)
            train_set_s = Task_Dataset(image_s, image_scale_s, targets_s)
            train_loader_s = DataLoader(train_set_s,
                                      num_workers=self.args.workers,
                                      worker_init_fn=worker_init_fn_seed,
                                      batch_sampler=BalancedBatchSampler(self.args, train_set_s))

            tbar = tqdm(train_loader_s)
            for i, sample in enumerate(tbar):
                image, image_scale, targets = sample['image'], sample['image_scale'], sample['label']
                if self.args.cuda:
                    image, image_scale, targets = image.cuda(), image_scale.cuda(), targets.cuda()

                _, loss = self.train_unit(image, image_scale, targets, var=self.model.parameters())
                grad = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
                adapt_weights = list(map(lambda p: p[1] - 0.0002 * p[0], zip(grad, self.model.parameters())))

            image_q, image_scale_q, targets_q = self.query_loader[episode][0], self.query_loader[episode][1], \
                                                self.query_loader[episode][2]
            image_q, image_scale_q, targets_q = torch.Tensor(image_q), torch.Tensor(image_scale_q), torch.Tensor(targets_q)
            train_set_q = Task_Dataset(image_q, image_scale_q, targets_q)
            train_loader_q = DataLoader(train_set_q,
                                        num_workers=self.args.workers,
                                        worker_init_fn=worker_init_fn_seed,
                                        batch_sampler=BalancedBatchSampler(args, train_set_q)
                                        )
            tbar = tqdm(train_loader_q)
            for i, sample in enumerate(tbar):
                image2, image_scale2, targets2 = sample['image'], sample['image_scale'], sample['label']
                if self.args.cuda:
                    image2, image_scale2, targets2 = image2.cuda(), image_scale2.cuda(), targets2.cuda()

                with torch.no_grad():
                    _, loss = self.train_unit(image2, image_scale2, targets2, var = self.model.parameters())
                    loss_q[0] = loss_q[0] + loss

                with torch.no_grad():
                    _, loss = self.train_unit(image2, image_scale2, targets2, var = adapt_weights)
                    loss_q[1] = loss_q[1] + loss

            for k in range(1, self.args.update_step):
                torch.cuda.empty_cache()
                tbar = tqdm(train_loader_s)
                for i, sample in enumerate(tbar):
                    image, image_scale, targets = sample['image'], sample['image_scale'], sample['label']
                    if self.args.cuda:
                        image, image_scale, targets = image.cuda(), image_scale.cuda(), targets.cuda()
                    _, loss = self.train_unit(image, image_scale, targets, var=adapt_weights)
                    grad = torch.autograd.grad(loss, adapt_weights)
                    adapt_weights = list(map(lambda p: p[1] - 0.0002 * p[0], zip(grad, adapt_weights)))

                tbar = tqdm(train_loader_q)
                for i, sample in enumerate(tbar):
                    image2, image_scale2, targets2 = sample['image'], sample['image_scale'], sample['label']
                    if self.args.cuda:
                        image2, image_scale2, targets2 = image2.cuda(), image_scale2.cuda(), targets2.cuda()
                    _, loss = self.train_unit(image2, image_scale2, targets2, var = adapt_weights)
                    if self.args.auxiliary == True:
                        if epoch <= self.args.sequence_len + 1:
                            loss_q[k + 1] = loss_q[k + 1] + loss
                        else:
                            loss_q[k + 1] = loss_q[k + 1] + (0.5+0.5*reward[episode]) * loss
                    else:
                        loss_q[k + 1] = loss_q[k + 1] + loss

            if self.args.auxiliary == True:
                score_q = []
                for i in range(self.args.episode_num):
                    if self.args.model_name == "DRA":
                        class_pred = list()
                        for k in range(self.args.total_heads):
                            class_pred.append(np.array([]))
                    else:
                        total_pred = np.array([])
                    image_q, image_scale_q, targets_q = self.query_loader[i][0], self.query_loader[i][1], \
                                                        self.query_loader[i][2]
                    image_q, image_scale_q, targets_q = torch.Tensor(image_q), torch.Tensor(image_scale_q), torch.Tensor(
                        targets_q)

                    image_q, image_scale_q, targets_q = image_q.cuda(), image_scale_q.cuda(), targets_q.cuda()
                    tmp_out, _ = self.train_unit(image_q, image_scale_q, targets_q, var=adapt_weights, st=True)

                    for k in range(self.args.total_heads):
                        if k == 0:
                            data = -1 * tmp_out[k].data.cpu().numpy()
                        else:
                            data = tmp_out[k].data.cpu().numpy()
                        class_pred[k] = np.append(class_pred[k], data)

                    total_pred = self.normalization(class_pred[0])
                    for k in range(1, self.args.total_heads):
                        total_pred = total_pred + self.normalization(class_pred[k])

                    score_q.append(total_pred)

                arry_q = np.zeros([len(score_q), len(max(score_q, key = lambda x:len(x)))])

                for i, j in enumerate(score_q):
                    arry_q[i][0:len(j)] = j

                real_score_q.append(arry_q)

        if self.args.auxiliary == True:
            real_score_q = torch.Tensor(real_score_q)
            real_score_q = rearrange(real_score_q, 'e k n-> k n e')
            self.query_real_score = real_score_q.clone()

            if epoch <= self.args.sequence_len - 1:
                for episode in range(len(self.aux_support_feature)):
                    for i in range(len(self.aux_query_feature[1])):
                        self.aux_query_feature[episode][i][epoch] = real_score_q[episode][i].clone()
            else:
                self.aux_query_current = self.aux_query_feature.clone()
                for episode in range(len(self.aux_support_feature)):
                    for i in range(len(self.aux_query_feature[1])):
                        self.aux_query_feature[episode][i][0] = self.aux_query_feature[episode][i][1].clone()   # 6*n*3*6
                        self.aux_query_feature[episode][i][1] = self.aux_query_feature[episode][i][2].clone()
                        self.aux_query_feature[episode][i][2] = self.aux_query_feature[episode][i][3].clone()
                        self.aux_query_feature[episode][i][3] = self.aux_query_feature[episode][i][4].clone()
                        self.aux_query_feature[episode][i][4] = real_score_q[episode][i].clone()

            if epoch >= self.args.sequence_len:
                total_loss = 0.0
                for episode in range(self.args.episode_num):
                    item_q = self.aux_model.forward(self.aux_query_current[episode][:self.query_len[episode]])
                    self.query_score = item_q.clone().cpu()
                    loss2 = self.mse_loss(
                        self.query_score,
                        self.query_real_score[episode][:self.query_len[episode]])
                    self.aux_optimizer.zero_grad()
                    loss2.backward()
                    self.aux_optimizer.step()
                    total_loss = total_loss + loss2.item()
                print("aux_loss:", total_loss)

            if epoch <= self.args.sequence_len + 1:
                loss_f = loss_q[-1] / self.args.episode_num
            else:
                loss_f = loss_q[-1]
            print("epoch_loss:", loss_f)

        else:
            loss_f = loss_q[-1] / self.args.episode_num
            print("epoch_loss:", loss_f)

        self.optimizer.zero_grad()
        loss_f.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        print("AHL testing finished")

    def eval_DRA(self):
        self.model.eval()
        tbar = tqdm(self.test_loader, desc='\r')
        test_loss = 0.0
        class_pred = list()
        for i in range(self.args.total_heads):
            class_pred.append(np.array([]))
        total_target = np.array([])
        for i, sample in enumerate(tbar):
            image, image_scale, target = sample['image'], sample["image_scale"], sample['label']
            if self.args.cuda:
                image, image_scale, target = image.cuda(), image_scale.cuda(), target.cuda()

            if self.args.total_heads == 4:
                try:
                    ref_image = next(self.ref)['image']
                    ref_image_scale = next(self.ref)['image_scale']
                except StopIteration:
                    self.ref = iter(self.ref_loader)
                    ref_image = next(self.ref)['image']
                    ref_image_scale = next(self.ref)['image_scale']
                ref_image = ref_image.cuda()
                ref_image_scale = ref_image_scale.cuda()
                image = torch.cat([ref_image, image], dim=0)
                image_scale = torch.cat([ref_image_scale, image_scale], dim=0)

            with torch.no_grad():
                outputs = self.model.forward(image=image, image_scale=image_scale, label=target, var=self.model.parameters())
                targets = self.generate_target(target, eval=True)

                losses = list()
                for i in range(self.args.total_heads):
                    if self.args.criterion == 'CE':
                        prob = F.softmax(outputs[i], dim=1)
                        losses.append(self.criterion(prob, targets[i].long()))
                    else:
                        losses.append(self.criterion(outputs[i], targets[i].float()))

                loss = torch.stack(losses)
                loss = torch.sum(loss)

            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            total_target = np.append(total_target, target.cpu().numpy())

            for i in range(self.args.total_heads):
                if i == 0:
                    data = -1 * outputs[i].data.cpu().numpy()
                else:
                    data = outputs[i].data.cpu().numpy()
                class_pred[i] = np.append(class_pred[i], data)

        total_pred = self.normalization(class_pred[0])
        for i in range(1, self.args.total_heads):
            total_pred = total_pred + self.normalization(class_pred[i])

        with open(self.args.experiment_dir + '/result.txt', mode='a+', encoding="utf-8") as w:
            for label, score in zip(total_target, total_pred):
                w.write(str(label) + '   ' + str(score) + "\n")

        total_roc, total_pr = aucPerformance(total_pred, total_target)
        if self.max_auroc < total_roc:
            self.max_auroc = total_roc
            self.max_pr = total_pr

        return total_roc, total_pr

    def eval_devnet(self):
        self.model.eval()
        tbar = tqdm(self.test_loader, desc='\r')
        test_loss = 0.0
        total_pred = np.array([])
        total_target = np.array([])
        for i, sample in enumerate(tbar):
            image, image_scale, target = sample['image'], sample["image_scale"], sample['label']
            if self.args.cuda:
                image, image_scale, target = image.cuda(), image_scale.cuda(), target.cuda()

            with torch.no_grad():
                outputs = self.model.forward(image=image, image_scale=image_scale, var=self.model.parameters())

                if self.args.criterion == 'CE':
                    prob = F.softmax(outputs, dim=1)
                    losses = self.criterion(prob, target.long())
                else:
                    losses = self.criterion(outputs, target.float())

            test_loss = test_loss + losses
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))

            data = outputs.data.cpu().numpy()
            total_pred = np.append(total_pred, data)
            total_target = np.append(total_target, target.cpu().numpy())

        with open(self.args.experiment_dir + '/result.txt', mode='a+', encoding="utf-8") as w:
            for label, score in zip(total_target, total_pred):
                w.write(str(label) + '   ' + str(score) + "\n")

        total_roc, total_pr = aucPerformance(total_pred, total_target)
        if self.max_auroc < total_roc:
            self.max_auroc = total_roc
            self.max_pr = total_pr

        return total_roc, total_pr


def aucPerformance(mse, labels, prt=True):
    roc_auc = roc_auc_score(labels, mse)
    ap = average_precision_score(labels, mse)
    if prt:
        print("AUC-ROC: %.4f, AUC-PR: %.4f" % (roc_auc, ap))
    return roc_auc, ap


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=48, help="batch size used in SGD")
    parser.add_argument("--epochs", type=int, default=30, help="the number of epochs")
    parser.add_argument("--steps_per_epoch", type=int, default=20, help="the number of batches per epoch")
    parser.add_argument("--cont_rate", type=float, default=0.0, help="the outlier contamination rate in the training data")
    parser.add_argument("--test_threshold", type=int, default=0,
                        help="the outlier contamination rate in the training data")
    parser.add_argument("--test_rate", type=float, default=0.0,
                        help="the outlier contamination rate in the training data")
    parser.add_argument("--dataset", type=str, default='mvtecad', help="a list of data set names")
    parser.add_argument("--ramdn_seed", type=int, default=42, help="the random seed number")
    parser.add_argument('--workers', type=int, default=4, metavar='N', help='dataloader threads')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--savename', type=str, default='model.pkl', help="save modeling")
    parser.add_argument('--dataset_root', type=str, default='../PDA/data/SDD_anomaly_detection/', help="dataset root")
    parser.add_argument('--experiment_dir', type=str, default='./experiment', help="dataset root")
    parser.add_argument('--classname', type=str, default='SDD', help="dataset class")
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
    parser.add_argument('--feat_classname', type=str, default='SDD', help="dataset class")
    parser.add_argument('--cluster_num', type=int, default=3, help="number of normal clusters")
    parser.add_argument('--AHL', type=bool, default=True, help="")
    parser.add_argument('--auxiliary', type=bool, default=True, help="whether use auxiliary model or not")
    parser.add_argument('--aug_task', type=bool, default=True, help="whether use different augmentation techniques in different tasks or not")
    parser.add_argument('--aug_type_num', type=int, default=3, help="number of augmentation techniques")
    parser.add_argument('--episode_num', type=int, default=6, help="number of episodes")
    parser.add_argument('--sequence_len', type=int, default=5, help="size of sequence")
    parser.add_argument('--update_step', type=int, default=3, help="number of inner loop")
    parser.add_argument('--extract', type=bool, default=True, help="whether use extracted feature or not")
    parser.add_argument('--save_feature', type=bool, default=False, help="whether save extracted feature or not")
    parser.add_argument('--model_name', type=str, default="DRA", help="use which model to test AHL-learning")
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    setup_seed(args.ramdn_seed)
    trainer = Trainer(args)

    if args.cuda:
        trainer.model = trainer.model.cuda()
        if args.auxiliary == True:
            trainer.aux_model = trainer.aux_model.cuda()

    argsDict = args.__dict__
    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)
    with open(args.experiment_dir + '/setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')

    print('Total Epoches:', trainer.args.epochs)
    print("Model Name:", trainer.args.model_name)
    print("Use Extracted Feature:", trainer.args.extract)
    if args.cuda:
        trainer.criterion = trainer.criterion.cuda()
        trainer.mse_loss = trainer.mse_loss.cuda()

    for epoch in range(0, trainer.args.epochs):
        trainer.training(epoch)
        if trainer.args.model_name == "DRA":
            trainer.eval_DRA()
        else:
            trainer.eval_devnet()

    args.savename = args.classname + "_ctest.pkl"
    trainer.save_weights(args.savename)

    if args.save_feature == True:
        save_feature(args, trainer.model)
        args.savename = args.classname + ".pkl"
        trainer.save_weights(args.savename)






