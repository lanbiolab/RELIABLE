import math
import sys
from typing import Iterable

import numpy as np
import torch

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import utils
from utils import adjust_learning_config, SmoothedValue, MetricLogger

def train_one_epoch(model: torch.nn.Module,
                    data_loader_train: Iterable,
                    data_loader_train_all: Iterable,
                    data_loader_test: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int,
                    state_logger=None,
                    args=None,
                    ):
    total_loss = 0
    # 热身环节不进行互补性计算
    if epoch >= args.start_rectify_epoch:
        # 添加伪样本模块
        commonZ_list = []
        data_loader = enumerate(data_loader_train_all)
        for data_iter_step, (ids, samples, mask, data_label) in data_loader:
            for i in range(args.n_views):
                samples[i] = samples[i].to(device, non_blocking=True)
            with torch.autocast('cuda', enabled=False):  # 自动分配到GPU上
                z, p = model.compute_feature(samples)
                commonz = model.fusion(z)
                commonZ_list.append(commonz)

        commonZ = torch.cat(commonZ_list, dim=0)
        psedo_labels = model.clustering(commonZ)
        model.psedo_labels = psedo_labels

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = args.print_freq
    if args.print_this_epoch:
        data_loader = enumerate(metric_logger.log_every(data_loader_train, print_freq, header))
    else:
        data_loader = enumerate(data_loader_train)

    model.train(True)
    optimizer.zero_grad()

    for data_iter_step, (ids, samples, mask, data_label) in data_loader:
        smooth_epoch = epoch + (data_iter_step + 1) / len(data_loader_train)
        mmt = args.momentum

        for i in range(args.n_views):
            samples[i] = samples[i].to(device, non_blocking=True)

        with torch.autocast('cuda', enabled=False):
            loss = model(samples, mmt, epoch < args.start_rectify_epoch)

        # 热身环节不进行互补性计算
        if epoch >= args.start_rectify_epoch:
            batch_psedo_label = model.psedo_labels[ids]
            zs, ps = model.compute_feature(samples)
            common_z = model.fusion(zs)  # Adaptive fusion
            q_centers = model.compute_centers(common_z, batch_psedo_label)
            loss_list = list()
            for i in range(args.n_views):
                k_centers = model.compute_centers(zs[i], batch_psedo_label)
                loss_list.append(model.compute_cluster_loss(q_centers, k_centers, batch_psedo_label))
            # 超参调整
            loss = args.alpha * loss + args.beta * sum(loss_list)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr = adjust_learning_config(optimizer, smooth_epoch, args)

        if args.print_this_epoch:
            metric_logger.update(lr=lr)
            metric_logger.update(loss=loss_value)

    # gather the stats from all processes
    if args.print_this_epoch:
        print("Averaged stats:", metric_logger)
        eval_result = evaluate(model, data_loader_test, device, epoch, args)
    else:
        eval_result = None
    return eval_result


def evaluate(model: torch.nn.Module, data_loader_test: Iterable,
             device: torch.device, epoch: int,
             args=None):
    model.eval()
    extracter = model.extract_feature
    with torch.no_grad():
        features_all = torch.zeros(args.n_views, args.n_samples, args.embed_dim).to(device)
        labels_all = torch.zeros(args.n_samples, dtype=torch.long).to(device)
        for indexs, samples, mask, labels in data_loader_test:
            for i in range(args.n_views):
                samples[i] = samples[i].to(device, dtype=torch.float32, non_blocking=True)

            labels = labels.to(device, non_blocking=True).to(indexs.dtype)
            features = extracter(samples, mask)

            for i in range(args.n_views):
                features_all[i][indexs] = features[i]

            labels_all[indexs] = labels

        features_cat = features_all.permute(1, 0, 2).reshape(args.n_samples, -1)
        features_cat = torch.nn.functional.normalize(features_cat, dim=-1).cpu().numpy()
        kmeans_label = KMeans(n_clusters=args.n_classes, random_state=0).fit_predict(features_cat)

    nmi, ari, f, acc = utils.evaluate(np.asarray(labels_all.cpu()), kmeans_label)
    result = {'nmi': nmi, 'ari': ari, 'f': f, 'acc': acc}
    return result