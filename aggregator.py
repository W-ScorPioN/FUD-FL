#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9

"""
使用前提：
    1. 保存本地模型；
    2. 保存全局模型；
功能：
    1. 基于余弦相似度，计算每个客户端对全局模型的贡献度；
    2. 根据相似性结果，用K-Means算法聚类得分相近的本地模型
"""
import gc
import logging
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from utils.options import args_parser
from utils.CKA import compute_linear_CKA
from utils.info import load_models
from utils.dataset import load_dataset2
from models.Update import DatasetSplit
from torch.utils.data import DataLoader

import time


class ModelAggregator:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters

    def compute_similarity(self, arg, similarity, local_models, global_model):

        _similarities = None
        # 计算本地模型与全局模型之间的余弦相似度
        if similarity == 'cosine':
            # 加载并展平模型参数
            global_params = self._flatten_model_params(global_model)
            local_params = [self._flatten_model_params(model) for model in local_models]

            # 确保参数是 NumPy 数组并且是二维的
            global_params = np.array(global_params).reshape(1, -1)  # 确保 global_params 是二维的
            local_params = np.array(local_params)  # 将 local_params 转换为 NumPy 数组
            print(f"Global params shape: {global_params.shape}")
            print(f"Local params shapes: {local_params.shape}")

            _similarities = cosine_similarity(local_params, global_params)

        elif similarity == 'CKA':
            dataset_train, _, dict_users = load_dataset2(arg)
            ldr_train_new = DataLoader(DatasetSplit(dataset_train, dict_users[0]), batch_size=arg.local_bs,
                                       shuffle=True)

            sample_train_data_local = None
            for images, _ in ldr_train_new:
                sample_train_data_local = images.to(arg.device)
                break

            _similarities = compute_linear_CKA(args, local_models, global_model, sample_train_data_local)
            print(f'_similarities: {_similarities}')
        return _similarities.flatten()



    @staticmethod
    def _flatten_model_params(model):
        try:
            # 展平模型参数
            state_dict = model.state_dict() if isinstance(model, torch.nn.Module) else model
            # params = np.concatenate([param.flatten().numpy() for param in state_dict.values()])
            params = np.concatenate([param.flatten().cpu().numpy() for param in state_dict.values()])
            return params
        except Exception as e:
            print(f"Error in flattening model params: {e}")
            return None

    def get_flattened_model_params(self, models):
        return [self._flatten_model_params(model) for model in models]


def plot_clusters(args, data, labels, title='Cluster Plot'):
    """
    使用t-SNE对数据降维并绘制聚类结果。

    :param data: 原始高维数据。
    :param labels: 聚类标签。
    :param title: 图的标题。
    """
    # 确定 perplexity 的值
    num_samples = len(data)
    perplexity_value = min(30, num_samples - 1)  # perplexity 必须小于样本数量

    # 使用 t-SNE 进行降维
    tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42, early_exaggeration=24.0, n_iter=1000, init='pca')
    reduced_data = tsne.fit_transform(data)

    # 绘制集群
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis')
    plt.title(title)
    plt.colorbar(scatter)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.savefig(f'./save/kmeans_clustering/png/{args.dataset}_{args.model}_{str(args.epochs)}_{str(args.num_users)}_{args.similarity}_{str(args.iid)}_{str(args.frac)}_{str(args.bs)}_{args.task_id}.png')
    plt.show()

if __name__ == '__main__':
    gc.collect()
    torch.cuda.empty_cache()

    # parse args
    args = args_parser()
    logging.basicConfig(
        filename=f'./save/kmeans_clustering/{args.dataset}_{args.model}_{str(args.epochs)}_{str(args.num_users)}_{args.similarity}_{str(args.iid)}_{str(args.frac)}_{str(args.bs)}_{args.task_id}.log',
        level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # 初始化变量
    # global_model_path = "./save/model/xxx.pth"  # 全局模型路径，需要调整
    # local_models_path = ["./save/model/xxx1.pth", "./save/model/xxx2.pth"]  # 本地模型路径，需要调整
    #
    # # 加载模型
    # global_model = torch.load(global_model_path)
    # local_models = [torch.load(path) for path in local_models_path]

    # 第0次完整训练的最后一个epoch的全局模型
    global_model_train = load_models('./save/FL/global_models/', args, 0, 202542243910)[-1]
    # print(global_model_train)
    # 第0次完整训练的最后一个epoch的所有本地模型
    local_models_train = load_models('./save/FL/local_models/', args, 0, 202542243910)[-1]
    # print(local_models_train)

    aggregator = ModelAggregator(n_clusters=args.n_clusters)
    start = time.time()
    similarities = aggregator.compute_similarity(args, args.similarity, local_models_train, global_model_train)
    end = time.time()

    logging.info(f'The time cost of {args.similarity} similarity computation is: {(end - start):6.6f}')
    logging.info(f'{args.similarity} similarity: \n {similarities}')

    # 使用 K-means 聚类
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42)
    print(f'similarities: {similarities}, similarities.reshape(-1, 1): {similarities.reshape(-1, 1)}, similarities.shape: {similarities.shape}')
    labels = kmeans.fit_predict(similarities.reshape(-1, 1))

    # 将所有模型参数展平并组合成一个数据矩阵
    local_param = aggregator.get_flattened_model_params(local_models_train)

    # 绘制聚类结果
    data = np.array(local_param)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    plot_clusters(args, data, labels, title='Model Clustering Result')

    logging.info(f"Cluster labels: {labels}")