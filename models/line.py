import math
import random

import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
import numpy as np
from utils import preprocess_nxgraph
from models.alias import create_alias_table, alias_sample

def line_loss(y_true, y_pred):
    # Make sure y_pred requires gradient and is involved in the computation graph
    return -torch.mean(torch.log(torch.sigmoid(y_true * y_pred)))

class LINEModel(nn.Module):
    def __init__(self, num_nodes, embedding_size, order='second'):
        super(LINEModel, self).__init__()
        self.embedding_size = embedding_size

        # Embeddings for first and second order
        self.first_emb = nn.Embedding(num_nodes, embedding_size)
        self.second_emb = nn.Embedding(num_nodes, embedding_size)
        self.context_emb = nn.Embedding(num_nodes, embedding_size)

        # Weight initialization
        nn.init.xavier_uniform_(self.first_emb.weight)
        nn.init.xavier_uniform_(self.second_emb.weight)
        nn.init.xavier_uniform_(self.context_emb.weight)

        self.order = order

    def forward(self, v_i, v_j):
        v_i_emb = self.first_emb(v_i)
        v_j_emb = self.first_emb(v_j)

        v_i_emb_second = self.second_emb(v_i)
        v_j_context_emb = self.context_emb(v_j)

        # First-order proximity
        first_order = torch.sum(v_i_emb * v_j_emb, dim=-1)

        # Second-order proximity
        second_order = torch.sum(v_i_emb_second * v_j_context_emb, dim=-1)

        if self.order == 'first':
            return first_order
        elif self.order == 'second':
            return second_order
        else:
            return first_order, second_order


class LINE:
    def __init__(self, graph, embedding_size=8, negative_ratio=5, order='second'):
        """
        :param graph: NetworkX graph
        :param embedding_size: Dimension of embeddings
        :param negative_ratio: Number of negative samples per positive sample
        :param order: 'first', 'second', or 'all'
        """
        if order not in ['first', 'second', 'all']:
            raise ValueError('mode must be first, second, or all')

        self.graph = graph
        self.idx2node, self.node2idx = preprocess_nxgraph(graph)

        self.rep_size = embedding_size
        self.order = order
        self.negative_ratio = negative_ratio
        self.node_size = graph.number_of_nodes()
        self.edge_size = graph.number_of_edges()
        self.samples_per_epoch = self.edge_size * (1 + negative_ratio)

        self._gen_sampling_table()
        self.reset_model()

    def reset_training_config(self, batch_size, times):
        self.batch_size = batch_size
        self.steps_per_epoch = (
            (self.samples_per_epoch - 1) // self.batch_size + 1) * times

    def reset_model(self):
        self.model = LINEModel(self.node_size, self.rep_size, self.order)
        self.optimizer = optim.Adam(self.model.parameters())

    def _gen_sampling_table(self):
        # Create sampling table for vertex
        power = 0.75
        num_nodes = self.node_size
        node_degree = np.zeros(num_nodes)  # Out degree
        node2idx = self.node2idx

        for edge in self.graph.edges():
            node_degree[node2idx[edge[0]]] += self.graph[edge[0]][edge[1]].get('weight', 1.0)

        total_sum = sum([math.pow(node_degree[i], power) for i in range(num_nodes)])
        norm_prob = [float(math.pow(node_degree[j], power)) / total_sum for j in range(num_nodes)]

        self.node_accept, self.node_alias = create_alias_table(norm_prob)

        # Create sampling table for edge
        num_edges = self.graph.number_of_edges()
        total_sum = sum([self.graph[edge[0]][edge[1]].get('weight', 1.0) for edge in self.graph.edges()])
        norm_prob = [self.graph[edge[0]][edge[1]].get('weight', 1.0) * num_edges / total_sum for edge in self.graph.edges()]

        self.edge_accept, self.edge_alias = create_alias_table(norm_prob)

    # Update batch_iter to disable gradient tracking for the `sign` tensor
    def batch_iter(self):
        edges = [(self.node2idx[x[0]], self.node2idx[x[1]]) for x in self.graph.edges()]

        data_size = self.graph.number_of_edges()
        shuffle_indices = np.random.permutation(np.arange(data_size))

        # Positive or negative mod
        mod = 0
        mod_size = 1 + self.negative_ratio
        h = []
        t = []
        sign = 0
        start_index = 0
        end_index = min(start_index + self.batch_size, data_size)
        while True:
            if mod == 0:
                h, t = [], []
                for i in range(start_index, end_index):
                    if random.random() >= self.edge_accept[shuffle_indices[i]]:
                        shuffle_indices[i] = self.edge_alias[shuffle_indices[i]]
                    cur_h = edges[shuffle_indices[i]][0]
                    cur_t = edges[shuffle_indices[i]][1]
                    h.append(cur_h)
                    t.append(cur_t)
                # Disable gradient tracking for sign tensor
                with torch.no_grad():
                    sign = torch.ones(len(h))
            else:
                with torch.no_grad():
                    sign = torch.ones(len(h)) * -1
                t = [alias_sample(self.node_accept, self.node_alias) for _ in range(len(h))]

            yield torch.tensor(h, dtype=torch.long), torch.tensor(t, dtype=torch.long), sign

            mod = (mod + 1) % mod_size
            if mod == 0:
                start_index = end_index
                end_index = min(start_index + self.batch_size, data_size)
                if start_index >= data_size:
                    shuffle_indices = np.random.permutation(np.arange(data_size))
                    start_index = 0
                    end_index = min(start_index + self.batch_size, data_size)

    def get_embeddings(self):
        embeddings = np.zeros((1024, 1024))
        if self.order == 'first':
            emb = self.model.first_emb.weight.data.cpu().numpy()
        elif self.order == 'second':
            emb = self.model.second_emb.weight.data.cpu().numpy()
        else:
            emb = np.hstack([self.model.first_emb.weight.data.cpu().numpy(),
                             self.model.second_emb.weight.data.cpu().numpy()])

        for i, embedding in enumerate(emb):
            embeddings[self.idx2node[i]] = embedding

        return embeddings

    def train(self, batch_size=1024, epochs=1, verbose=1):
        self.reset_training_config(batch_size, 1)

        data_iter = self.batch_iter()

        for epoch in range(epochs):
            total_loss = 0
            for step in range(self.steps_per_epoch):
                v_i, v_j, sign = next(data_iter)
                # print(v_i)
                # print(v_j)
                # print(sign)
                self.optimizer.zero_grad()
                preds = self.model(v_i, v_j)
                if isinstance(preds, tuple):  # When both first and second order are used
                    loss = sum(line_loss(sign, pred) for pred in preds)
                else:
                    loss = line_loss(sign, preds)

                # torch.set_grad_enabled(True)  # 启动梯度计算
                loss.requires_grad = True
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            # if verbose:
            #     print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / self.steps_per_epoch}')
