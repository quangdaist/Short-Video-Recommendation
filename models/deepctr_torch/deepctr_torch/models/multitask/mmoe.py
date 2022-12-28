# -*- coding:utf-8 -*-
"""
Author:
    zanshuxun, zanshuxun@aliyun.com

Reference:
    [1] Jiaqi Ma, Zhe Zhao, Xinyang Yi, et al. Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts[C] (https://dl.acm.org/doi/10.1145/3219819.3220007)
"""
import torch
import torch.nn as nn

from ..basemodel import BaseModel
from ...inputs import combined_dnn_input
from ...layers import DNN, PredictionLayer
import rtdl
import math
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional, Union
import numpy as np

import rtdl
import torch
import torch.nn as nn
import zero
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from torch import Tensor
from torch.nn import Parameter  # type: ignore[code]
from tqdm import trange


class NLinear(nn.Module):
    def __init__(self, n: int, d_in: int, d_out: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = Parameter(Tensor(n, d_in, d_out))
        self.bias = Parameter(Tensor(n, d_out)) if bias else None
        with torch.no_grad():
            for i in range(n):
                layer = nn.Linear(d_in, d_out)
                self.weight[i] = layer.weight.T
                if self.bias is not None:
                    self.bias[i] = layer.bias

    def forward(self, x):
        assert x.ndim == 3
        x = x[..., None] * self.weight[None]
        x = x.sum(-2)
        if self.bias is not None:
            x = x + self.bias[None]
        return x


@dataclass
class AutoDisOptions:
    n_meta_embeddings: int
    temperature: float


class AutoDis(nn.Module):
    """
    Paper (the version is important): https://arxiv.org/abs/2012.08986v2
    Code: https://github.com/mindspore-ai/models/tree/bdf2d8bcf11fe28e4ad3060cf2ddc818eacd8597/research/recommend/autodis

    The paper is significantly different from the code (it looks like the code
    implements the first version of the paper). We implement the second version
    here. Not all technical details are given for the second version, so what we do
    here can be different from what authors actually did.

    Anyway, AutoDis (v2) is essentially the following sequence of layers (applied from
    left to right): [Linear(no bias), LeakyReLU, Linear(no bias), Softmax, Linear]
    """

    def __init__(
            self, n_features: int, d_embedding: int, options: AutoDisOptions
    ) -> None:
        super().__init__()
        self.first_layer = rtdl.NumericalFeatureTokenizer(
            n_features,
            options.n_meta_embeddings,
            False,
            'uniform',
        )
        self.leaky_relu = nn.LeakyReLU()
        self.second_layer = NLinear(
            n_features, options.n_meta_embeddings, options.n_meta_embeddings, False
        )
        self.softmax = nn.Softmax(-1)
        self.temperature = options.temperature
        # "meta embeddings" from the paper are just a linear layer
        self.third_layer = NLinear(
            n_features, options.n_meta_embeddings, d_embedding, False
        )
        # 0.01 is taken from the source code
        nn.init.uniform_(self.third_layer.weight, 0.01)

    def forward(self, x: Tensor) -> Tensor:
        x = self.first_layer(x)
        x = self.leaky_relu(x)
        x = self.second_layer(x)
        x = self.softmax(x / self.temperature)
        x = self.third_layer(x)
        return x


class MMOE(BaseModel):
    """Instantiates the Multi-gate Mixture-of-Experts architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param num_experts: integer, number of experts.
    :param expert_dnn_hidden_units: list, list of positive integer or empty list, the layer number and units in each layer of expert DNN.
    :param gate_dnn_hidden_units: list, list of positive integer or empty list, the layer number and units in each layer of gate DNN.
    :param tower_dnn_hidden_units: list, list of positive integer or empty list, the layer number and units in each layer of task-specific DNN.
    :param l2_reg_linear: float, L2 regularizer strength applied to linear part.
    :param l2_reg_embedding: float, L2 regularizer strength applied to embedding vector.
    :param l2_reg_dnn: float, L2 regularizer strength applied to DNN.
    :param init_std: float, to use as the initialize std of embedding vector.
    :param seed: integer, to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN.
    :param dnn_use_bn: bool, Whether use BatchNormalization before activation or not in DNN.
    :param task_types: list of str, indicating the loss of each tasks, ``"binary"`` for  binary logloss, ``"regression"`` for regression loss. e.g. ['binary', 'regression'].
    :param task_names: list of str, indicating the predict target of each tasks.
    :param device: str, ``"cpu"`` or ``"cuda:0"``.
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.

    :return: A PyTorch model instance.
    """

    def __init__(self, dnn_feature_columns, num_experts=3, expert_dnn_hidden_units=(256, 128),
                 gate_dnn_hidden_units=(64,), tower_dnn_hidden_units=(64,), l2_reg_linear=0.00001,
                 l2_reg_embedding=0.00001, l2_reg_dnn=0,
                 init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False,
                 task_types=('binary', 'binary'), task_names=('ctr', 'ctcvr'), device='cpu', gpus=None,
                 use_autodis=False, use_transformers=False):
        super(MMOE, self).__init__(linear_feature_columns=[], dnn_feature_columns=dnn_feature_columns,
                                   l2_reg_linear=l2_reg_linear, l2_reg_embedding=l2_reg_embedding, init_std=init_std,
                                   seed=seed, device=device, gpus=gpus)
        self.num_tasks = len(task_names)
        if self.num_tasks <= 1:
            raise ValueError("num_tasks must be greater than 1")
        if num_experts <= 1:
            raise ValueError("num_experts must be greater than 1")
        if len(dnn_feature_columns) == 0:
            raise ValueError("dnn_feature_columns is null!")
        if len(task_types) != self.num_tasks:
            raise ValueError("num_tasks must be equal to the length of task_types")

        for task_type in task_types:
            if task_type not in ['binary', 'regression']:
                raise ValueError("task must be binary or regression, {} is illegal".format(task_type))

        self.use_autodis = use_autodis
        self.use_transfromers = use_transformers
        self.num_experts = num_experts
        self.task_names = task_names
        if use_autodis and use_transformers:
            # 2 transformers output + target_features
            self.input_dim = 72
        elif use_autodis:
            sparse_input_dim, dense_input_dim = self.compute_input_dim(dnn_feature_columns)
            self.input_dim = sparse_input_dim + dense_input_dim * 8
        else:
            sparse_input_dim, dense_input_dim = self.compute_input_dim(dnn_feature_columns)
            self.input_dim = sparse_input_dim + dense_input_dim

        self.expert_dnn_hidden_units = expert_dnn_hidden_units
        self.gate_dnn_hidden_units = gate_dnn_hidden_units
        self.tower_dnn_hidden_units = tower_dnn_hidden_units

        # autodis
        if self.use_autodis:
            options = AutoDisOptions(n_meta_embeddings=237, temperature=3.4177021326253723)
            self.autodis = AutoDis(55, 8, options)

        if use_transformers:
            # 2 transformers output + target_features
            self.input_dim = 72
            self.watched_list_multihead_attn = nn.MultiheadAttention(24, 8, kdim=400, vdim=400, batch_first=True)
            self.ordered_candidates_multihead_attn = nn.MultiheadAttention(24, 8, kdim=96, vdim=96, batch_first=True)

        # expert dnn
        self.expert_dnn = nn.ModuleList([DNN(self.input_dim, expert_dnn_hidden_units, activation=dnn_activation,
                                             l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                                             init_std=init_std, device=device) for _ in range(self.num_experts)])

        # gate dnn
        if len(gate_dnn_hidden_units) > 0:
            self.gate_dnn = nn.ModuleList([DNN(self.input_dim, gate_dnn_hidden_units, activation=dnn_activation,
                                               l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                                               init_std=init_std, device=device) for _ in range(self.num_tasks)])
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.gate_dnn.named_parameters()),
                l2=l2_reg_dnn)
        self.gate_dnn_final_layer = nn.ModuleList(
            [nn.Linear(gate_dnn_hidden_units[-1] if len(gate_dnn_hidden_units) > 0 else self.input_dim,
                       self.num_experts, bias=False) for _ in range(self.num_tasks)])

        # tower dnn (task-specific)
        if len(tower_dnn_hidden_units) > 0:
            self.tower_dnn = nn.ModuleList(
                [DNN(expert_dnn_hidden_units[-1], tower_dnn_hidden_units, activation=dnn_activation,
                     l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                     init_std=init_std, device=device) for _ in range(self.num_tasks)])
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.tower_dnn.named_parameters()),
                l2=l2_reg_dnn)
        self.tower_dnn_final_layer = nn.ModuleList([nn.Linear(
            tower_dnn_hidden_units[-1] if len(tower_dnn_hidden_units) > 0 else expert_dnn_hidden_units[-1], 1,
            bias=False)
            for _ in range(self.num_tasks)])

        self.out = nn.ModuleList([PredictionLayer(task) for task in task_types])

        regularization_modules = [self.expert_dnn, self.gate_dnn_final_layer, self.tower_dnn_final_layer]
        for module in regularization_modules:
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], module.named_parameters()), l2=l2_reg_dnn)
        self.to(device)

    def forward(self, X):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)

        # my code
        if self.use_autodis:
            dense_matrix = torch.stack(dense_value_list, dim=1)
            dense_matrix = torch.squeeze(dense_matrix, 2)
            dense_embedding_matrix = self.autodis(dense_matrix)

            dense_embedding_input = torch.flatten(dense_embedding_matrix, start_dim=1)
            sparse_embedding_input = torch.flatten(torch.cat(sparse_embedding_list, dim=-1), start_dim=1)

            dnn_input = torch.cat([sparse_embedding_input, dense_embedding_input], dim=-1)

            if self.use_transfromers:
                watched_list_features = None
                for index in range(0, 10):
                    # 32 = 4 dense features * 8 (dense dim)
                    # 8 = sparse dim
                    dense_embedding_input_for_video_i = dense_embedding_input[:, index * 32:(index + 1) * 32]
                    sparse_embedding_input_for_video_i = sparse_embedding_input[:, index * 8: (index + 1) * 8]
                    if watched_list_features is None:
                        watched_list_features = torch.cat(
                            [dense_embedding_input_for_video_i, sparse_embedding_input_for_video_i], dim=1)
                    else:
                        watched_list_features = torch.cat([watched_list_features, dense_embedding_input_for_video_i,
                                                           sparse_embedding_input_for_video_i], dim=1)

                # skip 'uid'
                ordered_candidates_features = dense_embedding_input[:, -3 * 8 * 5:-3 * 8]
                target_features = dense_embedding_input[:, -3 * 8:]

                watched_list_attn_output, attn_output_weights = self.watched_list_multihead_attn(target_features,
                                                                                                 watched_list_features,
                                                                                                 watched_list_features)
                ordered_candidates_attn_output, attn_output_weights = self.ordered_candidates_multihead_attn(
                    target_features, ordered_candidates_features, ordered_candidates_features)

                dnn_input = torch.cat([watched_list_attn_output, ordered_candidates_attn_output, target_features],
                                      dim=1)
        else:
            dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

        # expert dnn
        expert_outs = []
        for i in range(self.num_experts):
            expert_out = self.expert_dnn[i](dnn_input)
            expert_outs.append(expert_out)
        expert_outs = torch.stack(expert_outs, 1)  # (bs, num_experts, dim)

        # gate dnn
        mmoe_outs = []
        for i in range(self.num_tasks):
            if len(self.gate_dnn_hidden_units) > 0:
                gate_dnn_out = self.gate_dnn[i](dnn_input)
                gate_dnn_out = self.gate_dnn_final_layer[i](gate_dnn_out)
            else:
                gate_dnn_out = self.gate_dnn_final_layer[i](dnn_input)
            gate_mul_expert = torch.matmul(gate_dnn_out.softmax(1).unsqueeze(1), expert_outs)  # (bs, 1, dim)
            mmoe_outs.append(gate_mul_expert.squeeze())

        # tower dnn (task-specific)
        task_outs = []
        for i in range(self.num_tasks):
            if len(self.tower_dnn_hidden_units) > 0:
                tower_dnn_out = self.tower_dnn[i](mmoe_outs[i])
                tower_dnn_logit = self.tower_dnn_final_layer[i](tower_dnn_out)
            else:
                tower_dnn_logit = self.tower_dnn_final_layer[i](mmoe_outs[i])
            output = self.out[i](tower_dnn_logit)
            task_outs.append(output)
        task_outs = torch.cat(task_outs, -1)
        return task_outs
