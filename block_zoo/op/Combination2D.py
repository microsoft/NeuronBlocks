# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import logging

from block_zoo.BaseLayer import BaseConf
from utils.DocInherit import DocInherit
from utils.exceptions import ConfigurationError
import copy

class Combination2DConf(BaseConf):
    """ Configuration for combination layer

    Args:
        operations (list):  a subset of ["origin", "difference", "dot_multiply"].
                "origin" means to keep the original representations;\n
                "difference" means abs(sequence1 - sequence2);
                "dot_multiply" means element-wised product;

    """
    def __init__(self, **kwargs):
        super(Combination2DConf, self).__init__(**kwargs)
        if "bilinear" in self.operations:
            #bilinear_trans = nn.Linear(self.output_dim, self.output_dim)

            #ones=torch.Tensor(np.ones([2,2,3,3])) 
            #w.weight=torch.nn.Parameter(ones)

            #w = torch.empty(3, 5)
            #nn.init.uniform_(w)

            weight_bilinear = Parameter(torch.Tensor(self.output_dim, self.output_dim))


    @DocInherit
    def default(self):
        # supported operations: "origin", "difference", "dot_multiply"
        self.operations = ["origin", "difference", "dot_multiply", "cosine", "bilinear", "tensor"]

    @DocInherit
    def declare(self):
        self.num_of_inputs = -1
        self.input_ranks = [-1]

    @DocInherit
    def inference(self):
        #说明上一层的dim就有问题！
        self.output_dim = [self.input_dims[0][0], self.input_dims[0][1], self.input_dims[0][1]]
        #print("*&*&*")
        #print(self.output_dim)   #[[-1, -1, 256], [-1, -1, 256], [-1, -1, 256]]
        #copy.deepcopy(self.input_dims[0])
        '''
        self.output_dim[-1] = 0
        if "origin" in self.operations:
            self.output_dim[-1] += sum([input_dim[-1] for input_dim in self.input_dims])
        if "difference" in self.operations:
            self.output_dim[-1] += int(np.mean([input_dim[-1] for input_dim in self.input_dims]))     # difference operation requires dimension of all the inputs should be equal
        if "dot_multiply" in self.operations:
            self.output_dim[-1] += int(np.mean([input_dim[-1] for input_dim in self.input_dims]))     # dot_multiply operation requires dimension of all the inputs should be equal
        if "cosine" in self.operations:
            self.output_dim[-1]
        '''
        super(Combination2DConf, self).inference()

    @DocInherit
    def verify(self):
        super(Combination2DConf, self).verify()

        # to check if the ranks of all the inputs are equal
        rank_equal_flag = True
        for i in range(len(self.input_ranks)):
            if self.input_ranks[i] != self.input_ranks[0]:
                rank_equal_flag = False
                break
        if rank_equal_flag == False:
            raise ConfigurationError("For layer Combination, the ranks of each inputs should be consistent!")

        if "difference" in self.operations:
            assert len(self.input_dims) == 2, "Difference operation requires that there should be two inputs"

        if "difference" in self.operations or "dot_multiply" in self.operations:
            input_dims = list(self.input_dims)
            dim_equal_flag = True
            for i in range(len(input_dims)):
                if input_dims[i] != input_dims[0]:
                    dim_equal_flag = False
                    break
            if dim_equal_flag == False:
                raise Exception("Difference and dot multiply require that the input dimensions should be the same")


class Combination2D(nn.Module):
    """ Combination layer to merge the representation of two sequence

    Args:
        layer_conf (CombinationConf): configuration of a layer
    """
    def __init__(self, layer_conf):
        super(Combination2D, self).__init__()
        self.layer_conf = layer_conf
        #print("**************")
        #print(layer_conf.input_dims)   #[[-1, -1, 256], [-1, -1, 256]]
        #self.linear_bi = nn.Linear(layer_conf.input_dims[0][2], layer_conf.input_dims[0][2])
        #self.linear_ten1 = nn.Linear(layer_conf.input_dims[0][2], layer_conf.input_dims[0][2])
        #self.linear_ten2 = nn.Linear(layer_conf.input_dims[0][1], layer_conf.input_dims[0][2])

        logging.warning("The length Combination layer returns is the length of first input")

    def forward(self, *args):
        """ process inputs

        Args:
            args (list): [string, string_len, string2, string2_len, ...]
                e.g. string (Variable): [batch_size, dim], string_len (ndarray): [batch_size]

        Returns:
            Variable: [batch_size, output_dim], None

        """
        '''
        输入两个string都是[batch_size, seq_len, hidden_size]的
        生成的每个矩阵都是[batch_size, seq_len, seq_len]的
        最后输出的应该是[batch_size, num_map, seq_len, seq_len]的
        '''
        #print("&*&*&*")
        #print(args[0].size())   #[8, 15, 256]  每次长度都不一样
        #print(args[2].size())   #[8, 34, 256]
        result = []
        if "cosine" in self.layer_conf.operations:
            string1 = args[0]
            string2 = args[2]
            result_multiply = torch.matmul(string1, string2.transpose(1,2))
            result.append(torch.unsqueeze(result_multiply, 1))
            #result.append(torch.unsqueeze(result_multiply, 1))

        if "bilinear" in self.layer_conf.operations:
            string1 = args[0]
            string2 = args[2]
            string1 = linear_bi(string1)
            result_multiply = torch.matmul(string1, string2.transpose(1,2))
            result.append(torch.unsqueeze(result_multiply, 1))

        if "tensor" in self.layer_conf.operations:
            string1 = args[0]
            string2 = args[2]
            c = 10
            for i in range(0, c):
                string1 = linear_ten_1(string1)
                result_multiply = torch.matmul(string1, string2.transpose(1,2))
                string_concat = torch.cat((string1, string2), 1)

        return torch.cat(result, 1), args[1]

        #return torch.cat(result, 1), args[1]  

