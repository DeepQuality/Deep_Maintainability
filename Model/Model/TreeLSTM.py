from collections import defaultdict

import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class Node:
    def __init__(self, label="", parent=None, children=None, num=0):
        if children is None:
            children = []
        self.label = label
        self.parent = parent
        self.children = children
        self.num = num


class TreeEmbeddingLayer(tf.keras.Model):
    def __init__(self, dimensionOfEmbedding, sizeOfVocabulary):
        super(TreeEmbeddingLayer, self).__init__()
        self.E = tf.compat.v1.get_variable("E", [sizeOfVocabulary, dimensionOfEmbedding], tf.float32,
                                           initializer=tf.keras.initializers.RandomUniform())

    def call(self, x):
        lengths = [temp.shape[0] for temp in x]
        embeddings = tf.nn.embedding_lookup(self.E, tf.concat(x, axis=0))
        embeddings = tf.split(embeddings, lengths, 0)
        return embeddings


class ChildSumLSTMLayer(tf.keras.layers.Layer):
    def __init__(self, dimensionOfInput, dimensionOfOutput, name='ChildSumLSTMLayer'):
        super(ChildSumLSTMLayer, self).__init__(name=name)

        self.dimensionOfInput = dimensionOfInput
        self.dimensionOfOutput = dimensionOfOutput
        self.U_f = tf.keras.layers.Dense(input_shape=(self.dimensionOfInput,), units=dimensionOfOutput,
                                         use_bias=False, name='TreeLSTM_U_f')
        self.U_iuo = tf.keras.layers.Dense(input_shape=(self.dimensionOfInput,), units=dimensionOfOutput * 3,
                                           use_bias=False, name='TreeLSTM_U_iuo')
        self.W = tf.keras.layers.Dense(input_shape=(self.dimensionOfInput,), units=dimensionOfOutput * 4,
                                       name='TreeLSTM_W')
        self.h_init = tf.zeros([1, dimensionOfOutput], tf.float32)
        self.c_init = tf.zeros([1, dimensionOfOutput], tf.float32)

    def call(self, tensor, indices):
        h_tensor = self.h_init
        c_tensor = self.c_init
        res_h, res_c = [], []
        for indexes, x in zip(indices, tensor):
            h_tensor, c_tensor = self.apply(x, h_tensor, c_tensor, indexes)
            h_tensor = tf.concat([self.h_init, h_tensor], 0)
            c_tensor = tf.concat([self.c_init, c_tensor], 0)
            res_h.append(h_tensor[1:, :])
            res_c.append(c_tensor[1:, :])
        return res_h, res_c

    def apply(self, x, h_tensor, c_tensor, indexes):
        mask_bool = tf.not_equal(indexes, -1)
        # [batch, child]
        mask = tf.cast(mask_bool, tf.float32)
        # [nodes, child, dim]
        h = tf.gather(h_tensor, tf.where(mask_bool,
                                         indexes, tf.zeros_like(indexes)))
        c = tf.gather(c_tensor, tf.where(mask_bool,
                                         indexes, tf.zeros_like(indexes)))
        # [nodes, dim_out]
        h_sum = tf.reduce_sum(h * tf.expand_dims(mask, -1), 1)

        # [nodes, dim_out * 4]
        W_x = self.W(x)
        # [nodes, dim_out]
        W_f_x = W_x[:, :self.dimensionOfOutput * 1]
        W_i_x = W_x[:, self.dimensionOfOutput * 1:self.dimensionOfOutput * 2]
        W_u_x = W_x[:, self.dimensionOfOutput * 2:self.dimensionOfOutput * 3]
        W_o_x = W_x[:, self.dimensionOfOutput * 3:]

        branch_f_k = tf.reshape(self.U_f(tf.reshape(h, [-1, h.shape[-1]])), h.shape)
        branch_f_k = tf.sigmoid(tf.expand_dims(W_f_x, 1) + branch_f_k)
        # [node, dim_out]
        branch_f = tf.reduce_sum(branch_f_k * c * tf.expand_dims(mask, -1), 1)

        # [nodes, dim_out * 3]
        branch_iuo = self.U_iuo(h_sum)
        # [nodes, dim_out]
        branch_i = tf.sigmoid(branch_iuo[:, :self.dimensionOfOutput * 1] + W_i_x)
        branch_u = tf.tanh(branch_iuo[:, self.dimensionOfOutput * 1:self.dimensionOfOutput * 2] + W_u_x)
        branch_o = tf.sigmoid(branch_iuo[:, self.dimensionOfOutput * 2:] + W_o_x)

        # [node, dim_out]
        new_c = branch_i * branch_u + branch_f
        # [node, dim_out]
        new_h = branch_o * tf.tanh(new_c)
        return new_h, new_c


def parse(ast):
    roots = []
    index = 0
    numberOfTrees = ast[index]
    index += 1
    for _ in range(numberOfTrees):
        num_nodes = ast[index]
        index += 1
        nodes = []
        for i in range(int(num_nodes)):
            nodes.append(Node(num=i, children=[]))
        for i in range(num_nodes):
            label = ast[index]
            index += 1
            label = tf.cast(label, dtype=tf.int64)
            nodes[i].label = label
        num_edges = ast[index]
        index += 1
        for i in range(num_edges):
            p = ast[index]
            index += 1
            c = ast[index]
            index += 1
            nodes[p].children.append(nodes[c])
            nodes[c].parent = nodes[p]
        roots.append(nodes[0])
    return roots


def tree2tensor(trees):
    """
    indexes:
        this has structure data.
        0 represent init state,
        1<n represent children's number (1-indexed)
    depths:
        these are labels of nodes at each depth.
    tree_num:
        explain number of tree that each node was contained.
    """
    res = defaultdict(list)
    for e, depth2nodes in enumerate(trees):
        for k, v in depth2nodes.items():
            res[k] += v

    for k, v in res.items():
        for e, n in enumerate(v):
            n.num = e + 1
    depths = [x[1] for x in sorted(res.items(), key=lambda x: -x[0])]
    indices = [getNumbers(nodes) for nodes in depths]
    depths = [np.array([node.label for node in nodes], np.int32) for nodes in depths]
    return depths, indices


def getNumbers(roots):
    """convert roots to indexes"""
    res = [[child.num for child in node.children] if node.children != [] else [0] for node in roots]
    maxLength = max([len(x) for x in res])
    res = tf.keras.preprocessing.sequence.pad_sequences(res, maxLength,
                                                        padding="post", value=-1.)
    return tf.constant(res, tf.int32)

