"""
An example script to use expert_finding as a package. Its shows how to load a dataset, create a model and run an evaluation.
"""
import logging
import os
import pickle
import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

logger = logging.getLogger()
import numpy as np
import time
import warnings
import expert_finding.preprocessing.text.dictionary
import expert_finding.data.sets
import expert_finding.preprocessing.text.vectorizers
import expert_finding.models.data_generator
import expert_finding.models.tools
from expert_finding.models.transformer import activation_attention, cosine_loss

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import scipy.spatial.distance
import expert_finding.preprocessing.graph.random_walker
import expert_finding.preprocessing.graph.window_slider

import expert_finding.io
import expert_finding.evaluation
import expert_finding.models.random_model
import expert_finding.models.voting_idne_model
import expert_finding.models.idne
import expert_finding.models.gvnrt
import expert_finding.models.voting_tadw_model
import expert_finding.models.tadw
import expert_finding.models.graph2gauss
import expert_finding.models.post_ane_model
import expert_finding.models.bert_propagation_model

import os
import logging

# logger = logging.getLogger()


A_da, A_dd, T, L_d, L_d_mask, L_a, L_a_mask, tags = expert_finding.io.load_dataset("dblp")
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# tf = tf.compat.v1
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
print(" ================== " + str(tf.test.is_gpu_available()))

def print_detail_info():
    print("documents num : " + str(len(T)))
    print("labels associated to the document: " + str(L_d.shape[0]))
    print("names of the labels of expertise: " + str(len(tags)))
    print("labels associated to the candidates : " + str(L_a.shape[1]))
    print("link nums of d_a: " + str(A_da.shape[0]) + " * " + str(A_da.shape[1]))
    print("link nums of d_a: " + str(A_da.nnz))
    print("link nums of d_a: " + str(A_da.nnz))
    print("+++++++++++++++++++++++++++")
    print("link nums of d_d: " + str(A_dd.nnz))
    print("link nums of d_d: " + str(A_dd.nnz))


# def save_embedding(model_list):
#     for name in model_list:
#         data.io.check_and_create_dir(path)
#         r_path = path
#         if name is "/sci_bert_nil_sts":
#             r_path = dir
#         with open(path + "/Embedding" + name, "wb") as f:
#             pickle.dump(embedding_docs_vectors, f)


def save_network_embdding(data_path, A_da, A_dd, T):
    print("construct network")
    dd = A_da @ A_da.T  # 矩阵乘法. ->  i,j = i 行 * j 列
    dd.setdiag(0)
    # network = dd + A_dd
    network = dd + A_dd
    documents = T

    # print("init_model_g2g")
    # model_g2g = expert_finding.models.post_ane_model.Model(expert_finding.models.graph2gauss.Model)
    # model_g2g.fit(A_da, A_dd, T)
    # embeddings_g2g = model_g2g.embeddings
    # with open(data_path + "/embedding/g2g", "wb") as f:
    #     pickle.dump(embeddings_g2g, f)

    print("init_model_gvnrt")
    model_gvnrt = expert_finding.models.post_ane_model.Model(expert_finding.models.gvnrt.Model)
    model_gvnrt.fit(A_da, A_dd, T)
    embeddings_gvnrt = model_gvnrt.embeddings
    with open(data_path + "/embedding/gvnrt", "wb") as f:
        pickle.dump(embeddings_gvnrt, f)

    print("init_model_tadw")
    model_tadw = expert_finding.models.tadw.Model()
    model_tadw.fit(network, documents)
    embeddings_tadw = model_tadw.get_embeddings()
    with open(data_path + "/embedding/tadw", "wb") as f:
        pickle.dump(embeddings_tadw, f)

    model_idne = expert_finding.models.idne.Model()
    model_idne.fit(network, documents)
    embeddings_idne = model_idne.get_embeddings()
    with open(data_path + "/embedding/idne", "wb") as f:
        pickle.dump(embeddings_idne, f)


def save_document_embedding():
    print("")


#
dataset = expert_finding.data.sets.DataSet("aminer")
data_path = "/ddisk/lj/DBLP/data/V2/dataset_cleaned"
dataset.load(data_path)
A_da = dataset.ds.associations
A_dd = dataset.ds.citations
T = dataset.ds.documents

save_network_embdding(data_path, A_da, A_dd, T)


# eval_batches, merged_eval = expert_finding.evaluation.run(model, A_da, A_dd, T, L_d, L_d_mask, L_a, L_a_mask, tags)
# print_detail_info()
# print(tf.test.is_gpu_available())
# print(tf.test.is_built_with_cuda())


def bulid_data_for_query():
    path_lj = "/ddisk/lj"
    data_type = ["/dataset_associations", "/dataset_cleaned"]
    data_set = ["/DBLP"]
    data_version = ["/V1", "/V2"]
    data_path = path_lj + data_set[0] + "/data" + data_version[0] + data_type[1]
    dataset = expert_finding.data.sets.DataSet("aminer")
    dataset.load(data_path)
    # dataset.gt.candidates
    # dataset.gt.experts_mask
    # dataset.gt.associations

    doucments = dataset.ds.documents
    candidates = dataset.ds.candidates
    topics = dataset.gt.topics  # d-t  = 2 || t-d = 5

    associations = dataset.ds.associations  # d-a type = 0 || a-d = 3
    citations = dataset.ds.citations  # d-d cite = 1 || be cited = 4

    print("total node num : " + str(len(doucments) + len(candidates) + len(topics)))
    print("documents node num : " + str(len(doucments)))
    print("author node num : " + str(len(candidates)))
    print("topic node num : " + str(len(topics)))
    print("===================行动开始===================")


bulid_data_for_query()
