import sys
import os
import multiprocessing

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import expert_finding.preprocessing.text.dictionary
import expert_finding.preprocessing.text.vectorizers
import expert_finding.data.sets
import scipy.sparse
import sklearn
import numpy as np
import expert_finding.data.io
import time

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

print(" ================== " + tf.test.is_gpu_available())
def sava_tf_matrix():
    data_path = "/ddisk/lj/DBLP/data/"
    data_version = ["/V1", "/V2", "/V3"]
    for version in data_version:
        dataset = expert_finding.data.sets.DataSet("aminer")
        data_dir = data_path + version + "/dataset_cleaned"
        dataset.load(data_dir)
        print("building tf_matrix ..." + version)
        vocab = expert_finding.preprocessing.text.dictionary.Dictionary(dataset.ds.documents)
        M = get_tf_dictionary(vocab)
        scipy.sparse.save_npz((data_dir + "/embedding/tf_matrix"), M)
        print("save success : " + data_dir + "/embedding/tf_matrix")


def get_tf_dictionary(dictionary):
    start = time.time()
    tf_matrix = scipy.sparse.csr_matrix((dictionary.num_docs, dictionary.num_words))
    for k, seq in enumerate(dictionary.docs_seqs):
        tf_matrix += scipy.sparse.csr_matrix(
            (np.ones(dictionary.docs_lens[k]),
             (k * np.ones(dictionary.docs_lens[k]),
              seq)),
            shape=(dictionary.num_docs, dictionary.num_words)
        )
    tf_matrix = sklearn.preprocessing.normalize(tf_matrix, norm='l1', axis=1)
    end = time.time()
    # expert_finding.data.io.check_and_create_dir(data_path + "embedding/tf_matrix")
    print("ser spend time : " + str(end - start))
    return tf_matrix


data_path = "/ddisk/lj/DBLP/data/"
dataset = expert_finding.data.sets.DataSet("aminer")
data_dir = data_path + "/V2" + "/dataset_associations"
dataset.load(data_dir)
# print("building tf_matrix ..." + "/V1")
# dictionary = expert_finding.preprocessing.text.dictionary.Dictionary(dataset.ds.documents)


# print("star process ...")


def process(item):
    A = scipy.sparse.coo_matrix(
        (np.ones(dictionary.docs_lens[item[0]]),
         (item[0] * np.ones(dictionary.docs_lens[item[0]]),
          item[1])),
        shape=(dictionary.num_docs, dictionary.num_words)
    )
    return A


def muti_process(dictionary):
    start = time.time()
    pool = multiprocessing.Pool(10)
    items = [(x, dictionary.docs_seqs[x]) for x in range(0, len(dictionary.docs_seqs))]
    res = pool.map(process, items)
    its = time.time()
    print("before sum :" + str(its - start))

    matrix = sum(res)
    # matrix = sum(res)
    # total = delayed(sum)(res)
    # matrix = total.compute()
    ite = time.time()
    print("it spend time : " + str(ite - its))

    pool.close()
    pool.join()
    matrix = sklearn.preprocessing.normalize(matrix, norm='l1', axis=1)
    end = time.time()
    print("muti spend time : " + str(end - start))
    return matrix


# print(len(dictionary.docs_seqs))
# print(type(dictionary.docs_seqs))
# # items = [(x, dictionary.docs_seqs[x]) for x in range(0, len(dictionary.docs_seqs))]
# # print(items[0])
# matrix1 = muti_process()


# print(matrix1)
# print("-===============-")
# print(matrix2)
# print(dictionary.docs_seqs[0])

def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)


def get_tf_dictionary_with_tf(dictionary):
    start = time.time()

    # tf_matrix = scipy.sparse.csr_matrix((dictionary.num_docs, dictionary.num_words))

    sess = tf.Session(
        config=tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list='0', allow_growth=True),
                              log_device_placement=True))
    sess.run(tf.compat.v1.global_variables_initializer())
    with tf.device("GPU:0"):
        tf_matrix = tf.sparse.SparseTensor(indices=[[0, 0], [0, 1]], values=[0, 0],
                                           dense_shape=[dictionary.num_docs, dictionary.num_words])
        for k, seq in enumerate(dictionary.docs_seqs):
            indices = np.vstack(([k] * dictionary.docs_lens[k], seq)).transpose()
            matrix = tf.sparse.SparseTensor(indices=indices,
                                            values=[1] * dictionary.docs_lens[k],
                                            dense_shape=[dictionary.num_docs, dictionary.num_words])
            tf_matrix = tf.sparse_add(tf_matrix, matrix)
            # sess.run(tf_matrix)
    # tf_matrix = sklearn.preprocessing.normalize(tf_matrix, norm='l1', axis=1)
    end = time.time()
    print("ser spend time : " + str(end - start))
    return tf_matrix


# @jit(nopython=True)
def jit_get_tf_dictionary(docs_num, words_num, docs_seqs, docs_lens):
    mat = np.zeros((docs_num, words_num), dtype='uint8')
    for k, seq in enumerate(docs_seqs):
        for idx in range(0, docs_lens[k]):
            mat[k][seq[idx]] += 1
    # M = scipy.sparse.csr_matrix(matrix)
    # tf_matrix = sklearn.preprocessing.normalize(M, norm='l1', axis=1)
    # print("ser spend time : " + str(end - start))
    return mat


# matrix1 = get_tf_dictionary(dictionary)


def excu(dictionary):
    start = time.time()
    a = dictionary.num_docs
    b = dictionary.num_words
    c = dictionary.docs_seqs
    d = dictionary.docs_lens
    print("row : " + str(a) + " col : " + str(b))
    matrix2 = jit_get_tf_dictionary(a, b, c, d)
    M = scipy.sparse.csr_matrix(matrix2)
    end = time.time()
    print("ser spend time : " + str(end - start))


# sava_tf_matrix()

# scipy.sparse.save_npz(os.path.join(data_dir, "/embedding/tf_matrix"))
# tf_matrix = scipy.sparse.load_npz(data_dir + "/embedding/tf_matrix.npz")
# print(tf_matrix)
