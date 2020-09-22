"""
An example script to use expert_finding as a package. Its shows how to load a dataset, create a model and run an evaluation.
"""
import os
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import expert_finding.io
from sklearn.utils import shuffle

import os
import logging

logger = logging.getLogger()

# Load one dataset
A_da, A_dd, T, L_d, L_d_mask, L_a, L_a_mask, tags = expert_finding.io.load_dataset("dblp")
# A_da, A_dd, T, L_d, L_d_mask, L_a, L_a_mask, tags = expert_finding.io.load_dataset("academia.stackexchange.com")
# A_da, A_dd, T, L_d, L_d_mask, L_a, L_a_mask, tags = expert_finding.io.load_dataset("mathoverflow.net")

"""
 同一label. 同一作者. 的document - document 对1.
"""

# print(A_da.shape[1])
# print("------")
print(A_da)
# print("------")
# print(A_dd)
# print(len(T))
# print("------")
# print(T[1640])
# print("------")

# print(A_dd.toarray())

matrix = A_dd.toarray()


def bulidPos():
    pos = open('pos_doc_doc_04.txt', mode='w')
    for i, a in enumerate(matrix):
        for j, b in enumerate(a):
            if matrix[i][j]:
                pos.writelines(str(T[i]) + '\t' + str(T[j]) + '\t' + "0.4" + '\n')
    pos.close()


def buildNeg():
    neg = open('neg_doc_doc_0.txt', mode='w')
    num = 0
    for i, a in enumerate(matrix):
        for j, b in enumerate(a):
            # print(matrix[i][j])
            if matrix[i][j] == False and num < 500:
                num += 1
                i = i + 1
                neg.writelines(str(T[i]) + '\t' + str(T[j]) + '\t' + "0" + '\n')
            continue
        continue
    neg.close()




# A_da
def buildSameAuthor():
    # 作者的所有文章.
    matrix = A_da.toarray()
    lists = [[] for i in range(707)]  # 创建
    # maxValue = 0
    pos = open('pos_doc_auth_07.txt', mode='w')
    for i, a in enumerate(matrix):
        for j, b in enumerate(a):
            if matrix[i][j]:
                # maxValue= max(maxValue, j)
                lists[j].append(i)
    print("lists is:", lists)
    # print(maxValue)
    for list in lists:
        for cur in list:
            for to in list:
                if cur==to: continue
                else:
                    pos.writelines(str(T[cur]) + '\t' + str(T[to]) + '\t' + "0.7" + '\n')
    pos.close()
def buildCo():
    # 文章的共同创作者.一篇文章的所有作者.
    matrix = A_da.toarray()
    lists = [[] for i in range(1641)]  #
    for i, a in enumerate(matrix):
        for j, b in enumerate(a):
            if matrix[i][j]:
                lists[i].append(j)


    print("list is", lists)

# buildNeg()
# bulidPos()
# buildSameAuthor()

# buildCo()
same = pd.read_table('./to_csv.txt', sep='\t', header=None)
df = shuffle(same)
print(same.head())
df.to_csv("./continue_train.csv",  header=False)

"""
A_da : adjacency matrix of the document-candidate network (scipy.sparse.csr_matrix)
A_dd : adjacency matrix of the document-document network (scipy.sparse.csr_matrix)
T : raw textual content of the documents (numpy.array)
L_d : labels associated to the document (corresponding to T[L_d_mask]) (numpy.array)
L_d_mask : mask to select the labeled documents (numpy.array)
L_a : labels associated to the candidates (corresponding to A_da[:,L_d_mask]) (numpy.array)
L_a_mask : mask to select the labeled candidates (numpy.array)
tags : names of the labels of expertise (numpy.array)
"""
