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
import random

logger = logging.getLogger()

# Load one dataset
# A_da, A_dd, T, L_d, L_d_mask, L_a, L_a_mask, tags = expert_finding.io.load_dataset("dblp")
A_da, A_dd, T, L_d, L_d_mask, L_a, L_a_mask, tags = expert_finding.io.load_dataset("academia.stackexchange.com")
# A_da, A_dd, T, L_d, L_d_mask, L_a, L_a_mask, tags = expert_finding.io.load_dataset("mathoverflow.net")

"""
 同一label. 同一作者. 的document - document 对1.
"""

# print(A_da.shape[1])
# print("------")
# print(A_da)
# print("------")
# print(A_dd)
# print(len(T))
# print("------")
# print(T[1640])
# print("------")

# print(A_dd.toarray())

matrix = A_dd.toarray()
def bulidPos():
    pos = open('document.txt', mode='w')
    for i, a in enumerate(matrix):
        for j, b in enumerate(a):
            if matrix[i][j]:
                pos.writelines(str(T[i]) + '\t' + str(T[j]) + '\t' + "0.4" + '\n')
    pos.close()


def buildNeg():
    neg = open('.txt', mode='w')
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
    # authors documents
    matrix = A_da.toarray()
    lists = [[] for i in range(6030)]  # 创建
    # maxValue = 0
    pos = open('academia_author_docs.txt', mode='w')
    for i, a in enumerate(matrix):
        for j, b in enumerate(a):
            if matrix[i][j]:
                # maxValue= max(maxValue, j)
                lists[j].append(i)
    print("lists is:", lists)

    for i, a in enumerate(lists):
        for j, b in enumerate(a):
            pos.write(str(i) +'\t' + str(b) + '\n')
    pos.close()

    author_doc_list = open('author_academia_doc_list.txt', mode='w')
    for i, a in enumerate(lists):
        string = str(i) + '\t'
        for j, b in enumerate(a):
            string += ' ' + str(b)
        author_doc_list.write(string + '\n')
    author_doc_list.close()



def buildCo():
    # document _ authors
    #pos = open('docs_author.txt', mode='w')
    matrix = A_da.toarray()

    docs_author_list = open('academia_docs_author_list.txt', mode='w')
    lists = [[] for i in range(20800)]

    for i, a in enumerate(matrix):
        string = str(i) + '\t'
        for j, b in enumerate(a):
            if matrix[i][j]:
                # lists[i].append(j)
                string += ' ' + str(j)
        docs_author_list.write(string + '\n')
    docs_author_list.close()
    # for list in lists:
    #     for cur in list:
    #         for to in list:
    #             if cur == to:
    #                 continue
    #             else:
    #                 pos.writelines(str(cur) + '\t' + str(to) + '\t' + "1" + '\n')
    # pos.close()
    print("list is", docs_author_list)


# buildNeg()
# bulidPos()
# buildSameAuthor()
# buildCo()
# same = pd.read_table('./to_csv.txt', sep='\t', header=None)
# df = shuffle(same)
# print(same.head())
# df.to_csv("./continue_train.csv",  header=False)

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
def build_triplet():
    author_docs_list = pd.read_table('./author_academia_doc_list.txt', sep='\t', names=['author', 'document'])
    # author_docs_list.
    docs_author_list = pd.read_table('./academia_docs_author_list.txt', sep='\t', names=['document', 'author'])
    # docs_author_list.iloc[:115]

    ## concat
    doc_triples = pd.DataFrame(columns=['A', 'POS', 'NEG'])
    for i in range(len(docs_author_list)):
        author_str = docs_author_list.loc[docs_author_list['document'] == i, 'author']
        # print(author_str)
        author_arr = author_str[i]  # 第i 篇文章的所有作者
        relevant = set()
        # for e in author_arr:
        document_str = author_docs_list.loc[author_docs_list['author'] == author_arr, 'document']
        document_arr = document_str[author_arr].split()  # 第 e 这个作者的所有文章。
        for d in document_arr:
            relevant.add(d)

        negs = list(set(range(0, 20799)).difference(set(relevant)))

        for to in relevant:
            neg = random.choice(negs)
            doc_triples = doc_triples.append(pd.DataFrame({'A': [i], 'POS': [to], 'NEG': [neg]}), ignore_index=True)

        #docs_author_list.loc[i, 'documents'] = ','.join(relevant)

        doc_triples.to_csv('academia_doc_triples.csv')

build_triplet()