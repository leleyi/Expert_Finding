import numpy as np
import expert_finding.preprocessing.text.dictionary
import expert_finding.preprocessing.text.vectorizers
import scipy.sparse
import logging
logger = logging.getLogger()
from sklearn.preprocessing import normalize
from gensim.summarization import bm25
from sentence_transformers import SentenceTransformer
import pandas as pd
import random
path = "/home/lj/tmp/pycharm_project_463/scripts/continue/output"
class Model:

    def __init__(self):
        pass

    def fit(self, A_da, A_dd, T):
        self.A_da=A_da
        logger.debug("Building vocab")
        self.vocab = expert_finding.preprocessing.text.dictionary.Dictionary(T, min_df=20, max_df_ratio=0.25)
        logger.debug("Building tfidf vectors")
        self.docs_vectors = expert_finding.preprocessing.text.vectorizers.get_tfidf_dictionary(self.vocab)

        bert_model = SentenceTransformer(path + "/sci_bert_nil_sts")

        bert_model._first_module().max_seq_length = 500

        self.embedding_docs_vectors = normalize(bert_model.encode(T), norm='l2', axis=1)
        # self.bm25Model = bm25(T)
        # self.average_idf = sum(map(lambda k: float(self.bm25Model.idf[k]), self.bm25Model.idf.keys())) / len(self.bm25Model.idf.keys())
    def predict(self, d, mask = None):

        # scores = self.bm25Model.get_scores(d, self.average_idf)
        # print("tell me the scores :", scores)
        query_vector = self.docs_vectors[d]

        query_vector_emb = self.embedding_docs_vectors[d]
        embedding_scores = np.squeeze(query_vector_emb.dot(self.embedding_docs_vectors.T))
        embedding_scores_sorting_indices = embedding_scores.argsort()[::-1]

        documents_scores = np.squeeze(query_vector.dot(self.docs_vectors.T).A)
        documents_sorting_indices = documents_scores.argsort()[::-1]
        document_ranks = documents_sorting_indices.argsort() + 1
        # Sort scores and get ranks
        # save top 5 lex result
        relevant_lexical = documents_sorting_indices[0:5] # top 5
        irrelevant_lexical = documents_sorting_indices[1000:1600]
        relevant_semantic = embedding_scores_sorting_indices[0:5]
        irrelevant_semantic = embedding_scores_sorting_indices[1000:1600]

        # negs = list(set(range(0, 1640)).difference(set(relevant)))
        doc_triples = pd.DataFrame(columns=['A', 'POS', 'NEG'])
        for i in relevant_lexical:
            neg = random.choice(irrelevant_lexical)
            while neg not in irrelevant_semantic:
                neg = random.choice(irrelevant_lexical)

            doc_triples = doc_triples.append(pd.DataFrame({'A': [d], 'POS': [i], 'NEG': [neg]}), ignore_index=True)
        doc_triples.to_csv('lexical_triple.csv', mode='a', index=True, header=None)

        candidates_scores = np.ravel(
            self.A_da.T.dot(scipy.sparse.diags(1 / document_ranks, 0)).T.sum(
                axis=0))  # A.T.dot(np.diag(b)) multiply each column of A element-wise by b
        if mask is not None:
            candidates_scores = candidates_scores[mask]
        return candidates_scores



