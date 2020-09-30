import scipy.sparse
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
import scipy

logger = logging.getLogger()
path = "/home/lj/tmp/pycharm_project_463/scripts/continue/output"
path2 = "/home/lj/tmp/pycharm_project_463/tests/output"
class Model:

    def __init__(self):
        pass

    def fit(self, A_da, A_dd, T):
        self.A_da = A_da
        maxLen = 0;
        for d in T:
            maxLen = max(maxLen, len(d.split()))
        print(maxLen)
        # bert_model = SentenceTransformer('bert-base-nli-mean-tokens')
        # bert_model = SentenceTransformer(path + "/doc_docsci_bert_nil_sts")
        # bert_model = SentenceTransformer(path + "/training_0allenai-scibert_scivocab_uncased")
        bert_model = SentenceTransformer(path + "/doc_doc_docsci_bert_08_04")
        # bert_model = SentenceTransformer(path2 + "/training_nli_allenai-scibert_scivocab_uncased-2020-09-21_15-36-32")
        bert_model._first_module().max_seq_length = 500

        # bert_model = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')
        # bert_model = SentenceTransformer('bert-base-nli-stsb-wkpooling')
        # bert_model = SentenceTransformer("roberta-large-nli-stsb-mean-tokens")
        self.docs_vectors = bert_model.encode(T)

    def predict(self, d, mask=None):
        query_vector = self.docs_vectors[d]
        documents_scores = np.squeeze(query_vector.dot(self.docs_vectors.T))

        # documents_scores = np.squeeze(query_vector.dot(self.docs_vectors.T).A)
        documents_sorting_indices = documents_scores.argsort()[::-1]
        document_ranks = documents_sorting_indices.argsort() + 1

        # Sort scores and get ranks // Rank sore = 1 / docuement_ranks
        candidates_scores = np.ravel(
            self.A_da.T.dot(scipy.sparse.diags(1 / document_ranks, 0)).T.sum(
                axis=0))  # A.T.dot(np.diag(b)) multiply each column of A element-wise by b

        if mask is not None:
            candidates_scores = candidates_scores[mask]
        return candidates_scores
