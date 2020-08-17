import scipy.sparse
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
import scipy

logger = logging.getLogger()

class Model:

    def __init__(self):
        pass

    def fit(self, A_da, A_dd, T):
        self.A_da = A_da

        bert_model = SentenceTransformer('bert-base-nli-stsb-wkpooling')

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
