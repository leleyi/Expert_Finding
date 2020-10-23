"""
An example script to use expert_finding as a package. Its shows how to load a dataset, create a model and run an evaluation.
"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import expert_finding.io
import expert_finding.evaluation
import expert_finding.models.random_model

import expert_finding.models.voting_model
import expert_finding.models.bert_propagation_model

import os
import logging

logger = logging.getLogger()

# Load one dataset
# A_da, A_dd, T, L_d, L_d_mask, L_a, L_a_mask, tags = expert_finding.io.load_dataset("dblp")
A_da, A_dd, T, L_d, L_d_mask, L_a, L_a_mask, tags = expert_finding.io.load_dataset("academia.stackexchange.com")
# A_da, A_dd, T, L_d, L_d_mask, L_a, L_a_mask, tags = expert_finding.io.load_dataset("mathoverflow.net")
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


# You can load a model

import expert_finding.models.bert_voting_model
import expert_finding.models.hybrid_voting_model

# bert_model = SentenceTransformer(path + "/sci_bert_nil")
# bert_model = SentenceTransformer(path + "/sci_bert_nil_sts")
# bert_model = SentenceTransformer(path + "/doc_doc_sci_bert_triples")
# bert_model = SentenceTransformer(path + "/doc_doc_sci_bert_nil_sts_triples")
# bert_model = SentenceTransformer(path + "/doc_doc_sci_bert_triples_nil_sts")
# bert_model = SentenceTransformer(path + "/doc_doc_sci_bert_triples_lexical")
model_list = [
              '/sci_bert_nil',
              '/sci_bert_nil_sts',
              '/doc_doc_sci_bert_triples',
              '/doc_doc_sci_bert_nil_sts_triples',
              '/doc_doc_sci_bert_triples_nil_sts',
              '/doc_doc_sci_bert_triples_lexical',
              '/doc_doc_sci_bert_fusion_triples']

import expert_finding.models.hybrid_propagation_model
model = expert_finding.models.hybrid_propagation_model.Model()
hybird = {}
para = {}
para["i"] = 1
para["j"] = 1
hybird["k"] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for i in model_list:
    for k in hybird["k"]:
        para["k"] = k
        eval_batches, merged_eval = expert_finding.evaluation.run_multi(model, A_da, A_dd, T, L_d, L_d_mask, L_a, L_a_mask,
                                                                    tags, para=para, model_name=i)



