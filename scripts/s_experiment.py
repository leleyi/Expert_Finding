"""
An example script to use expert_finding as a package. Its shows how to load a dataset, create a model and run an evaluation.
"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import expert_finding.io
import expert_finding.evaluation
import expert_finding.models.random_model

import expert_finding.models.voting_model
import expert_finding.models.bert_propagation_model

import os
import logging

logger = logging.getLogger()

# Load one dataset
A_da, A_dd, T, L_d, L_d_mask, L_a, L_a_mask, tags = expert_finding.io.load_dataset("dblp")
# A_da, A_dd, T, L_d, L_d_mask, L_a, L_a_mask, tags = expert_finding.io.load_dataset("academia.stackexchange.com")
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

model = expert_finding.models.bert_voting_model.Model()
eval_batches, merged_eval = expert_finding.evaluation.run(model, A_da, A_dd, T, L_d, L_d_mask, L_a, L_a_mask, tags)



