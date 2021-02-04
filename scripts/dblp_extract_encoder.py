"""
An example script to use expert_finding as a package. Its shows how to load a dataset, create a model and run an evaluation.
"""
import os


import expert_finding.io
import expert_finding.models.bert_propagation_model
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
import pickle

A_da, A_dd, T, L_d, L_d_mask, L_a, L_a_mask, tags = expert_finding.io.load_dataset("dblp")

path = "/ddisk/lj/tmp/pycharm_project_332/expert_finding/resources/"
encoder_path = "/ddisk/lj/DBLP/data/models/"
modle_list = ["sci_bert_nil_sts",
              "academia_author_triplet",
              "doc_doc_sci_bert_nil_sts_triples",
              "doc_doc_sci_bert_triples_nil_sts"]


def load_encoder(encoder_path):
    bert_model = SentenceTransformer(encoder_path)
    bert_model._first_module().max_seq_length = 500
    return bert_model


def save_embedding(encoder_path, save_embedding_path, encoder_name):
    print("loading encoder .. ")
    model = load_encoder(encoder_path + encoder_name)
    print("embedding documents ..")
    embedding_docs_vectors = normalize(model.encode(T), norm='l2', axis=1)
    print("saving embedding .. ")
    with open(save_embedding_path + encoder_name, "wb") as f:
        pickle.dump(embedding_docs_vectors, f)


save_embedding(encoder_path, path, modle_list[1])
