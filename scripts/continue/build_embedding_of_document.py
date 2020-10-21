import os

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import expert_finding.io
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import expert_finding.preprocessing.text.dictionary
import expert_finding.preprocessing.text.vectorizers
import logging
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, losses, models, util
logger = logging.getLogger()
# Load one dataset
A_da, A_dd, T, L_d, L_d_mask, L_a, L_a_mask, tags = expert_finding.io.load_dataset("dblp")

path = "/home/lj/tmp/pycharm_project_463/scripts/continue/output"
def polt_cluster(embedding, name):
    tsne = TSNE(n_components=2, init='pca', verbose=1)
    embedd = tsne.fit_transform(embedding)
    # 可视化
    plt.figure(figsize=(40, 40))
    plt.scatter(embedd[:, 0], embedd[:, 1])

    for i in range(1000):
        x = embedd[i][0]
        y = embedd[i][1]
    plt.savefig('./output/'+name +'.png', dpi=120)
    plt.show()

def save_tfidf_embedding():
    vocab = expert_finding.preprocessing.text.dictionary.Dictionary(T, min_df=20, max_df_ratio=0.25)
    tfidf_docs_vectors = expert_finding.preprocessing.text.vectorizers.get_tfidf_dictionary(vocab)
    # polt_cluster(tfidf_docs_vectors, "tfidf")
    print(tfidf_docs_vectors[1640])

def tfidf_vector():
    cv = TfidfVectorizer(binary=False, decode_error='ignore', stop_words='english')
    vec = cv.fit_transform(T)# 传入句子组成的list
    arr = vec.toarray()
    polt_cluster(arr, "tfidf")
    print(arr[0])

def save_documents_embedding():
    model_name = 'allenai/scibert_scivocab_uncased'

    # Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
    word_embedding_model = models.Transformer(model_name)

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # model= SentenceTransformer(path + model_name)
    model._first_module().max_seq_length = 500
    embedding_docs_vectors = normalize(model.encode(T), norm='l2', axis=1)
    polt_cluster(embedding_docs_vectors, 'scibert_scivocab_uncased')

def save_documents_nil_embedding():
    model_name = '/doc_doc_sci_bert_fusion_triples'
    bert_model = SentenceTransformer(path + model_name);
    bert_model._first_module().max_seq_length = 500
    embedding_docs_vectors = normalize(bert_model.encode(T), norm='l2', axis=1)
    polt_cluster(embedding_docs_vectors, 'fusion')

# save_documents_embedding()
# tfidf_vector()
save_documents_embedding()



