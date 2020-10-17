"""
This examples trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) for the STSbenchmark from scratch. It generates sentence embeddings
that can be compared using cosine-similarity to measure the similarity.
Usage:
python training_nli.py
OR
python training_nli.py pretrained_transformer_model_name
"""


import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import STSBenchmarkDataReader, InputExample
import csv
import expert_finding.io
import expert_finding.models.bert_propagation_model

import logging

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

A_da, A_dd, T, L_d, L_d_mask, L_a, L_a_mask, tags = expert_finding.io.load_dataset("dblp")
# Check if dataset exsist. If not, download and extract  it
# You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = 'sci_bert'
#Read the dataset
train_batch_size = 16
num_epochs = 4
#model_save_path = 'output/training_dblp' + model_name.replace("/", "-") + '-' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_save_path = 'output/doc_doc' + model_name.replace("/", "-")


model_name = 'allenai/scibert_scivocab_uncased'



# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
word_embedding_model = models.Transformer(model_name)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])





train_samples = []
dev_samples = []
test_samples = []
with open("./doc_doc_all.csv") as fIn:
    reader = csv.reader(fIn)#quoting=csv.QUOTE_NONE)
    for i, row in enumerate(reader):
        # print(type(row))
        # print(row)
        score = float(row[3]) / 1.0  # Normalize score to range 0 ... 1
        # print(score)
        inp_example = InputExample(texts=[str(T[int(row[1])]), str(T[int(row[2])])], label=score)
        train_samples.append(inp_example)
        # if i % 4 == 0:
        #     inp_example = InputExample(texts=[str(T[int(row[1])]), str(T[int(row[2])])], label=score)
        #     dev_samples.append(inp_example)
        # else:

train_dataset = SentencesDataset(train_samples, model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
train_loss = losses.CosineSimilarityLoss(model=model)

logging.info("Read dev dataset")
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(train_samples, name='dev')

# Configure the training. We skip evaluation in this example
warmup_steps = math.ceil(len(train_dataset) * num_epochs / train_batch_size * 0.1)  # 10% of train data for warm-up
# logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path)

##############################################################################
#
# Load the stored model and evaluate its performance on dataset
#
##############################################################################
print(model_save_path)
# model = SentenceTransformer(model_save_path)
# test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
# test_evaluator(model, output_path=model_save_path)
