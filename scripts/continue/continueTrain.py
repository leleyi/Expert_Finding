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
import logging
from datetime import datetime
import sys
import os
import gzip
import csv
import pandas as pd

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout


# Check if dataset exsist. If not, download and extract  it
# You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = 'bert-base-uncased'
# Read the dataset
train_batch_size = 8
num_epochs = 4
#model_save_path = 'output/training_dblp' + model_name.replace("/", "-") + '-' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_save_path = 'output/training_dblp1' + model_name.replace("/", "-")

# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
model = SentenceTransformer(model_name)

train_samples = []
dev_samples = []
test_samples = []
with open("./continue_train.csv") as fIn:
    reader = csv.reader(fIn)#quoting=csv.QUOTE_NONE)
    for row in reader:
        # print(type(row))
        print(row)
        score = float(row[3]) / 1.0  # Normalize score to range 0 ... 1
        print(score)
        inp_example = InputExample(texts=[str(row[1]), str(row[2])], label=score)
        train_samples.append(inp_example)

train_dataset = SentencesDataset(train_samples, model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
train_loss = losses.CosineSimilarityLoss(model=model)

logging.info("Read dev dataset")
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(train_samples, name='sts-dev')

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
