"""
This script trains sentence transformers with a triplet loss function.
As corpus, we use the wikipedia sections dataset that was describd by Dor et al., 2018, Learning Thematic Similarity Metric Using Triplet Networks.
"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
from sentence_transformers.evaluation import TripletEvaluator
from zipfile import ZipFile
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, losses, models, util
from sentence_transformers.readers import STSBenchmarkDataReader, InputExample
import expert_finding.io

import expert_finding.models.bert_propagation_model
import csv
import logging
import os

A_da, A_dd, T, L_d, L_d_mask, L_a, L_a_mask, tags = expert_finding.io.load_dataset("dblp")

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

# You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = 'sci_bert_triples'

train_batch_size = 16
num_epochs = 4
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

logging.info("Read Triplet train dataset")
train_samples = []
with open("./doc_triples.csv") as fIn:
    reader = csv.reader(fIn)
    for i, row in enumerate(reader):
        train_samples.append(
            InputExample(texts=[str(T[int(row[1])]), str(T[int(row[2])]), str(T[int(row[3])])], label=0))

train_dataset = SentencesDataset(train_samples, model = model)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
train_loss = losses.TripletLoss(model=model)


evaluator = TripletEvaluator.from_input_examples(train_samples, name='dev')

warmup_steps = int(len(train_dataset) * num_epochs / train_batch_size * 0.1)  # 10% of train data

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path)

##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

print(model_save_path)