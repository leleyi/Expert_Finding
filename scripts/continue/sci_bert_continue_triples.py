import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

from sentence_transformers import (
    SentenceTransformer,
    SentenceLabelDataset,
    LoggingHandler,
    losses,
)
from torch.utils.data import DataLoader
from sentence_transformers.readers import InputExample
from sentence_transformers.evaluation import TripletEvaluator
from datetime import datetime

import logging
import os
import urllib.request
import random
from collections import defaultdict


import tarfile
def un_tar(file_name):
    #untar zip file
    tar = tarfile.open(file_name)
    names = tar.getnames()

    if os.path.isdir(file_name + "_files"):
        pass
    else:
        os.mkdir(file_name + "_files")
    #由于解压后是许多文件，预先建立同名文件夹
    for name in names:
        tar.extract(name, file_name + "_files/")
    tar.close()
un_gz("test.tar.gz")


train_samples = []
dev_samples = []
test_samples = []
with tarfile.open("\a", 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=score)

        if row['split'] == 'dev':
            dev_samples.append(inp_example)
        elif row['split'] == 'test':
            test_samples.append(inp_example)
        else:
            train_samples.append(inp_example)

# Inspired from torchnlp
def trec_dataset(
        directory="datasets/trec/",
        train_filename="train_5500.label",
        test_filename="TREC_10.label",
        validation_dataset_nb=500,
        urls=[
            "http://cogcomp.org/Data/QA/QC/train_5500.label",
            "http://cogcomp.org/Data/QA/QC/TREC_10.label",
        ],
):
    os.makedirs(directory, exist_ok=True)

    ret = []
    for url, filename in zip(urls, [train_filename, test_filename]):
        full_path = os.path.join(directory, filename)
        urllib.request.urlretrieve(url, filename=full_path)

        examples = []
        label_map = {}
        guid = 1
        for line in open(full_path, "rb"):
            # there is one non-ASCII byte: sisterBADBYTEcity; replaced with space
            label, _, text = line.replace(b"\xf0", b" ").strip().decode().partition(" ")

            # We extract the upper category (e.g. DESC from DESC:def)
            label, _, _ = label.partition(":")

            if label not in label_map:
                label_map[label] = len(label_map)

            label_id = label_map[label]
            guid += 1
            examples.append(InputExample(guid=guid, texts=[text], label=label_id))
        ret.append(examples)

    train_set, test_set = ret
    dev_set = None

    # Create a dev set from train set
    if validation_dataset_nb > 0:
        dev_set = train_set[-validation_dataset_nb:]
        train_set = train_set[:-validation_dataset_nb]

    # For dev & test set, we return triplets (anchor, positive, negative)
    random.seed(42)  # Fix seed, so that we always get the same triplets
    dev_triplets = triplets_from_labeled_dataset(dev_set)
    test_triplets = triplets_from_labeled_dataset(test_set)

    return train_set, dev_triplets, test_triplets


def triplets_from_labeled_dataset(input_examples):
    # Create triplets for a [(label, sentence), (label, sentence)...] dataset
    # by using each example as an anchor and selecting randomly a
    # positive instance with the same label and a negative instance with a different label
    triplets = []
    label2sentence = defaultdict(list)
    for inp_example in input_examples:
        label2sentence[inp_example.label].append(inp_example)

    for inp_example in input_examples:
        anchor = inp_example

        if len(label2sentence[inp_example.label]) < 2:  # We need at least 2 examples per label to create a triplet
            continue

        positive = None
        while positive is None or positive.guid == anchor.guid:
            positive = random.choice(label2sentence[inp_example.label])

        negative = None
        while negative is None or negative.label == anchor.label:
            negative = random.choice(input_examples)

        triplets.append(InputExample(texts=[anchor.texts[0], positive.texts[0], negative.texts[0]]))

    return triplets


logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

# You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = 'distilbert-base-nli-stsb-mean-tokens'

### Create a torch.DataLoader that passes training batch instances to our model
train_batch_size = 32
output_path = (
        "output/finetune-batch-hard-trec-"
        + model_name
        + "-"
        + "BatchHardTripletLoss"
)
num_epochs = 1

logging.info("Loading TREC dataset")
train_set, dev_set, test_set = trec_dataset()

# Load pretrained model
model = SentenceTransformer("/home/lj/tmp/pycharm_project_463/tests/output/training_stsbenchmark_continue_training-sci_bert_nil-2020-09-22_08-50-29")

logging.info("Read TREC train dataset")
train_dataset = SentenceLabelDataset(
    examples=train_set,
    model=model,
    provide_positive=False,  # For BatchHardTripletLoss, we must set provide_positive and provide_negative to False
    provide_negative=False,
)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)

### Triplet losses ####################
### There are 4 triplet loss variants:
### - BatchHardTripletLoss
### - BatchHardSoftMarginTripletLoss
### - BatchSemiHardTripletLoss
### - BatchAllTripletLoss
#######################################

train_loss = losses.BatchAllTripletLoss(model=model)
# train_loss = losses.BatchHardTripletLoss(model=model)
# train_loss = losses.BatchHardSoftMarginTripletLoss(sentence_embedder=model)
# train_loss = losses.BatchSemiHardTripletLoss(sentence_embedder=model)


logging.info("Read TREC val dataset")
dev_evaluator = TripletEvaluator.from_input_examples(dev_set, name='dev')

logging.info("Performance before fine-tuning:")
dev_evaluator(model)

warmup_steps = int(
    len(train_dataset) * num_epochs / train_batch_size * 0.1
)  # 10% of train data

# Train the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=dev_evaluator,
    epochs=num_epochs,
    evaluation_steps=1000,
    warmup_steps=warmup_steps,
    output_path=output_path,
)

##############################################################################
#
# Load the stored model and evaluate its performance on TREC dataset
#
##############################################################################

logging.info("Evaluating model on test set")
test_evaluator = TripletEvaluator.from_input_examples(test_set, name='test')
model.evaluate(test_evaluator)

