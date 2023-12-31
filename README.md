# mwcat

Project to create two models using Wikipedia (English).

- a text classifier, based on Wikipedia top 40 categories.
- a text summarizer, based on Wikipedia summaries.

Wikipedia has 40 top categories that cover all popular topics.
It's a much better taxonomy than the classical AG News or Reuters one.

See https://en.wikipedia.org/wiki/Category:Main_topic_classifications

The project comes with a few scripts:

- `mwcat-create-dataset`: extract pages from the Wikipedia and store them in a dataset
- `mwcat-train`: use the dataset to train the classifier or summarizer.
- `mwcat-evaluate`: evaluate the produced models using the model eval mode.
- `mwcat-validate`: validate the produced models against the test data.

## Training Dataset

The training (90% split) & test (10% split) dataset is composed of wikipedia pages.

Each page is composed of : id, title, summary and text.

Pages are selected directly under every root categories or their direct subcategories, ensuring a
wide coverage of topics and an even distribution of pages.
The tree of categories is visited until each root category has a corpus of 2000 pages.

To generate the dataset, run `mwcat-create-dataset`.

The script generates and uploads the dataset to the Hugging Face Hub at https://huggingface.co/datasets/tarekziade/wikipedia-topics

To use it:

```
from datasets import load_dataset

dataset = load_dataset("tarekziade/wikipedia-topics")
```

## Classification

The classification model can be downloaded from https://huggingface.co/tarekziade/wikipedia-topics-distilbert

XXX

### Training

Training is done by fine tuning DistilBERT.

XXX provide results from the script

### Evaluation & Validation

XXX provide results from the script

### Usage

TBD

XXX write an example using pytorch and transformers.js

## Summarization

### Training

TDB

### Evaluation & Validation

TBD

### Usage

TBD
