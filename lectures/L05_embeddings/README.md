# Lecture 05: Embeddings, from images to text

The lecture continues Lecture 04: the vectors a pretrained ResNet18 produces for images, then
CLIP, which puts images and text into one space: how it was trained (the contrastive loss, computed by hand),
zero-shot classification of the Lecture 04 cats and dogs compared with the fine-tuned networks, prompts, all ten
CIFAR-10 classes, how many labelled images zero-shot is worth (linear probes on ResNet18 and CLIP vectors), searching
images with text, a typographic attack, the gap between image and text vectors, and a cell for your own photos.
Then sentence embeddings: tokens, contextual vectors, mean pooling, a raw language model vs. the same model after
contrastive fine-tuning, and the input length limit. Semantic search on Quora questions with known duplicates:
TF-IDF and BM25 vs. dense retrieval, recall@k and MRR, where each one wins, hybrid search (reciprocal rank fusion),
a similarity threshold as a classifier, speed. Finally Czech: an English model vs. a multilingual model, and Czech
queries on an English corpus, with a checklist for the retrieval part of your project.

**Notebook (2026/27):** [01ADS_L05_Embeddings.ipynb](01ADS_L05_Embeddings.ipynb)
<a href="https://colab.research.google.com/github/francji1/01ADS/blob/main/lectures/L05_embeddings/01ADS_L05_Embeddings.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

The notebook runs on a CPU (a GPU only makes encoding faster). The first run downloads the models from Hugging Face,
about 1.4 GB in total.

## Files

- `data/quora_retrieval.csv`: 1 000 queries and a corpus of 10 000 questions cut from the
  *First Quora Dataset Release: Question Pairs* ([Kaggle](https://www.kaggle.com/c/quora-question-pairs); non-commercial
  use under Quora's terms); the column `group` links each query to its duplicates in the corpus (`-1` = unrelated question).
- `data/quora_pairs.csv`: 3 000 labelled question pairs from the same dataset (duplicate or not).
- `outputs/clip_cifar.npz`: CLIP image vectors of 9 400 CIFAR-10 images, computed by the notebook on a GPU.
- The notebook also reads `outputs/embeddings_resnet18.npz`, `outputs/results.json` and two pictures from
  [Lecture 04](../L04_classification_nn_tl/).

## Materials from previous years

- 2025/26: [LLM_embeddings_intro.ipynb](https://github.com/francji1/01ADS/blob/3ecb99a/lectures/L05_llm_intro/LLM_embeddings_intro.ipynb)
  (word2vec, GloVe, contextual embeddings).
