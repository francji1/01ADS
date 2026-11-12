# Lecture 07: Contextual and sentence embeddings, semantic search

One notebook. Two training objectives of transformers (masked vs. next-token prediction) and how a decoder generates
text (sampling by hand with temperature, top-k and top-p, then `generate()` with a chat template on Qwen3.5-0.8B);
from token vectors to sentence vectors (pooling, anisotropy of a raw language model, the STS-B test); semantic search
measured on a Quora benchmark (TF-IDF, BM25, dense retrievers, hybrid search, recall@k and MRR); contrastive
fine-tuning of a raw MiniLM live, with its effect on similarity and search and a clean vs. contaminated evaluation;
fine-tuning a small Czech encoder (Seznam RetroMAE) for sentiment of ČSFD film reviews in the three regimes of
Lecture 04; a checklist for choosing and evaluating embedding models.

**Notebook (2026/27):** [01ADS_L07_Sentence_Embeddings.ipynb](01ADS_L07_Sentence_Embeddings.ipynb)
<a href="https://colab.research.google.com/github/francji1/01ADS/blob/main/lectures/L07_sentence_embeddings/01ADS_L07_Sentence_Embeddings.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

In Colab choose *Runtime → Change runtime type → T4 GPU* and start with *Runtime → Run all*: the first run downloads
about 2.6 GB of models; with a GPU the two fine-tuning sections train live (about three minutes in total), without a
GPU the notebook downloads the results of the same code from `outputs/` and only evaluates them.

## Files

- `data/quora_train_pairs.csv`: 20 000 pairs of duplicate Quora questions for the contrastive fine-tuning, none of them in
  the retrieval benchmark of Lecture 05 (`../L05_embeddings/data/quora_retrieval.csv`, which the notebook also reads).
- `data/csfd_reviews.csv`: 5 000 Czech film reviews with star ratings from ČSFD, binarised (negative = 0–1 stars,
  positive = 4–5 stars); from the MTEB dataset `CSFDCZMovieReviewSentimentClassification` (CC-BY-SA-4.0).
- `data/stsb_test.csv`: the 1 379 test pairs of the Semantic Textual Similarity Benchmark with human scores in [0, 1].
- `outputs/`: results of the fine-tuning cells produced on a GPU (loss curves, metrics, embeddings of the benchmark,
  test probabilities), so that the notebook also runs on a CPU-only machine.

## Materials from previous years

- 2025/26: [LLM_embeddings_intro.ipynb](https://github.com/francji1/01ADS/blob/3ecb99a/lectures/L05_llm_intro/LLM_embeddings_intro.ipynb)
  (word2vec, GloVe, MiniLM sentence vectors, a small nearest-neighbour search over Quora questions).
- 2024/25: transformers in Google Slides, linked from
  [L07_transformers.md](https://github.com/francji1/01ADS/blob/3ecb99a/lectures/L05_llm_intro/2024/transformers/L07_transformers.md).
