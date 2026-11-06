# Lecture 06: From words to vectors, word2vec and an introduction to transformers

We train word embeddings ourselves. One-hot vectors and why an embedding layer is a lookup table, word counts in
context, then word2vec (skip-gram with negative sampling) written in PyTorch and trained on Wikipedia text (text8),
watching the word vectors form groups during training. What the vectors learned: neighbours, analogies, the Google
analogy benchmark against pretrained GloVe, and where it fails. Using the vectors: sentence vectors by averaging
(Quora pairs from Lecture 05), saving and loading, item2vec for recommendations, and the limits of one vector per
word form (Czech word forms, *bank*).

The second half introduces transformers as a family of models: subword tokenisation (BPE by hand, real tokenizers on
English and Czech), self-attention computed by hand, position encoding, encoder vs. decoder vs. encoder-decoder
(attention masks, training objectives, a small model of each kind on the same sentences), attention inside
DistilBERT, a history of the best-known transformers since 2017 with their published sizes, how the best (mostly
closed) models progressed on programming, mathematics and hard-task benchmarks compared with humans, and the state
of the art by task (November 2026) with pointers to the leaderboards.

**Notebook (2026/27):** [01ADS_L06_Word2vec_Transformers.ipynb](01ADS_L06_Word2vec_Transformers.ipynb)
<a href="https://colab.research.google.com/github/francji1/01ADS/blob/main/lectures/L06_word2vec_transformers/01ADS_L06_Word2vec_Transformers.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

The notebook runs on a CPU. The first run downloads about 1 GB (text8, GloVe, three small transformers and four
tokenizers).

## Files

- `outputs/w2v_text8.npz`: the word2vec network of the notebook trained on the whole text8 corpus (30 000 words x
  100 dimensions, 3 epochs), so that the CPU run does not have to train it.
- The notebook also reads `data/quora_pairs.csv` from [Lecture 05](../L05_embeddings/).

## Materials from previous years

- 2025/26: [LLM_embeddings_intro.ipynb](https://github.com/francji1/01ADS/blob/3ecb99a/lectures/L05_llm_intro/LLM_embeddings_intro.ipynb)
  (word2vec on Quora, GloVe, contextual embeddings).
- 2024/25: [word2vec notebook](https://github.com/francji1/01ADS/tree/3ecb99a/lectures/L05_llm_intro/2024/word2vec)
  (Viktor Sohajek, gensim on Quora) and transformer slides linked from
  [L07_transformers.md](https://github.com/francji1/01ADS/blob/3ecb99a/lectures/L05_llm_intro/2024/transformers/L07_transformers.md).
