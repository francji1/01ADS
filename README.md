# Application of Data Science (01ADS)

Materials for the course **01ADS - Application of Data Science** at FJFI CTU in Prague, winter semester 2026/27.
Lecturer: Jiri Franc.

The course combines lectures on applied machine learning, deep learning and large language models
with a semester-long team project.

## Repository structure

- **`lectures/`**
  One folder per lecture, named `LNN_topic/`, added as the semester progresses. Each folder contains
  - `README.md` with the links to all materials of the lecture,
  - the slides as `01ADS_LNN.pdf` (the original lives in Google Slides and is linked from the README) and/or
  - a Jupyter notebook `01ADS_LNN_Topic.ipynb` with an "Open in Colab" badge at its top,
  - optionally `data/`, `pics/` and helper scripts used by the notebook,
  - optionally `supplementary/` with additional material from previous years.

## Lectures 2026/27

| # | Topic | Materials |
|---|---|---|
| 1 | Introduction to 01ADS | [L01_intro](lectures/L01_intro/) |
| 2 | Git & Python environments (uv) | [L02_git_venv](lectures/L02_git_venv/) |
| 3 | Regression on tabular data: workflow, validation, decision trees and ensembles | [L03_decision_trees](lectures/L03_decision_trees/) |

## How to run the notebooks

- Open a notebook in Google Colab via the badge at its top (recommended), or
- clone the repository and run it locally:

  ```bash
  git clone https://github.com/francji1/01ADS.git
  ```

  with Python 3.11+ and the packages used by the given notebook (numpy, pandas, scikit-learn,
  matplotlib, seaborn and the libraries of the lecture, e.g. xgboost, lightgbm, catboost, torch).

## Course overview

The course covers key topics of applied data science:

1. Data preprocessing and cleaning, exploratory data analysis
2. Project management and working in teams
3. Machine learning basics and advanced techniques
4. Real-world applications of data science

### Learning outcomes

By the end of this course, students will:

- gain a strong understanding of data science workflows and methodologies,
- develop skills in using Python,
- be able to apply data science techniques to solve real-world problems and work in a team.

### Prerequisites

- Basic programming knowledge (Python).
- Familiarity with mathematical and statistical concepts.

Materials from previous years are available in the git history
(last state before the 2026 cleanup: [`3ecb99a`](https://github.com/francji1/01ADS/tree/3ecb99a)).
