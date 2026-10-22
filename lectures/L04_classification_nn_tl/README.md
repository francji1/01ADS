# Lecture 04: Classification and transfer learning

Two notebooks. The first covers classification on tabular data: the confusion matrix and its vocabulary, thresholds
and misclassification costs (with sliders), ROC (with an animated construction of the curve) and precision–recall
curves, calibration, a comparison of logistic regression, decision tree, random forest, XGBoost and a PyTorch MLP on
the IBM HR attrition data, coefficients and SHAP values, plus a multiclass example on handwritten digits. The second
moves to images: what a convolution computes (with an animated sliding window), a convolutional network from scratch
on CIFAR-10 cats vs. dogs, what a pretrained ResNet18 already knows, transfer learning (feature extraction, partial
and full fine-tuning) and how much data it saves, embeddings with nearest neighbours and a t-SNE map of the images,
feature maps, Grad-CAM, occlusion maps and a cell for your own photos.

**Notebook 1 (2026/27):** [01ADS_L04_Classification.ipynb](01ADS_L04_Classification.ipynb)
<a href="https://colab.research.google.com/github/francji1/01ADS/blob/main/lectures/L04_classification_nn_tl/01ADS_L04_Classification.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

**Notebook 2 (2026/27):** [01ADS_L04_Transfer_Learning.ipynb](01ADS_L04_Transfer_Learning.ipynb)
<a href="https://colab.research.google.com/github/francji1/01ADS/blob/main/lectures/L04_classification_nn_tl/01ADS_L04_Transfer_Learning.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
In Colab choose *Runtime → Change runtime type → T4 GPU* to train the networks live (about 6–8 minutes). Without a GPU the
notebook downloads the checkpoints from this folder and only evaluates them.

## Files

- `data/attrition.csv`: IBM HR employee attrition data (1 470 employees), used by notebook 1.
- `checkpoints/`, `outputs/`: weights, training histories and embeddings produced by notebook 2 on a GPU, so that the
  notebook also runs on a CPU-only machine.
- `pics/`: figures used in the notebooks.

## Supplementary materials

- [supplementary/keras_tf_2025/](supplementary/keras_tf_2025/): last year's versions of the same topics in Keras / TensorFlow,
  [intro_nn_classification_Keras.ipynb](supplementary/keras_tf_2025/intro_nn_classification_Keras.ipynb) and
  [intro_tranfer_learning_TF.ipynb](supplementary/keras_tf_2025/intro_tranfer_learning_TF.ipynb).
