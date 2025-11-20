                                  ████████╗██╗  ██╗███████╗     ███╗   ███╗██╗       ██╗       █████╗  ██████╗
                                  ╚══██╔══╝██║  ██║██╔════╝     ████╗ ████║██║       ██║      ██╔══██╗ ██╔══██╗
                                     ██║   ███████║█████╗       ██╔████╔██║██║       ██║      ███████║ ██████╔╝
                                     ██║   ██╔══██║██╔══╝       ██║╚██╔╝██║██║       ██║      ██╔══██  ██╔══██╗
                                     ██║   ██║  ██║███████╗     ██║ ╚═╝ ██║███████╗  ███████╗ ██║  ██║ ██████╔╝
                                     ╚═╝   ╚═╝  ╚═╝╚══════╝     ╚═╝     ╚═╝╚══════╝  ╚═╝╚═══╝ ╚═╝  ╚═╝ ╚═════╝
  
<h1 align="center">🧪 The-ML-Lab</h1>
<h3 align="center">KU EECS 658 — Introduction to Machine Learning</h3>
<p align="center">A curated collection of seven ML assignments covering supervised learning, clustering, dimensionality reduction, feature selection, and reinforcement learning.</p>

<p align="center">

  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python">
  <img src="https://img.shields.io/badge/NumPy-1.26+-orange?logo=numpy">
  <img src="https://img.shields.io/badge/scikit--learn-1.3+-green?logo=scikitlearn">
  <img src="https://img.shields.io/badge/License-MIT-purple">

</p>

# Repository Structure
```
The-ML-Lab/
│
├── README.md
├── LICENSE
├── assets/
│   ├── banner.svg
│   └── logo.svg
│
├── Assignment1_CheckVersions_NBClassifier/
│   ├── CheckVersions.py
│   ├── NBClassifier.py
│   ├── iris.csv
│   ├── Rubric 1.docx
│   └── results/
│
├── Assignment2_CompareMLModels/
│   ├── CompareMLModels.py
│   ├── iris.csv
│   ├── EECS658_Assignment2.pdf
│   ├── Rubric 2.docx
│   └── results/
│
├── Assignment3_ModelComparisonV2_DBN/
│   ├── CompareMLModelsV2.py
│   ├── dbn.py
│   ├── dbn/
│   ├── iris.csv
│   ├── Rubric 3.docx
│   └── results/
│
├── Assignment4_PCA_SA_GA/
│   ├── CompareFeatureSelectionMethods.py
│   ├── iris.csv
│   ├── PoV.xlsx
│   ├── Rubric 4.docx
│   └── results/
│
├── Assignment5_ImbalancedLearning/
│   ├── ImbalancedIris.py
│   ├── imbalanced iris.csv
│   ├── Rubric 5.docx
│   └── results/
│
├── Assignment6_UnsupervisedClustering/
│   ├── CompareClusters.py
│   ├── PlottingCode.py
│   ├── iris.csv
│   ├── Rubric 6.docx
│   └── plots/
│
└── Assignment7_Gridworld_RL/
    ├── GridWorld.py
    ├── Rubric 7.docx
    └── results/
```


# Coding Assignments Details 

*A concise overview of all seven coding assignments in this repository.*

This document provides high-level summaries of each assignment, the core ideas behind the programs, the skills developed, and instructions for running the code.

---

## Assignment 1 – Environment Check & Naive Bayes Classifier

**Folder:** `MalekKchaou_Assignment1`

### Overview

Verifies the scientific Python environment and implements a Gaussian Naive Bayes classifier with a manually coded 2-fold cross-validation procedure.

### What the Code Does

* Prints versions of Python, NumPy, SciPy, Pandas, and scikit-learn.
* Loads the Iris dataset and implements manual 2-fold CV (75/75 split).
* Trains Naive Bayes and prints confusion matrix, accuracy, precision, recall, and F1-score.
* Verifies metrics manually using the confusion matrix.

### Skills Learned

Environment setup, manual CV logic, reading datasets, classifier implementation, evaluation metrics.

### How to Run

```
python CheckVersions.py
python NBClassifier.py
```

---

## Assignment 2 – Comparing Seven Models (2-Fold CV)

**Folder:** `MalekKchaou_Assignment2`

### Overview

Compares seven foundational ML models on the Iris dataset using the same manual 2-fold CV structure.

### Models Evaluated

Linear Regression, Polynomial Regression (deg 2 and 3), GaussianNB, k-NN, LDA, QDA.

### What the Code Does

* Runs manual 2-fold CV for each model.
* Prints labeled confusion matrices (values sum to 150) and the accuracy.
* Written analysis explains which model performs best and why others perform worse.

### Skills Learned

Model comparison, confusion matrix interpretation, understanding algorithm behavior.

### How to Run

```
python CompareMLModels.py
```

---

## Assignment 3 – Extended Model Comparison & Deep Belief Network

**Folder:** `MalekKchaou_Assignment3`

### Part 1 – CompareMLModelsV2 (12 Models)

Extends Assignment 2 by evaluating 12 total classifiers using manual 2-fold CV.

**What the Code Does**

* Evaluates 12 models and prints confusion matrices and accuracy.
* Written explanation compares model performance based on conceptual reasoning.

**Skills Learned**
Bias/variance understanding, linear vs. nonlinear behavior, deeper model comparison.

---

### Part 2 – Deep Belief Network (DBN) with MNIST

Trains a Deep Belief Network using RBM pretraining and supervised fine-tuning.

**What the Code Does**

* Loads MNIST.
* Performs RBM-based layerwise pretraining.
* Fine-tunes using backpropagation.
* Prints final accuracy (typically around 97%).
* Written responses analyze dataset dimensions and DBN structure.

**Skills Learned**
Unsupervised pretraining, MNIST handling, modifying external ML code, understanding DBNs.

### How to Run

```
python dbn.py
```

---

## Assignment 4 – Feature Selection (PCA, Simulated Annealing, Genetic Algorithm)

**Folder:** `MalekKchaou_Assignment4`

### Overview

Compares multiple feature-selection strategies: PCA, Simulated Annealing (SA), and Genetic Algorithm (GA).

### What the Code Does

* Baseline model using all four Iris features.
* PCA transformation and selection of components using PoV > 0.90.
* SA search on an 8-feature space (4 original + 4 PCA features).
* GA search on the same feature space using a predefined initial population.
* Prints selected feature sets, confusion matrices, and accuracy for each method.

### Skills Learned

Dimensionality reduction, PCA interpretation, search-based optimization (SA and GA), model evaluation.

### How to Run

```
python CompareFeatureSelectionMethods.py
```

---

## Assignment 5 – Imbalanced Iris (Oversampling & Undersampling)

**Folder:** `MalekKchaou_Assignment5`

### Overview

Investigates how oversampling and undersampling techniques affect classifier performance on an imbalanced version of the Iris dataset.

### Methods Used

Random Oversampling, SMOTE, ADASYN, Random Undersampling, ClusterCentroids, Tomek Links.

### What the Code Does

* Uses a Neural Network classifier with manual 2-fold CV.
* Applies each resampling technique.
* Prints labeled confusion matrices, accuracy, and balanced accuracy (manual and sklearn).

### Skills Learned

Handling imbalanced data, resampling techniques, evaluating balanced metrics.

### How to Run

```
python ImbalancedIris.py
```

---

## Assignment 6 – Unsupervised Learning (K-Means, GMM, SOM)

**Folder:** `MalekKchaou_Assignment6`

### Overview

Explores the Iris dataset using three unsupervised learning methods.

### What the Code Does

* K-Means: computes reconstruction error for k=1 to 20, selects elbow_k, produces confusion matrices.
* GMM: computes AIC/BIC curves, selects optimal cluster count, produces confusion matrices.
* SOM: trains four grid sizes (3×3, 7×7, 15×15, 25×25), produces U-matrices, species plots, and quantization error curves.

### Skills Learned

Cluster evaluation, AIC/BIC model selection, SOM interpretation, unsupervised learning workflow.

### How to Run

```
python CompareClusters.py
```

---

## Assignment 7 – Gridworld Reinforcement Learning

**Folder:** `MalekKchaou_Assignment7`

### Overview

Implements Policy Iteration and Value Iteration on a 5×5 Gridworld to compute the optimal policy and value function.

---

### Part 1 – Policy Iteration

**What the Code Does**

* Initializes a random policy and zero-valued V.
* Alternates between policy evaluation (iterative V-updates) and policy improvement (greedy w.r.t. V).
* Prints V at specific iterations (0, 1, 10, final).
* Extracts final optimal policy and plots convergence.

**Skills Learned**
Bellman expectation updates, policy improvement, convergence behavior.

---

### Part 2 – Value Iteration

**What the Code Does**

* Uses Bellman optimality updates until convergence.
* Prints V at iterations 0, 1, 2, and final.
* Extracts optimal policy.
* Plots the value-change curve and compares convergence speed with Policy Iteration.

**Skills Learned**
Optimality backups, dynamic programming, optimal control, convergence diagnostics.

### How to Run

```
python GridWorld.py
```


