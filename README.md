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

# Assignment 1 — CheckVersions & NBClassifier

This assignment implements:

### ✔ Part 1: CheckVersions  
Prints the versions of Python, SciPy, NumPy, Pandas, and scikit-learn  
(as required in Assignment 1 instructions).

### ✔ Part 2: NBClassifier  
Implements 2-fold cross-validation on the iris dataset using GaussianNB and prints:

- Accuracy  
- Confusion matrix  
- Precision, Recall, F1 for each iris class  

All calculations match the rubric requirements.

---

# Assignment 2 — CompareMLModels

Implements 2-fold CV across seven models (LinearReg, Poly2, Poly3, NB, kNN, LDA, QDA) per Assignment 2 specifications :contentReference[oaicite:1]{index=1}.

For each model, the program prints:

- Confusion matrix  
- Accuracy  
- Label identifying the model currently being evaluated  

All confusion matrices sum to 150 samples as required.

---

# Assignment 3 — CompareMLModelsV2 & DBN

### ✔ CompareMLModelsV2  
Expands the model comparison to 12 ML models including  
SVM, Decision Tree, Random Forest, ExtraTrees, and MLPClassifier  
(as required in Assignment 3 instructions :contentReference[oaicite:2]{index=2}).

### ✔ DBN Implementation  
Includes `dbn.py` and the DBN folder. Outputs accuracy on MNIST.

### ✔ Written answers  
All answers (train/test sizes, class listing, CV usage, etc.) are included.

---

# Assignment 4 — Feature Selection Techniques

Implements four parts based on Assignment 4 instructions :contentReference[oaicite:3]{index=3}:

## Part 1  
Baseline Decision Tree using original 4 iris features.

## Part 2 — PCA  
- Compute eigenvalues/eigenvectors  
- Compute PoV and verify > 0.90  
- Select transformed features for classification

## Part 3 — Simulated Annealing  
Runs 100 iterations with 1–2 random perturbations.

## Part 4 — Genetic Algorithm  
Runs 50 generations on initial populations defined in the instructions.

Includes all required outputs and PoV spreadsheet verification.

---

# Assignment 5 — Imbalanced Iris Dataset

Follows Assignment 5 specifications :contentReference[oaicite:4]{index=4}.

## Part 1  
Compute confusion matrix, accuracy, class-balanced accuracy, and sklearn balanced accuracy.

## Part 2 — Oversampling  
Random Oversampling, SMOTE, ADASYN.

## Part 3 — Undersampling  
Random undersampling, ClusterCentroids, Tomek Links.

Each section prints labeled confusion matrices and accuracy.

---

# Assignment 6 — Unsupervised Machine Learning (K-Means, GMM, SOM)

This assignment follows the official EECS 658 instructions exactly.  
It investigates unsupervised clustering approaches applied to the Iris dataset.

---

## 📌 Part 1 — K-Means Clustering
- Run K-Means for k = 1 → 20  
- Plot **reconstruction error vs k**  
- Identify **elbow_k** manually  
- Use predict() with clusters for:
  - **k = elbow_k**
  - **k = 3**
- Print confusion matrix and accuracy (only if k = 3)
- Answer **Question 1** about number of species implied by elbow_k

## 📌 Part 2 — Gaussian Mixture Models (GMM)
- Run GMM for k = 1 → 20  
- Plot **AIC vs k** → pick **aic_elbow_k**  
- Plot **BIC vs k** → pick **bic_elbow_k**  
- Use predict() to classify data for:
  - **k = aic_elbow_k**
  - **k = bic_elbow_k**
- Print confusion matrix and accuracy (only if k = 3)
- Answer:
  - **Question 2a** — AIC interpretation  
  - **Question 2b** — BIC interpretation

## 📌 Part 3 — Self-Organizing Map (SOM)
- Normalize features to the range [0,1]
- Train MiniSom maps of sizes:
  - 3×3  
  - 7×7  
  - 15×15  
  - 25×25  
- Plot **U-Matrices**
- Print **Quantization Error** for each
- Plot **Q.E. vs Grid Size**
- Answer:
  - **Question 3a** — elbow grid size  
  - **Question 3b** — effect of grid size on performance  
  - **Question 3c** — best fit between 7×7 and 25×25

---

# Assignment 7 — Gridworld RL (Policy Iteration & Value Iteration)

Implements the full 5×5 Gridworld per instructions :contentReference[oaicite:5]{index=5}.

## Part 1 — Policy Iteration  
- Policy evaluation  
- Policy improvement  
- Print V at iterations 0, 1, 10, and final  
- Convergence plot |Vᵏ − Vᵏ⁻¹|

## Part 2 — Value Iteration  
- Bellman optimality updates  
- Print V at iterations 0, 1, 2, and final  
- Extract optimal policy  
- Convergence plot

Includes written answers for Questions 1–3.




