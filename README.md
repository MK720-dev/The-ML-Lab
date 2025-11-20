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
│   ├── README.md
│   ├── CheckVersions.py
│   ├── NBClassifier.py
│   ├── iris.csv
│   ├── Rubric 1.docx
│   └── results/
│
├── Assignment2_CompareMLModels/
│   ├── README.md
│   ├── CompareMLModels.py
│   ├── iris.csv
│   ├── EECS658_Assignment2.pdf
│   ├── Rubric 2.docx
│   └── results/
│
├── Assignment3_ModelComparisonV2_DBN/
│   ├── README.md
│   ├── CompareMLModelsV2.py
│   ├── dbn.py
│   ├── dbn/
│   ├── iris.csv
│   ├── Rubric 3.docx
│   └── results/
│
├── Assignment4_PCA_SA_GA/
│   ├── README.md
│   ├── CompareFeatureSelectionMethods.py
│   ├── iris.csv
│   ├── PoV.xlsx
│   ├── Rubric 4.docx
│   └── results/
│
├── Assignment5_ImbalancedLearning/
│   ├── README.md
│   ├── ImbalancedIris.py
│   ├── imbalanced iris.csv
│   ├── Rubric 5.docx
│   └── results/
│
├── Assignment6_UnsupervisedClustering/
│   ├── README.md
│   ├── CompareClusters.py
│   ├── PlottingCode.py
│   ├── iris.csv
│   ├── Rubric 6.docx
│   └── plots/
│
└── Assignment7_Gridworld_RL/
    ├── README.md
    ├── GridWorld.py
    ├── Rubric 7.docx
    └── results/
```

# Assignment Details:

### **Assignment 1 – Environment Verification & Naive Bayes Classifier**  
**Folder:** `MalekKchaou_Assignment1`

This assignment validates the Python/ML environment and implements a complete 2-fold cross-validated **Gaussian Naive Bayes classifier** on the Iris dataset. It is the introductory programming assignment for EECS 658.

#### **Deliverables (as specified in the assignment instructions)**  
The folder contains everything required by the assignment:

- ✔ `CheckVersions.py`  
  Prints versions of:
  - Python  
  - SciPy  
  - NumPy  
  - Pandas  
  - scikit-learn  
  …and prints **“Hello World!”**
- ✔ Console screenshot or run output demonstrating successful execution  
- ✔ `NBClassifier.py`  
  Implements **manual 2-fold cross-validation** (not using sklearn’s KFold).  
- ✔ Screenshot of NBClassifier execution  
- ✔ Written calculations (accuracy, precision, recall, F1)  
  - These metrics must be computed manually from the confusion matrix and compared to the program’s output.

#### **What the Program Does**

**1. Environment verification**  
`CheckVersions.py` confirms installation of:

- Python  
- SciPy  
- NumPy  
- Pandas  
- scikit-learn  

and prints version numbers exactly as required by the assignment.

**2. Manual 2-fold cross-validation**  
The program:

- Splits the dataset into **Fold 1** (75 samples) and **Fold 2** (75 samples)  
- Train on Fold 1 → Test on Fold 2  
- Train on Fold 2 → Test on Fold 1  
- Combines predictions to produce **150 predictions** (for the full dataset)

**3. Evaluate Gaussian Naive Bayes**  
Outputs:

- Confusion matrix (must sum to **150**)  
- Overall accuracy  
- Precision for each class  
- Recall for each class  
- F1 score for each class  

**4. Manual metric verification**  
Per the instructions, these metrics must also be computed by hand using the confusion matrix and validated against program output.

#### **Skills Demonstrated**

- Setting up a Python ML environment  
- Verifying scientific packages  
- Manual k-fold cross-validation logic  
- Loading and using the Iris dataset  
- Running Gaussian Naive Bayes  
- Understanding classification metrics  
- Connecting confusion matrix values to derived metrics  

#### **How to Run**

```bash
python CheckVersions.py
python NBClassifier.py

---

### **Assignment 2 – CompareMLModels (Seven Classifiers Using 2-Fold Cross-Validation)**  
**Folder:** `MalekKchaou_Assignment2`

This assignment implements and compares seven classical machine learning models using **manual 2-fold cross-validation** on the Iris dataset. The goal is to evaluate model performance (confusion matrices + accuracy) and answer conceptual questions about why some models outperform others.

#### **Deliverables (as required by the assignment instructions)**  
The assignment folder contains:

- ✔ `CompareMLModels.py` (main program)
- ✔ `EECS658_Assignment2.pdf` (problem statement)
- ✔ `Rubric 2.docx` with name + ID filled in
- ✔ Output screenshot(s) showing program execution  
  - A screen print **is required** (the grader does *not* run your code)
- ✔ Written answers for:
  - **a.** Which model is best based on accuracy?  
  - **b.** Why each of the remaining 6 performs worse than the best model?

#### **Models Implemented (exactly as required)**
The program compares the following classifiers:

1. **Linear Regression** (`LinearRegression`)
2. **Polynomial Regression (degree 2)**  
   - Using `LinearRegression` with polynomial feature expansion
3. **Polynomial Regression (degree 3)`**
4. **Gaussian Naive Bayes** (`GaussianNB`)
5. **k-Nearest Neighbors** (`KNeighborsClassifier`)
6. **Linear Discriminant Analysis** (`LinearDiscriminantAnalysis`)
7. **Quadratic Discriminant Analysis** (`QuadraticDiscriminantAnalysis`)

#### **Manual 2-Fold Cross-Validation Procedure**
The script uses 2-fold CV:

1. Split dataset into **Fold 1** (75 samples) and **Fold 2** (75 samples)
2. Train model on Fold 1 → Test on Fold 2  
3. Train model on Fold 2 → Test on Fold 1  
4. Concatenate predictions → **150 total test predictions**

For **each** model, the program prints:

- A labeled **confusion matrix**  
  - MUST sum to **150**
- **Accuracy**

#### **Questions Answered in the Report**
Your written submission addresses:

1. **Which model has the highest accuracy?**  
   - Based on the confusion matrices + accuracy printed by the program.
2. **Why each of the six remaining models performs worse than the best one**  
   - Requires conceptual reasoning:
     - e.g., overfitting of polynomial regression, sensitivity of kNN, Gaussian assumptions, linearity limits, etc.

#### **Skills Demonstrated**

- Manual implementation of 2-fold cross-validation  
- Use of multiple classification algorithms in scikit-learn  
- Treating regression models as classifiers (via rounding or argmax of outputs)  
- Confusion matrix interpretation  
- Accuracy evaluation and comparison  
- Clear code organization and commenting (required by the rubric)

#### **How to Run**

```bash
python CompareMLModels.py

---

# Assignment 3 — CompareMLModelsV2 & DBN

## CompareMLModelsV2  
Expands the model comparison to 12 ML models including  
SVM, Decision Tree, Random Forest, ExtraTrees, and MLPClassifier  
(as required in Assignment 3 instructions).

## DBN Implementation  
Includes `dbn.py` and the DBN folder. Outputs accuracy on MNIST.

## Written answers  
All answers (train/test sizes, class listing, CV usage, etc.) are included.

---

# Assignment 4 — Feature Selection Techniques

Implements four parts based on Assignment 4 instructions:

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

Follows Assignment 5 specifications.

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

## Part 1 — K-Means Clustering
- Run K-Means for k = 1 → 20  
- Plot **reconstruction error vs k**  
- Identify **elbow_k** manually  
- Use predict() with clusters for:
  - **k = elbow_k**
  - **k = 3**
- Print confusion matrix and accuracy (only if k = 3)
- Answer **Question 1** about number of species implied by elbow_k

## Part 2 — Gaussian Mixture Models (GMM)
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

## Part 3 — Self-Organizing Map (SOM)
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

Implements the full 5×5 Gridworld per instructions.

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




