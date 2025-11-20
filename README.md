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
```
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
```

---

# Assignment 3 — CompareMLModelsV2 & DBN

### **Assignment 3 – CompareMLModelsV2 & Deep Belief Network (DBN)**  
**Folder:** `MalekKchaou_Assignment3`

This assignment expands the classifier comparison from Assignment 2 and adds a full Deep Belief Network (DBN) implementation trained on the MNIST handwritten digits dataset. It is divided into **two major parts**, each with its own code, outputs, and written answers.  
:contentReference[oaicite:1]{index=1}

---

## **📌 Part 1 — CompareMLModelsV2 (12 Classifiers, 2-Fold Cross-Validation)**

You upgrade the previous CompareMLModels program to **CompareMLModelsV2**, adding five additional models and evaluating a total of **12 machine learning classifiers** using **manual 2-fold cross-validation** on the 150-sample Iris dataset.

### ✔ **Models Required (All 12)**  
As specified in the assignment:

1. **Gaussian Naive Bayes** (`GaussianNB`)
2. **Linear Regression**
3. **Polynomial Regression (degree 2)**
4. **Polynomial Regression (degree 3)**
5. **k-Nearest Neighbors** (`KNeighborsClassifier`)
6. **Linear Discriminant Analysis (LDA)**
7. **Quadratic Discriminant Analysis (QDA)**
8. **Support Vector Machine (Linear SVC)** (`svm.LinearSVC`)
9. **Decision Tree** (`DecisionTreeClassifier`)
10. **Random Forest** (`RandomForestClassifier`)
11. **Extra Trees** (`ExtraTreesClassifier`)
12. **Neural Network (MLPClassifier)**

### ✔ **Manual 2-Fold Cross-Validation (Strict Requirement)**  
The scripts use 2-Fold Cross-Validation:

- Split dataset into **Fold 1 (75 samples)** and **Fold 2 (75 samples)**  
- Train on Fold 1 → Test on Fold 2  
- Train on Fold 2 → Test on Fold 1  
- Combine predictions → **150 total predictions**

### ✔ **Program Output**  
For **each of the 12 models**, I printed:

- A **clearly labeled confusion matrix**
- **Accuracy**
- Matrices must sum to **150**  
  > If they do not, the assignment is considered incorrect.

### ✔ **Written Questions Required**
The PDF answers include answers to:

1. **Which model is the best based on accuracy?**
2. **For each of the 11 remaining models**, explain in detail:
   - **Why the model performs worse than the best one**
   - Explanations must be conceptual, not code-based  
     (e.g., overfitting, linear assumptions, variance, instability, etc.)

---

## **📌 Part 2 — Deep Belief Network (DBN)**

This part uses the **MNIST digit recognition** dataset and implements a Deep Belief Network based on the public GitHub repository:

🔗 https://github.com/albertbup/deep-belief-network

### ✔ **Required Setup Steps**
Following assignment instructions strictly:

1. Download the Deep Belief Network ZIP from GitHub.  
2. Copy the **`dbn/` folder** into the same directory as your Python file.  
3. Create a new file named **`dbn.py`** in that same directory.  
4. Copy the code from the repository’s *Overview* section (from `import numpy as np` down to the final accuracy print).  
5. Modify the import:  
   - Use:  
     ```python
     from dbn import SupervisedDBNClassification
     ```  
   - And comment out:  
     ```python
     from dbn.tensorflow import SupervisedDBNClassification
     ```
6. Replace the deprecated import:  
   ```python
   from sklearn.metrics.classification import accuracy_score


---

### **Assignment 4 – Feature Selection (PCA, Simulated Annealing, Genetic Algorithm)**  
**Folder:** `MalekKchaou_Assignment4`

This assignment evaluates multiple dimensionality-reduction and feature-selection strategies on the Iris dataset using **2-fold cross-validation with a Decision Tree classifier**.  
It contains **four major parts**: baseline, PCA, simulated annealing, and the genetic algorithm.  
:contentReference[oaicite:1]{index=1}

---

## **📌 Deliverables (as required in the instructions)**

- ✔ `CompareFeatureSelectionMethods.py` (main program)  
- ✔ `Rubric 4.docx` (with name + ID filled in — *must not be submitted as PDF*)  
- ✔ A **screen print** of the console output  
  - Required — the grader does **not** run your code  
- ✔ **PoV verification spreadsheet** (Part 2) in `.xlsx` format  
- ✔ Written PDF answering Part 4 conceptual questions  
- ✔ Screenshots for all parts (confusion matrices, accuracies, U-matrices, plots)

---

## **📌 Program Requirements (All Four Parts)**

For **each part**, the program prints:

- **Labeled Confusion Matrix**  
- **Accuracy**  
- **List of features used to generate that confusion matrix**

This applies to **Parts 1, 2, 3, and 4**.

---

# **PART 1 — Baseline (No Dimensionality Reduction)**

- Use the original four features:  
  `sepal-length, sepal-width, petal-length, petal-width`  
- Perform **2-fold cross-validation** with a Decision Tree  
- Print:  
  - Confusion matrix  
  - Accuracy  
  - List of used features (the 4 original ones)

---

# **PART 2 — PCA Feature Transformation**

This part follows the “PCA Feature Transformation” lecture slides.

### ✔ Steps Required:

1. Compute PCA on the original 4 features → produce transformed features:  
   **z1, z2, z3, z4**
2. Display:
   - **Eigenvalues matrix**
   - **Eigenvectors matrix**  
3. Compute **PoV (Proportion of Variance)** using the eigenvalues  
4. Select the smallest subset of transformed features such that:  
   **PoV > 0.90**
5. Perform 2-fold CV using only the selected components  
6. Print:
   - Selected PCA components  
   - PoV  
   - Confusion matrix  
   - Accuracy  

### ✔ Additional Deliverable  
- **PoV verification** using the eigenvalues in an `.xlsx` spreadsheet  
  (per instructions, “Deliverable 4 (PoV) Example” on Canvas)

---

# **PART 3 — Simulated Annealing over 8 Features**

Feature space =  
4 original features + 4 PCA features from Part 2 → **8 total features**.

### ✔ Required Settings:

- **iterations = 100**  
- **Perturb with 1 or 2 randomly selected parameters**  
  - Because 1–5% of 8 features < 1  
- Acceptance probability:  
  \[
  \Pr[\text{accept}] = e^{-\Delta / c},\quad c = 1
  \]
- Restart after **10** consecutive discarded iterations

### ✔ For *each iteration*, program MUST print:

- Current subset of features  
- Accuracy  
- Acceptance probability  
- Random uniform number  
- Status: **Improved**, **Accepted**, **Discarded**, or **Restart**

### ✔ End of Part 3:
- Best feature set discovered  
- Final confusion matrix  
- Final accuracy  
- List of chosen features

---

# **PART 4 — Genetic Algorithm (GA)**

GA searches the **same 8-feature space** as Part 3.

### ✔ Required Initial Population (exactly as specified):

1. `{z1, sepal-length, sepal-width, petal-length, petal-width}`  
2. `{z1, z2, sepal-width, petal-length, petal-width}`  
3. `{z1, z2, z3, sepal-width, petal-length}`  
4. `{z1, z2, z3, z4, sepal-width}`  
5. `{z1, z2, z3, z4, sepal-length}`  

### ✔ Required GA Parameters:
- **50 generations**  
- At the end of **each generation**, print the **five best** feature sets with:
  - Their feature lists  
  - Their accuracies  
  - The generation number  

### ✔ End of Part 4:
- Best feature set found by the GA  
- Final confusion matrix  
- Final accuracy  

---

# **Required Written Questions (PDF)**  
Assignment Folder includes answers to following questions (a)–(f):

a. Which dimensionality reduction method (PCA, SA, GA) performed best?  
b. Why did the other two perform worse?  
c. Did the best method outperform using all 4 original features? Why/why not?  
d. Did PCA and simulated annealing select the same features? Explain.  
e. Did PCA and GA select the same features? Explain.  
f. Did SA and GA select the same features? Why/why not?


## **How to Run**

```bash
python CompareFeatureSelectionMethods.py
```

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




