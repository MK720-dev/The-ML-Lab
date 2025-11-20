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

## **PART 2 — Deep Belief Network (DBN) on the MNIST Dataset**

Part 2 of Assignment 3 introduces a **Deep Belief Network (DBN)** for handwritten digit classification using the MNIST dataset.  
The goal is to learn:
- How an unsupervised layer-wise pretraining pipeline works  
- How DBNs differ from traditional feed-forward neural networks  
- How to interpret MNIST dataset shapes (samples, features, classes)  
- How to debug and run real deep learning code

All instructions in this part must be followed **exactly as specified**.  
:contentReference[oaicite:1]{index=1}

---

# 📌 **Required Setup (Must Follow Exactly)**

1. Download the GitHub ZIP file containing the DBN library:  
   👉 https://github.com/albertbup/deep-belief-network  

2. Extract the ZIP and **copy the entire `dbn/` folder** into your assignment directory  
   (the same directory containing your `dbn.py` file).

3. Create a new file called **`dbn.py`**  
   (This file will run the MNIST experiment.)

4. From the GitHub page, copy the complete code shown in the “Overview” section  
   — beginning at:  
   ```python
   import numpy as np

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

### **Assignment 5 – Imbalanced Iris Dataset (Oversampling & Undersampling Techniques)**  
**Folder:** `MalekKchaou_Assignment5`

This assignment focuses on handling **imbalanced datasets** using oversampling and undersampling techniques from the *imbalanced-learn* package.  
All experiments must use **2-fold cross-validation** with a **Neural Network classifier**.  
:contentReference[oaicite:1]{index=1}

---

## **📌 Deliverables (as required in the instructions)**

The assignment folder contains:

- ✔ `ImbalancedIris.py` (main program)  
- ✔ `imbalanced iris.csv` (provided dataset)  
- ✔ `Rubric 5.docx` (with name + ID, must NOT be PDF)  
- ✔ A **screen print** showing successful program execution  
  - Required — the grader does **not** run your code  
- ✔ Result images for Part 1, 2, and 3

---

## **📌 Global Requirements**

- Use **Neural Network classifier** (MLPClassifier)  
- Use **manual 2-fold cross-validation**  
- Before printing each part’s results, print the part label:  
  - `"Part 1:"`  
  - `"Part 2: Random Oversampling"` etc.  
- All confusion matrices must sum to the size of the dataset used  
- All confusion matrices must be **clearly labeled**

---

# **PART 1 — Baseline on Imbalanced Dataset**

Using the provided **`imbalanced iris.csv`**:

### ✔ Required Outputs:

1. **Confusion Matrix** (labeled)  
2. **Accuracy**  
3. **Class Balanced Accuracy**  
   - As defined in the *Imbalanced Datasets* lecture  
4. **Balanced Accuracy (manual)**  
   - Average of `Recall_i` for all classes  
5. **Balanced Accuracy (sklearn)**  
   ```python
   balanced_accuracy_score(y_true, y_pred)
   ```

---

### **Assignment 6 – Unsupervised Learning (K-Means, GMM, SOM)**  
**Folder:** `MalekKchaou_Assignment6`

This assignment applies three unsupervised machine learning methods to the Iris dataset:  
**K-means**, **Gaussian Mixture Models**, and **Self-Organizing Maps (SOM)**.  
You will determine how well each method clusters the data and whether the results support the hypothesis that there are **three species of iris** in Fisher’s dataset.  
:contentReference[oaicite:1]{index=1}

All clustering uses the **entire Iris dataset** (no train/test split), because this is **unsupervised learning**.

---

## 📌 Deliverables (Required in the ZIP File)

- ✔ `Rubric 6.docx` with your name and ID (NOT a PDF)  
- ✔ Python source code  
- ✔ A **screen print** (Word/PDF) showing the outputs listed below  
- ✔ All plots for Parts 1, 2, and 3  
- ✔ Written answers to:
  - Part 1 Question 1  
  - Part 2 Questions 2a and 2b  
  - Part 3 Questions 3a, 3b, 3c  

---

# **PART 1 — K-Means Clustering (k = 1 to 20)**

### ✔ Required Program Outputs

1. **Plot of reconstruction error vs k** (k = 1 → 20)  
2. **Find elbow_k manually** (the “knee” of the curve)  
3. **Use predict() with k = elbow_k**  
   - Print **confusion matrix**  
   - Print **accuracy** *only if elbow_k = 3*  
     - (If elbow_k ≠ 3 → accuracy cannot be computed; print required message.)  
4. **Use predict() with k = 3**  
   - Print labeled **confusion matrix**  
   - Print **accuracy**

### ✔ Cluster-Label Matching Requirement  
When k = 3, k-means assigns clusters arbitrarily.  
You must relabel clusters such that diagonal sum of confusion matrix is maximized:  

- Identify majority class per cluster  
- Rearrange columns accordingly  

(assignment references StackOverflow method)

### ✔ Required Written Answer  
**Question 1:** Based on elbow_k, are there **3 species** represented in the dataset?

---

# **PART 2 — Gaussian Mixture Models (GMM)**  
Use GMM with **diag covariance_type** (required).

### ✔ Required Program Outputs

#### AIC Curve
1. Run GMM for k = 1 → 20  
2. Plot **AIC vs k**  
3. Select **aic_elbow_k**  
4. Predict labels using k = aic_elbow_k  
   - Print confusion matrix  
   - Print accuracy *only if aic_elbow_k = 3*  
     - Otherwise: print required “Cannot calculate Accuracy Score…” message  

#### BIC Curve
5. Run GMM again (k = 1 → 20)  
6. Plot **BIC vs k**  
7. Select **bic_elbow_k**  
8. Predict labels using k = bic_elbow_k  
   - Print confusion matrix  
   - Print accuracy *only if bic_elbow_k = 3*  

### ✔ Required Written Answers  
- **Question 2a:** Based on **AIC** elbow, are there 3 species?  
- **Question 2b:** Based on **BIC** elbow, are there 3 species?

---

# **PART 3 — Self-Organizing Map (SOM)**  

### ✔ Setup Requirements
- Use **MiniSom** from:  
  https://github.com/JustGlowing/minisom  
- Normalize each feature to [0, 1] using:  
  \[
  f(x) = \frac{x - \min(x)}{\max(x) - \min(x)}
  \]

### ✔ Required Grid Sizes  
Train SOMs with grid sizes:

- **3×3**  
- **7×7**  
- **15×15**  
- **25×25**

### ✔ Required Program Outputs  

For each grid size (4 total):

1. **U-Matrix plot** (distance map)  
2. **SOM response plot** with species markers  
3. **Quantization Error**  

Then produce:  
4. **Plot: Quantization Error vs Grid Size**

### ✔ Required Written Answers  
- **Question 3a:** Based on QE elbow, which grid size should be selected?  
- **Question 3b:** How does grid size affect SOM performance?  
- **Question 3c:** Which grid (7×7 or 25×25) is a “perfect fit” and why?

---

## 📁 Folder Contents

- `CompareClusters.py`  
- `PlottingCode.py`  
- `iris.csv`  
- `EECS658_Coding_Assignment6_Written_Questions.pdf`  
- `Rubric 6.docx`  
- `Results&Plots/`
  - `K-means_Reconstruction_Error_vs_k.png`
  - `GMM_Reconstruction_Error_vs_k.png`
  - `Part1&2_results.png`
  - `Part3_results.png`
  - `UMatrix_3x3.png`, `UMatrix_7x7.png`, `UMatrix_15x15.png`, `UMatrix_25x25.png`
  - `QuantizationError_vs_GridSize.png`

---

## 🔧 How to Run

```bash
python CompareClusters.py
```
---

# Assignment 7 — Gridworld RL (Policy Iteration & Value Iteration)

### **Assignment 7 – Reinforcement Learning: Policy Iteration & Value Iteration**  
**Folder:** `MalekKchaou_Assignment7`

This assignment implements **Model-Based Reinforcement Learning** on a modified **5×5 Gridworld**.  
You will implement **Policy Iteration** (Part 1) and **Value Iteration** (Part 2) to compute the optimal value function and the optimal policy π\*.  
:contentReference[oaicite:1]{index=1}

The Gridworld is identical to the version used in lecture, except it is **5×5 instead of 4×4**, with the two terminal states located at the **upper-left** and **lower-right** grey squares.

---

## 📌 **Deliverables (as required in the instructions)**

1. ✔ `Rubric7.docx` with your name and ID (do **not** submit as PDF)  
2. ✔ All Python source code  
3. ✔ A **screen print** showing successful execution of the code  
4. ✔ Part 1 — Plot of error vs. t (with ε labeled)  
5. ✔ Part 1 — Written answer to Question 1  
6. ✔ Part 2 — Plot of error vs. t (with ε labeled)  
7. ✔ Part 2 — Written answer to Question 2  
8. ✔ Part 2 — Written answer to Question 3  

---

# **Gridworld Definition (Shared by Both Parts)**

- Grid size: **5×5**  
- States (s): 25 states numbered **0–24**  
- Actions (a): `up`, `down`, `left`, `right`  
- Rewards (r):
  - `-1` for all transitions  
  - `0` for entering a terminal state  
  - `-1` for hitting a wall (state does not change)  
- Transition probabilities:  
  \[
  p(r_{t+1} \mid s_t, a_t) = 0.25,\qquad
  p(s_{t+1} \mid s_t, a_t) = 0.25
  \]
- Discount factor:  
  \[
  \gamma = 1
  \]
- Termination squares: **top-left** and **bottom-right**

---

# **PART 1 — Policy Iteration**

Write a Python program that performs full **Policy Iteration**:

### ✔ Required Program Behavior

1. **Initialize**:
   - Policy: can be arbitrary (e.g., random or all “right”)
   - Value function: all zeros  
2. **Policy Evaluation**:
   - Update V(s) for the current policy  
   - Print value grid at:
     - Iteration **0**
     - Iteration **1**
     - Iteration **10**
     - **Final iteration**  
3. **Policy Improvement**:
   - Update the policy by choosing the action that leads to the **largest V(s′)**  
4. **Convergence Criterion**:
   - Must implement a method of your choice (examples discussed in lecture)
   - Must explain in **Question 1** why you chose this method  
5. **Plot**:
   - Error vs iteration:
     \[
     |V_k(s_n) - V_{k-1}(s_n)|
     \]
   - Label the ε threshold  

### ✔ Required Output Format

Print the policy grid π(s) using action symbols:





