"""
===============================================================================
Assignment 5: Imbalanced Iris Dataset
===============================================================================
Description:
This program performs classification on an imbalanced Iris dataset using a
Neural Network (MLPClassifier) with 2-Fold Cross Validation. It evaluates
three scenarios:
  1. Imbalanced dataset (with manual and sklearn balanced accuracy calculations)
  2. Oversampling (Random, SMOTE, ADASYN)
  3. Undersampling (Random, ClusterCentroids, TomekLinks)
-------------------------------------------------------------------------------
Inputs:
    - imbalanced_iris.csv
        - A modified version of the classic Iris dataset with class imbalance.
        - Columns: sepal_length, sepal_width, petal_length, petal_width, class
-------------------------------------------------------------------------------
Outputs:
- Printed labeled results for each part:
    - Confusion Matrix
    - Accuracy
    - Class Balanced Accuracy (manual, lecture-defined)  (only for Part 1)
    - Balanced Accuracy (manual, lecture-defined)        (only for Part 1)
    - Balanced Accuracy (scikit-learn)                   (only for Part 1)
------------------------------------------------------------------------------
Author:        Malek Kchaou
Course:        EECS 658 - Intro to Machine Learning
Instructor:    Prof. David Orville Johnson
Collaborators: None
Sources: 
               1. scikit-learn Documentation:
                   https://scikit-learn.org/stable/modules/neural_networks_supervised.html
                   https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
                   https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html

               2. imbalanced-learn Documentation:
                   https://imbalanced-learn.org/stable/references/over_sampling.html
                   https://imbalanced-learn.org/stable/references/under_sampling.html

               3. Lecture slides and notes from EECS 658 - "Imbalanced Datasets" by Prof. David Orville Johnson (University of Kansas)

               4. ChatGPT
File:          ImbalancedIris.py
Created:       10/20/2025
===============================================================================
"""

# ============= 
#   Imports     
# ============= 
import numpy as np                                    # For numerical operations and arrays
import pandas as pd                                   # For loading and manipulating datasets
from sklearn.preprocessing import LabelEncoder, StandardScaler   # For encoding labels and scaling features
from sklearn.model_selection import KFold             # For performing 2-fold cross-validation
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score  # Evaluation metrics
from sklearn.neural_network import MLPClassifier      # Neural Network classifier
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN   # Oversampling methods
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, TomekLinks  # Undersampling methods


# ===================== 
#     Load Dataset      
# ===================== 

data = pd.read_csv("imbalanced iris.csv")             # Load the imbalanced Iris dataset from CSV file
X = data.iloc[:, :-1].values                          # Extract all feature columns (sepal & petal measurements)
y_raw = data.iloc[:, -1].values                       # Extract the class column (string labels)
le = LabelEncoder()                                   # Create a label encoder to convert class names to integers
y = le.fit_transform(y_raw)                           # Encode class labels into numeric values (0, 1, 2)


# ===================== 
#   Model and K-Folds   
# ===================== 

kf = KFold(n_splits=2, shuffle=True, random_state=42) # Define 2-fold cross-validation with shuffling for reproducibility
nn = MLPClassifier(hidden_layer_sizes=(10,),          # Define a neural network with one hidden layer of 10 neurons
                   max_iter=1500, random_state=42)    # Use 1500 iterations (epochs) and fixed seed for reproducibility


# ===================== 
#  Evaluation Function  
# ===================== 
def evaluate_model(X, y, model, description=""):
    """
    Performs 2-fold cross-validation using the provided model and dataset.
    Prints the confusion matrix and accuracy for each evaluation scenario.
    """
    print("\n" + "-"*60)
    print(description)
    print("-"*60)

    scaler = StandardScaler()                         # Initialize a standard scaler for feature normalization
    X_scaled = scaler.fit_transform(X)                # Scale the feature set for better NN convergence

    y_true_all, y_pred_all = [], []                   # Initialize lists to collect predictions from both folds

    # Perform 2-fold cross-validation
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]   # Split the scaled features
        y_train, y_test = y[train_idx], y[test_idx]                 # Split the encoded labels

        model.fit(X_train, y_train)                   # Train the neural network on training data
        y_pred = model.predict(X_test)                # Predict class labels on test data

        y_true_all.extend(y_test)                     # Append true labels from this fold
        y_pred_all.extend(y_pred)                     # Append predicted labels from this fold

    cm = confusion_matrix(y_true_all, y_pred_all)     # Compute confusion matrix for combined results
    acc = accuracy_score(y_true_all, y_pred_all)      # Compute overall accuracy

    print("Confusion Matrix:\n", cm)                  # Display the confusion matrix
    print("Accuracy:", round(acc,4))                  # Display the accuracy rounded to 4 decimals

    return np.array(y_true_all), np.array(y_pred_all) # Return arrays of true and predicted labels


# ==========================================================
#                     PART 1: Imbalanced Dataset
# ==========================================================
print("\n" + "="*60)
print("Part1: Imbalanced Dataset")
print("="*60)

# Evaluate the model on the original imbalanced dataset
y_true, y_pred = evaluate_model(X, y, nn, "Training Neural Network on Imbalanced Dataset")

# Compute confusion matrix and extract class-wise metrics
cm = confusion_matrix(y_true, y_pred)
TP = np.diag(cm)                                     # True Positives: diagonal elements
FN = np.sum(cm, axis=1) - TP                         # False Negatives: row sum minus TP
FP = np.sum(cm, axis=0) - TP                         # False Positives: column sum minus TP
TN = np.sum(cm) - (TP + FN + FP)                     # True Negatives: all others

# -------------------------------
# Class Balanced Accuracy (Lecture Definition)
# -------------------------------
recall = TP / (TP + FN)                              # Recall for each class
precision = TP / (TP + FP)                           # Precision for each class
min_pr_rc = np.minimum(precision, recall)            # Take the minimum between precision and recall per class
cba = np.mean(min_pr_rc)                             # Average across all classes
print("Class Balanced Accuracy (manual):", round(cba, 4))

# -------------------------------
# Balanced Accuracy (Lecture Definition)
# -------------------------------
specificity = TN / (TN + FP)                         # Specificity for each class
manual_bal_acc = np.mean((recall + specificity) / 2) # Average recall and specificity across classes
print("Balanced Accuracy (manual):", round(manual_bal_acc, 4))

# -------------------------------
# Scikit-learn Balanced Accuracy (for comparison)
# -------------------------------
sk_bal_acc = balanced_accuracy_score(y_true, y_pred) # Built-in sklearn balanced accuracy (mean recall)
print("Balanced Accuracy (sklearn):", round(sk_bal_acc, 4))


# ==========================================================
#                     PART 2: Oversampling
# ==========================================================
print("\n" + "="*60)
print("Part 2: Oversampling")
print("="*60)

# Random Oversampling (duplicates minority samples)
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X, y)                # Apply random oversampling
evaluate_model(X_res, y_res, nn, "Random Oversampling")

# SMOTE Oversampling (synthetic samples via interpolation)
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)              # Apply SMOTE to balance classes
evaluate_model(X_res, y_res, nn, "SMOTE Oversampling")

# ADASYN Oversampling (adaptive synthetic sampling)
adasyn = ADASYN(sampling_strategy='minority', random_state=42)
X_res, y_res = adasyn.fit_resample(X, y)             # Apply ADASYN to oversample the minority class adaptively
evaluate_model(X_res, y_res, nn, "ADASYN Oversampling")


# ==========================================================
#                     PART 3: Undersampling
# ==========================================================
print("\n" + "="*60)
print("Part 3: Undersampling")
print("="*60)

# Random Undersampling (remove samples from majority class)
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X, y)                # Randomly remove samples from majority classes
evaluate_model(X_res, y_res, nn, "Random Undersampling")

# Cluster Centroids Undersampling (replace samples with centroids)
cc = ClusterCentroids(random_state=42)
X_res, y_res = cc.fit_resample(X, y)                 # Replace clusters of samples with their centroids
evaluate_model(X_res, y_res, nn, "Cluster Centroids Undersampling")

# Tomek Links Undersampling (remove borderline majority samples)
tl = TomekLinks()
X_res, y_res = tl.fit_resample(X, y)                 # Clean overlapping samples using Tomek Links
evaluate_model(X_res, y_res, nn, "Tomek Links Undersampling")
