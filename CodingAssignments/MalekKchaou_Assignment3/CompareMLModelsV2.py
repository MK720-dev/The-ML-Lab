"""
================================================================================
Name of program: EECS 658 Assignment 3 - CompareMLModelsV2: Iris Classification

Brief description:
This program evaluates twelve machine learning models on the iris dataset using
2-fold cross-validation. The models include:
- Linear Regression
- Polynomial Regression (degree 2 and 3)
- Gaussian Naive Bayes
- k-Nearest Neighbors (k=5)
- Linear Discriminant Analysis (LDA)
- Quadratic Discriminant Analysis (QDA)
- Support Vector Machine (LinearSVC)
- Decision Tree
- Random Forest
- Extra Trees
- Neural Network (MLPClassifier)

For each model, the program outputs a confusion matrix and overall accuracy.

Inputs:
- iris.csv file containing iris flower measurements and species labels
- Features: sepal_length, sepal_width, petal_length, petal_width
- Target: species (setosa, versicolor, virginica)

Outputs:
- Confusion matrix for each model
- Overall accuracy of each model as a percentage
-Total Samples Evaluated 

Collaborators: None

Other sources: ChatGPT, scikit-learn.org

Author: Malek Kchaou

Creation date: September 25, 2025
================================================================================
"""

# ===============================
# Imports
# ===============================

import numpy as np                                  # Numerical arrays and utilities
import pandas as pd                                 # Data loading and manipulation

# Model selection / cross-validation
from sklearn.model_selection import KFold           # KFold for 2-fold cross-validation

# Base regressors/classifiers and preprocessing helpers
from sklearn.linear_model import LinearRegression   # Linear regression model (used for baseline + poly)
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder  # Poly features + label encoding

# Classic ML classifiers
from sklearn.naive_bayes import GaussianNB          # Gaussian Naive Bayes classifier
from sklearn.neighbors import KNeighborsClassifier  # k-Nearest Neighbors classifier
from sklearn.discriminant_analysis import (         # LDA and QDA classifiers
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
)

# SVM, tree-based, and neural network models
from sklearn.svm import LinearSVC                   # Linear SVM classifier
from sklearn.tree import DecisionTreeClassifier     # Decision Tree classifier
from sklearn.ensemble import (                      # Ensemble tree classifiers
    RandomForestClassifier, ExtraTreesClassifier
)
from sklearn.neural_network import MLPClassifier    # Multi-layer Perceptron (neural network)

# Metrics for evaluation
from sklearn.metrics import confusion_matrix, accuracy_score  # Confusion matrix + accuracy metric


# ===============================
# Configuration
# ===============================

SEED = 42  # Global random seed for reproducibility across CV splits and models


def evaluate_model(model, X, y, model_name, poly=None):
    """
    Evaluate a given ML model using 2-fold cross-validation.
    If `poly` is provided, polynomial feature expansion will be applied.

    Parameters
    ----------
    model : scikit-learn estimator
        The model to train/evaluate (must implement fit/predict).
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Integer-encoded labels of shape (n_samples,).
    model_name : str
        Name printed before results for clarity.
    poly : PolynomialFeatures or None
        If not None, applies polynomial expansion to X.
    """

    # Create a 2-fold cross-validation splitter.
    # shuffle=True randomizes the split; random_state fixes the shuffle for reproducibility.
    kf = KFold(n_splits=2, shuffle=True, random_state=SEED)

    # Storage for concatenating predictions/targets across both folds
    y_true_all = []  # collects true labels from both test folds
    y_pred_all = []  # collects predicted labels from both test folds

    # Cache the unique class labels (e.g., [0, 1, 2]) to keep confusion matrix shape consistent
    classes = np.unique(y)

    # Iterate over the two folds produced by KFold
    for train_idx, test_idx in kf.split(X):
        # Split features/labels by indices for this fold
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # If polynomial expansion is requested, fit on train and transform both train and test
        if poly is not None:
            X_train = poly.fit_transform(X_train)  # learn polynomial mapping on training data
            X_test = poly.transform(X_test)        # apply the same mapping to test data

        # Fit the model on the training fold
        model.fit(X_train, y_train)

        # Get model predictions on the test fold
        y_pred = model.predict(X_test)

        # Special handling for regression models:
        # LinearRegression predicts continuous values; convert to nearest class index [0..K-1]
        if isinstance(model, LinearRegression):
            y_pred = np.round(y_pred).astype(int)              # round to nearest integer
            y_pred = np.clip(y_pred, 0, len(classes) - 1)      # clip to valid class range

        # Accumulate results from this fold
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    # Convert accumulated lists back to arrays
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    # Compute confusion matrix with a fixed label order to ensure consistent 3x3 shape
    cm = confusion_matrix(y_true_all, y_pred_all, labels=classes)

    # Compute overall accuracy across both folds
    acc = accuracy_score(y_true_all, y_pred_all)

    # Print model header for clarity in console output
    print(f"\n=== {model_name} ===")

    # Show the confusion matrix
    print("Confusion Matrix:\n", cm)

    # Show the accuracy as both fraction and percentage (easier to read)
    print(f"Accuracy: {acc:.4f}  ({acc*100:.2f}%)")

    # Sanity check: the confusion matrix entries must sum to total number of samples (150 for iris)
    total_predictions = cm.sum()
    print("Total samples evaluated:", total_predictions)

    # If this isn't 150, something is wrong with the splitting/accumulation logic
    if total_predictions != len(y):
        print("WARNING: Confusion matrix entries do not sum to 150. Check your CV logic and data handling.")


def main():
    """
    Program entry point:
    - Loads the iris dataset from iris.csv
    - Encodes species labels to integers
    - Defines 12 models (including polynomial variants)
    - Evaluates each with 2-fold CV, printing confusion matrix and accuracy
    """

    # -------------------------------
    # Load and prepare the dataset
    # -------------------------------

    # Read iris data from CSV file with no header; we assign the column names manually.
    iris_df = pd.read_csv(
        "iris.csv",
        header=None,
        names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    )

    # Extract features X (all columns except the last)
    X = iris_df.iloc[:, :-1].values

    # Extract labels y (last column, species names)
    y_str = iris_df.iloc[:, -1].values

    # Encode string species labels (e.g., "setosa") into integers (0,1,2)
    le = LabelEncoder()
    y = le.fit_transform(y_str)

    # -------------------------------
    # Define models to evaluate
    # -------------------------------

    # Polynomial feature generators for degree 2 and 3
    poly2 = PolynomialFeatures(degree=2, include_bias=True)  # include_bias=True adds the intercept term
    poly3 = PolynomialFeatures(degree=3, include_bias=True)

    # Build a list of (display_name, estimator_instance, poly_transformer_or_None)
    models = [
        ("Linear Regression", LinearRegression(), None),
        ("Polynomial Regression (degree=2)", LinearRegression(), poly2),
        ("Polynomial Regression (degree=3)", LinearRegression(), poly3),
        ("Naive Bayes (GaussianNB)", GaussianNB(), None),
        ("kNN (k=5)", KNeighborsClassifier(n_neighbors=5), None),
        ("LDA", LinearDiscriminantAnalysis(), None),
        ("QDA", QuadraticDiscriminantAnalysis(), None),
        # LinearSVC is a linear SVM; dual=False is recommended when n_samples > n_features
        ("SVM (LinearSVC)", LinearSVC(dual=False, max_iter=5000, random_state=SEED), None),
        ("Decision Tree", DecisionTreeClassifier(random_state=SEED), None),
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=SEED), None),
        ("Extra Trees", ExtraTreesClassifier(n_estimators=100, random_state=SEED), None),
        ("Neural Network (MLP)", MLPClassifier(max_iter=2000, random_state=SEED), None),
    ]

    # -------------------------------
    # Evaluate each model
    # -------------------------------

    # Loop through every model configuration and run the evaluation function
    for name, model, poly in models:
        evaluate_model(model, X, y, name, poly)


# Standard Python "entry point" guard so this file can be imported without running main()
if __name__ == "__main__":
    main()

