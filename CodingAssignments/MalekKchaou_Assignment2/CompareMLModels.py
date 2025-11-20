"""
================================================================================
Name of program: EECS 658 Assignment 2 - CompareMLModels: Iris Classification

Brief description: 
This program evaluates multiple machine learning models on the iris dataset using 
2-fold cross-validation. The models include:
- Linear Regression
- Polynomial Regression (degree 2 and 3)
- Gaussian Naive Bayes
- k-Nearest Neighbors (k=5)
- Linear Discriminant Analysis (LDA)
- Quadratic Discriminant Analysis (QDA)

For each model, the program outputs a confusion matrix and overall accuracy.

Inputs:
- iris.csv file containing iris flower measurements and species labels
- Features: sepal_length, sepal_width, petal_length, petal_width
- Target: species (setosa, versicolor, virginica)

Outputs:
- Confusion matrix for each model
- Overall accuracy of each model as a percentage

Collaborators: None

Other sources: ChatGPT, scikit-learn.org

Author: Malek Kchaou

Creation date: September 9, 2025
================================================================================
"""

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold                    # For 2-fold cross-validation
from sklearn.linear_model import LinearRegression            # Linear regression model
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.naive_bayes import GaussianNB                   # Naive Bayes classifier
from sklearn.neighbors import KNeighborsClassifier           # kNN classifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, accuracy_score # Evaluation metrics


def evaluate_model(model, X, y, model_name, poly=None):
    """
    Evaluate a given ML model using 2-fold cross-validation.
    If poly is provided, polynomial feature expansion will be applied.
    """

    # Create a 2-fold cross-validation object (shuffle ensures randomness, random_state makes it reproducible)
    kf = KFold(n_splits=2, shuffle=True, random_state=42)

    # Lists to store all true and predicted labels across both folds
    y_true_all, y_pred_all = [], []

    # Loop through the 2 folds
    for train_idx, test_idx in kf.split(X):
        # Split the dataset into training and testing parts based on fold indices
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Apply polynomial feature expansion if this is a polynomial regression model
        if poly is not None:
            X_train = poly.fit_transform(X_train)   # Expand training features
            X_test = poly.transform(X_test)         # Expand test features using the same transformation

        # Train the model on training data
        model.fit(X_train, y_train)

        # Predict labels for the test set
        y_pred = model.predict(X_test)

        # Special case: regression models give continuous outputs → convert to class labels
        if isinstance(model, LinearRegression):
            y_pred = np.round(y_pred).astype(int)                   # Round predictions to nearest integer
            y_pred = np.clip(y_pred, 0, len(np.unique(y)) - 1)      # Ensure values stay within class label range

        # Add the test set results to the running list
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    # After both folds: compute confusion matrix and accuracy
    cm = confusion_matrix(y_true_all, y_pred_all)
    acc = accuracy_score(y_true_all, y_pred_all)

    # Display results for this model
    print(f"\n=== {model_name} ===")
    print("Confusion Matrix:\n", cm)
    print("Accuracy:", acc)


def main():
    # Load the iris dataset (CSV format, no header in file so we add column names manually)
    iris_df = pd.read_csv("iris.csv", header=None,
                          names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])

    # Features (all columns except last) → X
    X = iris_df.iloc[:, :-1].values

    # Labels (last column = species) → y
    y = iris_df.iloc[:, -1].values

    # Convert string labels ("setosa", "versicolor", "virginica") into integer labels (0,1,2)
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Define all models we want to evaluate
    # Each entry = (name of model, model object, polynomial feature transformer or None)
    models = [
        ("Linear Regression", LinearRegression(), None),
        ("Polynomial Regression (degree=2)", LinearRegression(), PolynomialFeatures(degree=2)),
        ("Polynomial Regression (degree=3)", LinearRegression(), PolynomialFeatures(degree=3)),
        ("Naive Bayes (GaussianNB)", GaussianNB(), None),
        ("kNN (k=5)", KNeighborsClassifier(n_neighbors=5), None),
        ("LDA", LinearDiscriminantAnalysis(), None),
        ("QDA", QuadraticDiscriminantAnalysis(), None)
    ]

    # Run evaluation for each model in the list
    for name, model, poly in models:
        evaluate_model(model, X, y, name, poly)


# Run the program
if __name__ == "__main__":
    main()

