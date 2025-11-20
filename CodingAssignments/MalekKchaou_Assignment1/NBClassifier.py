"""
================================================================================
Name of program: EECS 658 Assignment 1 - Iris Classification with Naive Bayes

Brief description: 
This program implements a Gaussian Naive Bayes classifier to classify iris flowers
into three species (setosa, versicolor, virginica) based on their sepal and petal
measurements. The program uses 2-fold cross-validation to evaluate model performance
and provides comprehensive evaluation metrics.

Inputs:
- iris.csv file containing iris flower measurements and species labels
- Features: sepal_length, sepal_width, petal_length, petal_width
- Target: species (setosa, versicolor, virginica)

Output:
- Overall accuracy of the classifier as a percentage
- Confusion matrix showing prediction results for each class
- Classification report with precision, recall, and F1-score for each of the 3 iris varieties
- Sum of all values in confusion matrix (total number of samples)

Collaborators: None

Other sources: ChatGPT, scikit-learn.org

Author: Malek Kchaou

Creation date: August 28, 2025
================================================================================
"""


# Import necessary libraries for machine learning and data manipulation
from sklearn.naive_bayes import GaussianNB  # Gaussian Naive Bayes classifier
from sklearn.preprocessing import LabelEncoder  # For encoding string labels to integers
from sklearn.model_selection import train_test_split, KFold  # For data splitting and cross-validation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # For model evaluation
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations

# Load the iris dataset from CSV file
# header=None indicates there are no column headers in the CSV
# names parameter assigns custom column names to the dataset
iris_df = pd.read_csv('iris.csv', header=None, 
                     names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])

# Extract features (X) - all columns except the last one (species)
# iloc[:, :-1] selects all rows and all columns except the last one
X = iris_df.iloc[:, :-1].values

# Extract target variable (y) - the last column (species)
# iloc[:, -1] selects all rows and only the last column
y = iris_df.iloc[:, -1].values

# Create a label encoder to convert string labels to numerical values
le = LabelEncoder()
# Transform species names (e.g., 'setosa', 'versicolor', 'virginica') to integers (0, 1, 2)
y = le.fit_transform(y)

# Initialize prediction array with same shape as y, filled with zeros
# This will store predictions from cross-validation
y_pred = np.zeros_like(y)

# Set up 2-fold cross-validation with shuffling for better data distribution
# random_state=42 ensures reproducible results across runs
kf = KFold(n_splits=2, shuffle=True, random_state=42)

# Perform cross-validation: iterate through each fold
for train_index, test_index in kf.split(X):
    # Step 1: Split the data into training and testing sets for this fold
    # Use indices from KFold to create train/test splits
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Step 2: Create a Gaussian Naive Bayes classifier instance
    model = GaussianNB()
    # Train the model using the training data for this fold
    model.fit(X_train, y_train)
    
    # Step 3: Make predictions on the test data for this fold
    # Store predictions in the correct positions of the overall prediction array
    y_pred[test_index] = model.predict(X_test)

# Calculate overall accuracy by comparing true labels with predictions
accuracy = accuracy_score(y, y_pred)
# Display accuracy as a percentage with 2 decimal places
print("Accuracy: {:.2f}%\n".format(accuracy*100))

# Generate confusion matrix to show prediction performance for each class
cm = confusion_matrix(y, y_pred)
print("Confusion Matrix:\n", cm)
# Print total number of samples (should equal dataset size)
print("Sum of all values in confusion matrix: {} \n".format(cm.sum()))

# Generate detailed classification report with precision, recall, and f1-score
# target_names uses original class names from label encoder
print("Classification Report:")
print(classification_report(y, y_pred, target_names=le.classes_))