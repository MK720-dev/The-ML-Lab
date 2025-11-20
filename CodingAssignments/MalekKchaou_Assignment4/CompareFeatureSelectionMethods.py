# -*- coding: utf-8 -*-
# The encoding declaration ensures that non-ASCII characters are interpreted correctly.

"""
================================================================================
Name of program: EECS 658 Assignment 4 - CompareFeatureSelectionMethods: 
                Feature Selection on Iris Dataset

Brief description:
This program applies multiple feature selection and transformation techniques 
to the Iris dataset, and evaluates their performance using 2-fold 
cross-validation with a Decision Tree classifier. It consists of four main parts:

1. Part 1 – Baseline Evaluation:
   Uses all four original Iris features (sepal length, sepal width, 
   petal length, petal width) to train and evaluate a Decision Tree model.

2. Part 2 – PCA Feature Transformation:
   Performs Principal Component Analysis (PCA) to generate transformed features 
   (z1–z4), computes eigenvalues, eigenvectors, and cumulative Proportion of 
   Variance (PoV), and evaluates model accuracy using selected principal components.

3. Part 3 – Simulated Annealing Feature Selection:
   Uses simulated annealing to search for the best subset of features 
   from the combined set of 8 total features (4 original + 4 PCA-derived).
   Displays feature subsets, accuracy, acceptance probability, random uniform 
   values, and iteration status at each step.

4. Part 4 – Genetic Algorithm Feature Selection:
   Implements a genetic algorithm using the feature subsets specified in the 
   assignment as the initial population. Uses union and intersection for crossover 
   and random add/delete/replace for mutation. Displays top 5 feature sets and 
   accuracy per generation, and outputs the final best subset.

Inputs:
- iris.csv file containing iris flower measurements and species labels
- Features: sepal_length, sepal_width, petal_length, petal_width
- Target: species (setosa, versicolor, virginica)

Outputs:
- Confusion matrix and accuracy for each part
- Eigenvalues, eigenvectors, and PoV from PCA
- Iteration details for simulated annealing
- Generation details for genetic algorithm
- Final best feature subset and overall accuracy

Collaborators: None

Other sources:
- ChatGPT (guidance and documentation generation)
- scikit-learn.org (model and PCA documentation)

Author: Malek Kchaou

Creation date: October 4, 2025
================================================================================
"""


# Import required libraries
import random                                # For random number generation (used in SA and GA)
from sklearn.model_selection import KFold    # For K-Fold cross-validation
from sklearn.decomposition import PCA         # For PCA feature transformation
from sklearn.tree import DecisionTreeClassifier  # For decision tree classifier
from sklearn.preprocessing import LabelEncoder   # For converting string labels to numeric codes
from sklearn.metrics import confusion_matrix, accuracy_score  # For evaluation metrics
import numpy as np                            # For numerical operations
import pandas as pd                           # For data loading and DataFrame creation
import math                                   # For mathematical functions (e.g., exp)

# ===========================================================
# Part 1 – Using all 4 original features of the Iris dataset
# ===========================================================

# Load the Iris dataset from CSV (no header, so we assign column names manually)
iris_df = pd.read_csv("iris.csv", 
                      header=None, 
                      names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])

# Split features (X) and target labels (y)
X = iris_df.iloc[:, :-1].values              # Extract all columns except the last one (features)
y_str = iris_df.iloc[:, -1].values           # Extract the last column (species names)

# Encode string labels (species names) into numeric integers (0,1,2)
le = LabelEncoder()
y = le.fit_transform(y_str)

# Create and display a mapping table between label names and encoded integers
label_table = pd.DataFrame({
    'Label': le.classes_,
    'Encoded Value': range(len(le.classes_))
})
print(label_table)

# Store feature names for later display
features = iris_df.columns[:-1].tolist()

# Create 2-fold cross-validation setup
kf = KFold(n_splits=2, shuffle=True, random_state=42)

# Initialize arrays to store all predictions and ground truth across folds
y_true_all = []
y_pred_all = []

# Loop through 2 folds for training/testing
for train_index, test_index in kf.split(X):
    X_train, y_train = X[train_index], y[train_index]   # Training data
    X_test, y_test = X[test_index], y[test_index]       # Testing data

    # Train Decision Tree on current fold
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)                        # Predict on test data

    # Accumulate predictions and true labels
    y_true_all.extend(y_test)
    y_pred_all.extend(y_pred)

# Convert lists to NumPy arrays for evaluation
y_true_all = np.array(y_true_all)
y_pred_all = np.array(y_pred_all)

# Compute confusion matrix and accuracy for Part 1
cm = confusion_matrix(y_true_all, y_pred_all)
acc = accuracy_score(y_true_all, y_pred_all)

# Display results for Part 1
print("========== Part 1 ==========")
print("Features used:", features)
print("\nConfusion Matrix:")
print(cm)
print("\nAccuracy:", acc)


# ===================================
# Part 2 – PCA Feature Transformation
# ===================================

# Create PCA model keeping 4 components (equal to number of original features)
pca = PCA(n_components=4)
pca.fit(X)                                   # Fit PCA to the original data (compute eigenvalues/vectors)

# Retrieve eigenvalues (variance captured by each component)
eigenvalues = pca.explained_variance_
# Retrieve eigenvectors (principal component directions)
eigenvectors = pca.components_

# Display eigenvalues and eigenvectors
print("\n========== Part 2 ==========")
print("Eigenvalues:\n", eigenvalues)
print("\nEigenvectors (each row corresponds to a principal component):\n", eigenvectors)

# Transform original features into PCA feature space (z1…z4)
principleComponents = pca.transform(X)

# Compute cumulative Proportion of Variance (PoV) explained by components
pov = np.cumsum(pca.explained_variance_ratio_)
print("\nCumulative Proportion of Variance (PoV):", pov)

# Determine smallest number of components such that PoV > 0.90
n_components_selected = np.argmax(pov > 0.90) + 1
print(f"\nNumber of components selected (PoV > 0.90): {n_components_selected}")

# Create labels for selected PCA features (e.g., ['z1','z2',...])
pca_features = [f"z{i+1}" for i in range(n_components_selected)]

# Combine eigenvalues and eigenvectors for verification or manual reconstruction
eigen_pairs = list(zip(eigenvalues, eigenvectors))
# Sort pairs by descending eigenvalue
eigen_pairs.sort(key=lambda x: x[0], reverse=True)

# Construct transformation matrix W manually using top principal components
W = np.column_stack([pair[1] for pair in eigen_pairs[:n_components_selected]])
# Perform manual PCA transformation using mean-centered data
Z = (X - pca.mean_).dot(W)

# (Optional comparison of manual vs scikit-learn PCA)
# comparison = np.isclose(Z, principleComponents[:, :n_components_selected], atol=1e-5)
# print(comparison)

# Evaluate PCA-transformed data using Decision Tree + 2-fold CV
y_true_all = []
y_pred_all = []

for train_index, test_index in kf.split(Z):
    X_train, X_test = Z[train_index], Z[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    y_pred_all.extend(y_pred)
    y_true_all.extend(y_test)

# Compute confusion matrix and accuracy for PCA features
cm = confusion_matrix(y_true_all, y_pred_all)
acc = accuracy_score(y_true_all, y_pred_all)

# Display PCA evaluation results
print("Features used:", pca_features)
print("\nConfusion Matrix:")
print(cm)
print("\nAccuracy:", acc)


# ===================================
# Part 3 – Simulated Annealing Feature Selection
# ===================================

# Combine 8 total features: 4 original + 4 PCA components
X_combined = np.hstack((X, principleComponents))  # Combined data matrix (150×8)
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 
                 'z1', 'z2', 'z3', 'z4']


# ----------------------------------------------------------
# Helper: evaluate a feature subset using 2-fold CV Decision Tree
# ----------------------------------------------------------
def evaluate_subset(feature_mask):
    """Compute confusion matrix and accuracy for a selected feature subset."""
    selected_indices = [i for i, keep in enumerate(feature_mask) if keep]
    if not selected_indices:                         # If no features selected
        return np.zeros((3, 3)), 0                   # Return dummy results

    X_sel = X_combined[:, selected_indices]           # Select subset of features
    
    kf = KFold(n_splits=2, shuffle=True, random_state=42)
    y_true_all = []
    y_pred_all = []

    for train_idx, test_idx in kf.split(X_sel):
        X_train, X_test = X_sel[train_idx], X_sel[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    # Compute confusion matrix and accuracy
    cm = confusion_matrix(y_true_all, y_pred_all)
    acc = accuracy_score(y_true_all, y_pred_all)
    return cm, acc


# --- Initialize SA parameters ---
iterations = 100            # Number of iterations
c = 1                       # Cooling constant
restart_interval = 10       # Restart frequency

# Initialize random starting subset (binary mask)
current_mask = np.random.choice([0, 1], size=8).astype(bool)
current_cm, current_acc = evaluate_subset(current_mask)
best_mask = current_mask.copy()
best_cm = current_cm
best_acc = current_acc 

print("\n========== Part 3 ==========")
print("Starting simulated annealing feature selection...\n")

# --- Main Simulated Annealing Loop ---
for i in range(1, iterations + 1):
    # Restart the search periodically
    if i % restart_interval == 0:
        current_mask = np.random.choice([0, 1], size=8).astype(bool)
        current_cm, current_acc = evaluate_subset(current_mask)
        status = "Restart"
    else:
        # Perturb the current subset by flipping 1 or 2 feature bits
        new_mask = current_mask.copy()
        num_to_flip = random.choice([1, 2])
        flip_indices = random.sample(range(8), num_to_flip)
        for idx in flip_indices:
            new_mask[idx] = not new_mask[idx]

        # Evaluate new subset
        new_cm, new_acc = evaluate_subset(new_mask)
        delta = new_acc - current_acc

        # Acceptance probability based on decay formula
        P_accept = math.exp(-(i / c) * ((current_acc - new_acc) / current_acc))
        U = random.random()  # Random uniform number for acceptance decision

        # Decide whether to accept new subset
        if delta > 0:
            current_mask, current_cm, current_acc = new_mask, new_cm, new_acc
            status = "Improved"
        elif U < P_accept:
            current_mask, current_cm, current_acc = new_mask, new_cm, new_acc
            status = "Accepted"
        else:
            status = "Discarded"
    
    # Keep track of global best subset
    if current_acc > best_acc:
        best_acc, best_cm = current_acc, current_cm
        best_mask = current_mask.copy()

    # Display iteration information
    subset_features = [feature_names[j] for j, keep in enumerate(current_mask) if keep]
    print(f"Iteration {i:03d} | Features: {subset_features} | "
          f"Accuracy: {current_acc:.4f} | Pr[accept]: {P_accept:.4f} | "
          f"RandU: {U:.4f} | Status: {status}")

# Print final best results for SA
print("\nBest subset found:")
best_features = [feature_names[j] for j, keep in enumerate(best_mask) if keep]
print("Features:", best_features)
print("Confusion Matrix\n", best_cm)
print("Accuracy:", best_acc)


# ===================================
# Part 4 – Genetic Algorithm Feature Selection
# ===================================

print("\n========== Part 4 ==========")
print("Running Genetic Algorithm for Feature Selection...\n")

# Step 1 – Initialize population (as specified in assignment)
initial_sets = [
    {'z1', 'sepal_length', 'sepal_width', 'petal_length', 'petal_width'},
    {'z1', 'z2', 'sepal_width', 'petal_length', 'petal_width'},
    {'z1', 'z2', 'z3', 'sepal_width', 'petal_length'},
    {'z1', 'z2', 'z3', 'z4', 'sepal_width'},
    {'z1', 'z2', 'z3', 'z4', 'sepal_length'}
]      

# Convert each feature set into boolean mask representation
population = []
for s in initial_sets:
    mask = np.array([f in s for f in feature_names], dtype=bool)
    population.append(mask)

generations = 50            # Number of generations for GA

# Create cache to avoid recomputing fitness for identical feature sets
evaluation_cache = {}

def mask_to_tuple(mask):
    """Convert NumPy boolean mask to hashable tuple for dictionary keys."""
    return tuple(mask)

def evaluate_with_cache(mask):
    """Evaluate subset accuracy with memoization to save time."""
    key = mask_to_tuple(mask)
    if key not in evaluation_cache:
        cm, acc = evaluate_subset(mask)
        evaluation_cache[key] = (cm, acc)
    return evaluation_cache[key]


# ----------------------------------------------------------
# Mutation operator (with probability to preserve good sets)
# ----------------------------------------------------------
def mutate(mask, mutation_prob=0.3):
    """Randomly mutate a feature mask (add/delete/replace) with given probability."""
    # Skip mutation sometimes (1 - mutation_prob) to retain elite individuals
    if random.random() > mutation_prob:
        return mask.copy()
    
    new_mask = mask.copy()
    mutation_type = random.choice(['add', 'delete', 'replace'])
    selected_indices = np.where(new_mask)[0]
    unselected_indices = np.where(~new_mask)[0]

    # Add a new feature
    if mutation_type == 'add' and len(unselected_indices) > 0:
        new_mask[random.choice(unselected_indices)] = True
    # Delete an existing feature
    elif mutation_type == 'delete' and len(selected_indices) > 1:
        new_mask[random.choice(selected_indices)] = False
    # Replace one feature with another
    elif mutation_type == 'replace' and len(selected_indices) > 0 and len(unselected_indices) > 0:
        drop = random.choice(selected_indices)
        add = random.choice(unselected_indices)
        new_mask[drop] = False
        new_mask[add] = True
    return new_mask


# ----------------------------------------------------------
# Utility: remove duplicate masks to keep population unique
# ----------------------------------------------------------
def remove_duplicates(masks):
    """Remove duplicate individuals from list while preserving order."""
    unique = []
    seen = set()
    for mask in masks:
        key = mask_to_tuple(mask)
        if key not in seen:
            unique.append(mask)
            seen.add(key)
    return unique


# ----------------------------------------------------------
# Main Genetic Algorithm Evolution Loop
# ----------------------------------------------------------
for gen in range(1, generations + 1):
    # Evaluate all individuals (using cached results if available)
    scored = []
    for mask in population:
        cm, acc = evaluate_with_cache(mask)
        scored.append((mask, acc, cm))

    # Sort population: first by accuracy, then by fewer features (for simplicity)
    scored.sort(key=lambda x: (x[1], -np.sum(x[0])), reverse=True)
    best_masks = [m for (m, _, _) in scored[:5]]  # Keep top 5 elite individuals

    # Generate new offspring through union and intersection crossover
    offspring = []
    for i in range(len(best_masks)):
        for j in range(i + 1, len(best_masks)):
            union = np.logical_or(best_masks[i], best_masks[j])   # Combine features
            inter = np.logical_and(best_masks[i], best_masks[j])  # Keep shared features
            offspring.extend([union, inter])

    # Combine parents and offspring for next generation
    all_candidates = best_masks + offspring
    all_candidates = remove_duplicates(all_candidates)  # Remove identical sets

    # Apply mutation to all candidates
    mutated_candidates = []
    for mask in all_candidates:
        mutated = mutate(mask, mutation_prob=0.3)
        mutated_candidates.append(mutated)
    mutated_candidates = remove_duplicates(mutated_candidates)

    # Evaluate all unique candidates
    evaluated = []
    for mask in mutated_candidates:
        cm, acc = evaluate_with_cache(mask)
        evaluated.append((mask, acc, cm))

    # Sort candidates by fitness (accuracy, then smaller subsets)
    evaluated.sort(key=lambda x: (x[1], -np.sum(x[0])), reverse=True)
    # Select top 5 individuals for next generation
    population = [m for (m, _, _) in evaluated[:5]]

    # Display generation summary (required by assignment)
    print(f"\nGeneration {gen}")
    print("-" * 60)
    for rank, (mask, acc, cm) in enumerate(evaluated[:5], 1):
        selected_feats = [f for f, keep in zip(feature_names, mask) if keep]
        print(f"{rank}. {selected_feats} | Accuracy: {acc:.4f}")


# ----------------------------------------------------------
# Final Evaluation and Best Feature Set Reporting
# ----------------------------------------------------------
final_results = []
for mask in population:
    cm, acc = evaluate_with_cache(mask)
    final_results.append((mask, acc, cm))

# Sort final population again by accuracy (and fewer features)
final_results.sort(key=lambda x: (x[1], -np.sum(x[0])), reverse=True)

# Extract the single best individual
best_mask, best_acc, best_cm = final_results[0]
best_features = [f for f, keep in zip(feature_names, best_mask) if keep]

# Print final GA output
print("\n========== FINAL BEST FEATURE SET ==========")
print("Features:", best_features)
print("Number of features:", len(best_features))
print("Confusion Matrix:\n", best_cm)
print(f"Accuracy: {best_acc:.4f}")
print(f"\nTotal unique feature sets evaluated: {len(evaluation_cache)}")
