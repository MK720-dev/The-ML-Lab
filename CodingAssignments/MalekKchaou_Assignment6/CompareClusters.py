# -*- coding: utf-8 -*-
"""
================================================================================
Name of program: EECS 658 Assignment 6

Brief description:
    This program performs three unsupervised learning tasks on the Iris dataset:
    1. K-Means clustering (Part 1)
    2. Gaussian Mixture Models with AIC/BIC model selection (Part 2)
    3. Self-Organizing Map (SOM) visualization and quantization analysis (Part 3)

    For each part, the program computes and visualizes model evaluation metrics
    such as reconstruction error, confusion matrices, and quantization error.

Inputs:
    - iris.csv : Dataset file containing iris feature measurements and species labels.

Outputs:
    - Plots of reconstruction error (K-Means)
    - Plots of AIC and BIC vs k (GMM)
    - Confusion matrices and accuracies (K-Means and GMM)
    - U-Matrix plots and quantization error vs grid size (SOM)
    - Printed quantization errors for each SOM grid

Collaborators: None

Other sources:
    - MiniSom library: https://github.com/JustGlowing/minisom
    - MiniSom Iris example: https://github.com/JustGlowing/minisom/blob/master/examples/BasicUsage.ipynb
    - scikit-learn documentation for KMeans and GaussianMixture

Author: Malek Kchaou

Creation date: November 6, 2025
================================================================================
"""

# =================================================================================
#       Imports
# =================================================================================
import pandas as pd                     # For reading and manipulating tabular data
import numpy as np                      # For numerical computations
import matplotlib.pyplot as plt         # For data visualization
from sklearn.cluster import KMeans      # K-Means clustering
from sklearn.preprocessing import LabelEncoder   # Encode class labels numerically
from sklearn.mixture import GaussianMixture       # Gaussian Mixture Model clustering
from sklearn.metrics import confusion_matrix, accuracy_score  # Evaluation metrics
from scipy.optimize import linear_sum_assignment  # For best cluster–class label matching
from PlottingCode import plot_graph               # Custom plotting function (provided)
from minisom import MiniSom                       # Self-Organizing Map implementation


# =============================================================================
#                      LOAD AND PREPARE THE IRIS DATASET
# =============================================================================
"""
This section loads the dataset and encodes categorical species labels as numeric values.
We extract:
    X — the feature matrix (4 numeric attributes per flower)
    y — the encoded labels (Setosa=0, Versicolor=1, Virginica=2)
"""

# Load CSV file into a Pandas DataFrame
data = pd.read_csv("iris.csv")

# Separate input features (sepal/petal length/width) and target species
X = data.iloc[:, :-1].values
y_raw = data.iloc[:, -1].values

# Convert string labels (species names) into integer codes (0, 1, 2)
le = LabelEncoder()
y = le.fit_transform(y_raw)


# =============================================================================
#                               PART 1: K-MEANS CLUSTERING
# =============================================================================
"""
K-Means groups the data into k clusters by minimizing within-cluster variance.
We compute the reconstruction error (inertia) for k = 1 → 20 and plot the curve
to find the "elbow" — the optimal number of clusters.

Once elbow_k is selected (manually = 3 for Iris), we:
    • Compute confusion matrix
    • Align cluster labels with true labels using the Hungarian algorithm
    • Compute accuracy (only if k == number of classes)
"""
print("=========== Part 1 ===============\n")
inertias = []                    # Store total within-cluster variance for each k
K = range(1, 21)                 # k values from 1 to 20

# --- Compute inertia (reconstruction error) for each k ---
for k in K:
    km = KMeans(n_clusters=k, random_state=42)  # Initialize K-Means
    km.fit(X)                                   # Run clustering
    inertias.append(km.inertia_)                # Store within-cluster sum of squares

# Plot the reconstruction error curve
plot_graph(inertias, "K-means Reconstruction Error", l1="Error")

# Elbow point (chosen manually based on the curve)
k_elbow = 3

# Fit K-Means model with elbow_k clusters
km_elbow = KMeans(n_clusters=k_elbow, random_state=42)
y_pred_elbow = km_elbow.fit_predict(X)

# Compute confusion matrix (rows = true labels, columns = clusters)
cm = confusion_matrix(y, y_pred_elbow)

# --- Compute accuracy if number of clusters = number of classes ---
if k_elbow == 3:
    # Hungarian algorithm to optimally map clusters to true labels
    row_ind, col_ind = linear_sum_assignment(-cm)
    corrected_cm = cm[:, col_ind]                   # Permute columns for best match
    accuracy = corrected_cm.trace() / corrected_cm.sum()
    print("Confusion Matrix (aligned):\n", corrected_cm)
    print(f"Accuracy (K-Means, k={k_elbow}): {accuracy:.3f}")
else:
    # Cannot compute accuracy if dimensions differ
    print("Confusion Matrix:\n", cm)
    print("Cannot calculate accuracy: number of clusters != number of classes.")


# =============================================================================
#                     PART 2: GAUSSIAN MIXTURE MODELS (GMM)
# =============================================================================
"""
GMM extends K-Means by modeling each cluster as a Gaussian distribution.
We use both AIC and BIC to evaluate model fit and complexity for k = 1 → 20.
Lower AIC/BIC indicates a better tradeoff between data likelihood and complexity.

Steps:
    1. Fit GMM for each k and compute AIC/BIC.
    2. Plot AIC and BIC vs k.
    3. Choose elbow_k from each curve (manually = 3).
    4. Compute confusion matrix and accuracy (if k == 3).
"""
print("\n=========== Part 2 ===============")

aic_values, bic_values = [], []

# --- Compute AIC and BIC for GMMs with increasing components ---
for k in K:
    gmm = GaussianMixture(n_components=k, covariance_type='diag', random_state=42)
    gmm.fit(X)
    aic_values.append(gmm.aic(X))  # AIC = 2p - 2ln(L)
    bic_values.append(gmm.bic(X))  # BIC = p*ln(N) - 2ln(L)

# Plot AIC/BIC vs k for model selection
plot_graph(aic_values, "GMM Reconstruction Error", arr2=bic_values, l1="AIC", l2="BIC")


# --- Evaluate AIC-based model ---
aic_elbow_k = 3  # Chosen manually from curve
gmm_aic = GaussianMixture(n_components=aic_elbow_k, covariance_type='diag', random_state=42)
gmm_aic.fit(X)
y_pred_aic = gmm_aic.predict(X)
cm_aic = confusion_matrix(y, y_pred_aic)

if aic_elbow_k == 3:
    row_ind, col_ind = linear_sum_assignment(-cm_aic)
    cm_aic_reordered = cm_aic[:, col_ind]
    acc_aic = cm_aic_reordered.trace() / cm_aic_reordered.sum()
    print("Confusion Matrix (AIC, reordered):\n", cm_aic_reordered)
    print(f"Accuracy (AIC, k={aic_elbow_k}): {acc_aic:.3f}")
else:
    print("Confusion Matrix (AIC):\n", cm_aic)
    print("Cannot calculate accuracy (AIC): number of clusters != number of classes.")


# --- Evaluate BIC-based model ---
bic_elbow_k = 3  # Chosen manually from curve
gmm_bic = GaussianMixture(n_components=bic_elbow_k, covariance_type='diag', random_state=42)
gmm_bic.fit(X)
y_pred_bic = gmm_bic.predict(X)
cm_bic = confusion_matrix(y, y_pred_bic)

if bic_elbow_k == 3:
    row_ind, col_ind = linear_sum_assignment(-cm_bic)
    cm_bic_reordered = cm_bic[:, col_ind]
    acc_bic = cm_bic_reordered.trace() / cm_bic_reordered.sum()
    print("Confusion Matrix (BIC, reordered):\n", cm_bic_reordered)
    print(f"Accuracy (BIC, k={bic_elbow_k}): {acc_bic:.3f}")
else:
    print("Confusion Matrix (BIC):\n", cm_bic)
    print("Cannot calculate accuracy (BIC): number of clusters != number of classes.")


# =============================================================================
#                       PART 3: SELF-ORGANIZING MAP (SOM)
# =============================================================================
"""
SOMs create a 2D map that preserves topological relationships in high-dimensional data.

Procedure:
    1. Normalize input features to [0, 1] (min–max normalization).
    2. Train SOMs of different grid sizes (3×3, 7×7, 15×15, 25×25).
    3. Visualize each SOM’s U-Matrix (distance map between neurons).
    4. Compute quantization error (average distance between each data point and its BMU).
    5. Plot quantization error vs grid size to observe convergence behavior.
"""
print("\n=========== Part 3 ===============")

# --- Normalize features using min–max scaling ---
X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))


# -------------------------------------------------------------------------
# Function: train_som
# Trains a MiniSom object of given grid size using normalized data.
# -------------------------------------------------------------------------
def train_som(x, y, size, iters=1000):
    som = MiniSom(x=size, y=size, input_len=x.shape[1],
                  sigma=1.0, learning_rate=0.5, random_seed=42)
    som.random_weights_init(x)                # Randomly initialize neuron weights
    som.train_random(data=x, num_iteration=iters, verbose=True)  # Train using random sampling
    return som


# -------------------------------------------------------------------------
# Function: plot_umatrix
# Generates a U-Matrix plot (inter-neuron distances) with sample markers
# overlaid to show class distributions.
# -------------------------------------------------------------------------
def plot_umatrix(som, x, y, grid_size):
    plt.figure(figsize=(6,6))
    plt.pcolor(som.distance_map().T, cmap='bone_r')   # Distance map visualization
    plt.colorbar(label='Distance')

    # Define marker shape and color for each true species class
    markers = ['o', 's', 'D']
    colors = ['r', 'g', 'b']

    # Overlay each input sample at its BMU location
    for i, xi in enumerate(x):
        w = som.winner(xi)   # Coordinates of Best Matching Unit
        plt.plot(w[0]+0.5, w[1]+0.5,
                 markers[y[i]], markerfacecolor='None',
                 markeredgecolor=colors[y[i]], markersize=8, markeredgewidth=2)

    plt.title(f'U-Matrix for {grid_size}x{grid_size} SOM')
    plt.savefig(f"Plots/UMatrix_{grid_size}x{grid_size}.png")
    plt.show()


# -------------------------------------------------------------------------
# Function: quantization_error
# Computes the average Euclidean distance between input vectors and the
# weight vector of their Best Matching Unit (BMU).
# -------------------------------------------------------------------------
def quantization_error(som, x):
    return np.mean([np.linalg.norm(xi - som._weights[som.winner(xi)])
                    for xi in x])


# -------------------------------------------------------------------------
# Train SOMs of increasing grid sizes and measure quantization errors
# -------------------------------------------------------------------------
grid_sizes = [3, 7, 15, 25]
q_errors = []

for size in grid_sizes:
    print(f"\n--- Training SOM of size {size}x{size} ---")
    som = train_som(X_norm, y, size)
    q_err = quantization_error(som, X_norm)
    q_errors.append(q_err)
    print(f"Quantization Error for {size}x{size}: {q_err:.4f}")
    plot_umatrix(som, X_norm, y, size)


# -------------------------------------------------------------------------
# Plot Quantization Error vs SOM grid size
# -------------------------------------------------------------------------
plt.figure(figsize=(6,4))
plt.plot(grid_sizes, q_errors, marker='o')
plt.title('Quantization Error vs Grid Size')
plt.xlabel('Grid Size (NxN)')
plt.ylabel('Quantization Error')
plt.grid(True)
plt.savefig("Plots/QuantizationError_vs_GridSize.png")
plt.show()

