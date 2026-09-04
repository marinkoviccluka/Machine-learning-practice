import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

def plot_decision_regions(X, y, classifier, resolution=0.02):
    # Setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # Plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution)
    )
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # Plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            alpha=0.8,
            c=colors[idx],
            marker=markers[idx],
            label=cl
        )


# Load data
data = pd.read_csv("LV6/Social_Network_Ads.csv")
print(data.info())

data.hist()
plt.show()

# Convert DataFrame to NumPy
X = data[["Age", "EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

# Split data (80-20% ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=10
)

# Feature scaling (standardization)
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform(X_test)

# Logistic Regression Model
LogReg_model = LogisticRegression(penalty=None) 
LogReg_model.fit(X_train_n, y_train)

# Evaluation of Logistic Regression Model
y_train_p = LogReg_model.predict(X_train_n)
y_test_p = LogReg_model.predict(X_test_n)

print("-" * 30)
print("Logistic Regression: ")
print("Train Accuracy: " + "{:0.3f}".format(accuracy_score(y_train, y_train_p)))
print("Test Accuracy: " + "{:0.3f}".format(accuracy_score(y_test, y_test_p)))
print("-" * 30)

# Decision boundary using Logistic Regression
plot_decision_regions(X_train_n, y_train, classifier=LogReg_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Accuracy: " + "{:0.3f}".format(accuracy_score(y_train, y_train_p)))
plt.tight_layout()
plt.show()


# Standardized data must be used for the KNN model
KNN_model = KNeighborsClassifier(n_neighbors=5)
KNN_model.fit(X_train_n, y_train)

y_pred_train_KNN = KNN_model.predict(X_train_n)
y_pred_test_KNN = KNN_model.predict(X_test_n)

# Accuracy
acc_train_knn = accuracy_score(y_train, y_pred_train_KNN)
acc_test_knn = accuracy_score(y_test, y_pred_test_KNN)

print(f"KNN Accuracy (Train set K=5): {acc_train_knn:.3f}")
print(f"KNN Accuracy (Test set K=5): {acc_test_knn:.3f}")
print("-" * 30)

# Plotting decision boundary
plot_decision_regions(X_train_n, y_train, classifier=KNN_model)
plt.xlabel("x_1")
plt.ylabel("x_2")
plt.legend(loc="upper left")
plt.title("Accuracy: " + "{:0.3f}".format(acc_train_knn) + " K:5")
plt.tight_layout()
plt.show()


# K = 1 
KNN_model = KNeighborsClassifier(n_neighbors=1)
KNN_model.fit(X_train_n, y_train)

y_pred_train_KNN = KNN_model.predict(X_train_n)
y_pred_test_KNN = KNN_model.predict(X_test_n)

# Accuracy
acc_train_knn = accuracy_score(y_train, y_pred_train_KNN)
acc_test_knn = accuracy_score(y_test, y_pred_test_KNN)

print(f"KNN Accuracy (Train set K=1): {acc_train_knn:.3f}")
print(f"KNN Accuracy (Test set K=1): {acc_test_knn:.3f}")
print("-" * 30)

# Plotting decision boundary
plot_decision_regions(X_train_n, y_train, classifier=KNN_model)
plt.xlabel("x_1")
plt.ylabel("x_2")
plt.legend(loc="upper left")
plt.title("Accuracy: " + "{:0.3f}".format(acc_train_knn) + " K:1")
plt.tight_layout()
plt.show()


# K = 100
KNN_model = KNeighborsClassifier(n_neighbors=100)
KNN_model.fit(X_train_n, y_train)

y_pred_train_KNN = KNN_model.predict(X_train_n)
y_pred_test_KNN = KNN_model.predict(X_test_n)

# Accuracy
acc_train_knn = accuracy_score(y_train, y_pred_train_KNN)
acc_test_knn = accuracy_score(y_test, y_pred_test_KNN)

print(f"KNN Accuracy (Train set K=100): {acc_train_knn:.3f}")
print(f"KNN Accuracy (Test set K=100): {acc_test_knn:.3f}")
print("-" * 30)

# Plotting decision boundary
plot_decision_regions(X_train_n, y_train, classifier=KNN_model)
plt.xlabel("x_1")
plt.ylabel("x_2")
plt.legend(loc="upper left")
plt.title("Accuracy: " + "{:0.3f}".format(acc_train_knn) + " K:100")
plt.tight_layout()
plt.show()

# For K=1 (very small value), each point is treated as critical, resulting in overfitting.
# For K=5 to K=15, isolated noise points are suppressed and boundary islands diminish.
# For K=100, the decision boundary averages over a large volume of data, causing underfitting.

# Parameter grid setup for GridSearchCV cross-validation
knn = KNeighborsClassifier()
param_grids = {
    "n_neighbors": range(1, 31)
}

# Cross-validation with specified parameters
grid_search = GridSearchCV(estimator=knn, param_grid=param_grids, cv=5, scoring="accuracy")
grid_search.fit(X_train_n, y_train)

print(f"Best hyperparameter: {grid_search.best_params_}")
print(f"Highest cross-validation accuracy: {grid_search.best_score_:.3f}")
print("-" * 30)

# KNN - Plotting accuracy across K values
K_values = list(range(1, 31))
cv_scores_mean = []
cv_scores_std = []

for k in K_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_n, y_train, cv=5, scoring='accuracy')
    cv_scores_mean.append(scores.mean())
    cv_scores_std.append(scores.std())

optimal_k = K_values[np.argmax(cv_scores_mean)]

plt.figure(figsize=(10, 5))
plt.errorbar(K_values, cv_scores_mean, yerr=cv_scores_std, marker='o', linewidth=2)
plt.xlabel('K value')
plt.ylabel('Accuracy (5-fold CV)')
plt.title('KNN - Cross Validation')
plt.grid(True)
plt.axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal K={optimal_k}')
plt.legend()
plt.show()

# Automatically using the best model
best_knn = grid_search.best_estimator_
acc_test_best = best_knn.score(X_test_n, y_test)

# K = 7
KNN_model = KNeighborsClassifier(n_neighbors=7)
KNN_model.fit(X_train_n, y_train)

y_pred_train_KNN = KNN_model.predict(X_train_n)
y_pred_test_KNN = KNN_model.predict(X_test_n)

# Accuracy
acc_train_knn = accuracy_score(y_train, y_pred_train_KNN)
acc_test_knn = accuracy_score(y_test, y_pred_test_KNN)

print(f"KNN Accuracy (Train set K=7): {acc_train_knn:.3f}")
print(f"KNN Accuracy (Test set K=7): {acc_test_knn:.3f}")
print("-" * 30)

# Conclusion:
# Hyperparameter tuning finds the optimal K value to balance overfitting and underfitting.
# Essential for obtaining the most optimal model.

selected_C = 1.0
selected_gamma = 5.0

svm_rbf_model = SVC(kernel='rbf', C=selected_C, gamma=selected_gamma)

# Training model on standardized data
svm_rbf_model.fit(X_train_n, y_train)

# Prediction
y_pred_train_svm = svm_rbf_model.predict(X_train_n)
y_pred_test_svm = svm_rbf_model.predict(X_test_n)

acc_train_svm = accuracy_score(y_train, y_pred_train_svm)
acc_test_svm = accuracy_score(y_test, y_pred_test_svm)

print(f"SVM RBF Accuracy (Train set): {acc_train_svm:.3f}")
print(f"SVM RBF Accuracy (Test set): {acc_test_svm:.3f}")
print("-" * 30)

# Plotting decision boundary
plot_decision_regions(X_train_n, y_train, classifier=svm_rbf_model)
plt.xlabel("Standardized Age")
plt.ylabel("Standardized Salary")
plt.legend(loc="upper left")
plt.title(f"SVM RBF | C={selected_C}, gamma={selected_gamma} | Accuracy: {acc_train_svm:.3f}")
plt.tight_layout()
plt.show()

plot_decision_regions(X_test_n, y_test_p, classifier=svm_rbf_model)
plt.xlabel("Standardized Age")
plt.ylabel("Standardized Salary")
plt.legend(loc="upper left")
plt.title(f"SVM RBF | C={selected_C}")
plt.tight_layout()
plt.show()

# Gamma controls the influence radius of individual training samples. 
# Higher gamma values increase the risk of overfitting.

# C parameter controls error penalty:
# Smaller values yield smoother/more generalized boundaries (more forgiving).
# Larger values force the model to classify all points strictly (less forgiving).

# SVM: decision boundaries for different kernels
kernels = ['linear', 'rbf', 'poly', 'sigmoid']
selected_C = 1.0
selected_gamma = 5.0

plt.figure(figsize=(16, 12))

for i, kernel in enumerate(kernels, 1):
    svm_model = SVC(kernel=kernel, C=selected_C, gamma=selected_gamma)
    svm_model.fit(X_train_n, y_train)
    
    y_pred_train = svm_model.predict(X_train_n)
    acc_train = accuracy_score(y_train, y_pred_train)
    
    plt.subplot(2, 2, i)
    plot_decision_regions(X_train_n, y_train, classifier=svm_model)
    plt.title(f"SVM {kernel} | Train Acc={acc_train:.3f}")
    plt.xlabel("Standardized Age")
    plt.ylabel("Standardized Salary")
    plt.legend(loc="upper left")

plt.tight_layout()
plt.show()

# Changing kernels (poly, rbf, linear) changes how the model projects and separates data.

svm_base = SVC(kernel='rbf')

param_grid = {
    'C': [0.1, 1, 5, 10, 50, 100],
    'gamma': [1, 0.1, 0.01, 0.001]
}

# Setting up GridSearchCV 
grid_search_svm = GridSearchCV(
    estimator=svm_base, 
    param_grid=param_grid, 
    cv=5, 
    scoring='accuracy'
)

grid_search_svm.fit(X_train_n, y_train)

# Print best results
print("-" * 30)
print(f"Optimal parameters: {grid_search_svm.best_params_}")
print(f"Highest accuracy (Cross-Validation): {grid_search_svm.best_score_:.3f}")
print("-" * 30)

# Extract best model
best_svm_model = grid_search_svm.best_estimator_

# Evaluation on test set
y_pred_test_best = best_svm_model.predict(X_test_n)
acc_test_best = accuracy_score(y_test, y_pred_test_best)

print(f"Accuracy of best SVM model on test set: {acc_test_best:.3f}")

plot_decision_regions(X_train_n, y_train, classifier=best_svm_model)
plt.xlabel("Standardized Age")
plt.ylabel("Standardized Salary")
plt.legend(loc="upper left")
plt.title(f"Optimal SVM | C={grid_search_svm.best_params_['C']}, gamma={grid_search_svm.best_params_['gamma']}")
plt.tight_layout()
plt.show()