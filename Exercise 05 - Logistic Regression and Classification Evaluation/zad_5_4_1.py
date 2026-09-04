import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report, precision_score, recall_score


X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=213,
    n_clusters_per_class=1,
    class_sep=1
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)


# A: Visualize training and testing sets
plt.scatter(X_train[:, 0], X_train[:, 1], cmap="coolwarm", label="Train data", alpha=0.7, c=y_train)
plt.scatter(X_test[:, 0], X_test[:, 1], cmap="coolwarm", marker="*", label="Test data", alpha=0.7, c=y_test)
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.show()


# B: Train Logistic Regression model and predict
LogRegression_model = LogisticRegression()
LogRegression_model.fit(X_train, y_train)
y_test_p = LogRegression_model.predict(X_test)


# C: Plot decision boundary
coef = LogRegression_model.coef_[0]
intercept = LogRegression_model.intercept_


def decision_boundary(x1):
    return (-intercept - coef[0] * x1) / coef[1]


plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm')
plt.plot(X_train[:, 0], decision_boundary(X_train[:, 0]))
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Logistic Regression Decision Boundary')
plt.show()


# D: Evaluation metrics and confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_test_p))
disp.plot()
plt.show()

print(" Accuracy : ", accuracy_score(y_test, y_test_p))
print(" Precision : ", precision_score(y_test, y_test_p))
print(" Recall : ", recall_score(y_test, y_test_p))
print(classification_report(y_test, y_test_p))


# E: Highlight correct (green) and incorrect (red) predictions on the test set
plt.scatter(X_test[:, 0], X_test[:, 1], label="test", cmap="seismic", c=y_test)
for i in range(len(y_test)):
    if y_test[i] == y_test_p[i]:
        plt.scatter(X_test[i, 0], X_test[i, 1], c='g')
    else:
        plt.scatter(X_test[i, 0], X_test[i, 1], c='r')

plt.xlabel("x1")
plt.ylabel("x2")
plt.show()