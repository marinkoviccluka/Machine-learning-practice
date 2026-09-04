import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


labels = {0: 'Adelie', 1: 'Chinstrap', 2: 'Gentoo'}

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
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
            edgecolor='w',
            label=labels[cl]
        )

df = pd.read_csv(r"osu_lv\LV5\penguins.csv")

print(df.isnull().sum())

df = df.drop(columns=['sex'])
df.dropna(axis=0, inplace=True)

# Clean strings
df['species'] = df['species'].str.strip()

# Mapping
mapping = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}
df['species'] = df['species'].map(mapping)

# If something isn't mapped -> becomes NaN -> drop those rows
df.dropna(subset=['species'], inplace=True)

# Now safely convert to int
df['species'] = df['species'].astype(int)
print(df.info())

output_variable = ['species']
input_variables = ['bill_length_mm', 'flipper_length_mm']

X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy()[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Find all unique class values and count their occurrences
classes, counts_train = np.unique(y_train, return_counts=True)
classes, counts_test = np.unique(y_test, return_counts=True)

X_axis = np.arange(len(classes))
plt.bar(X_axis - 0.2, counts_train, 0.4, label='Train')
plt.bar(X_axis + 0.2, counts_test, 0.4, label='Test') 
plt.xticks(X_axis, ['Adelie(0)', 'Chinstrap(1)', 'Gentoo(2)'])
plt.xlabel("Penguins")
plt.ylabel("Counts")
plt.title("Number of each class of penguins, train and test data")
plt.legend()
plt.show()

logRegModel = LogisticRegression(max_iter=120)
logRegModel.fit(X_train, y_train)

# Due to 3 classes, there are 3 intercepts (one per class, multinomial / 1xk dimensions); binary classification had 1 element
theta0 = logRegModel.intercept_
coefs = logRegModel.coef_ 
print('Theta0:')
print(theta0)
# Due to 3 classes, there are 3 rows of parameters (one per class) and 2 columns (one per input feature) (k x m dimensions)
print('Model parameters:')
print(coefs)  # Binary classification had 1 row with 2 columns (1 binary classifier, 2 input features)

plot_decision_regions(X_train, y_train, logRegModel)
plt.show()

y_test_p = logRegModel.predict(X_test)
cm = ConfusionMatrixDisplay(confusion_matrix(y_test, y_test_p))
cm.plot()
plt.show()

print(f"Accuracy: {accuracy_score(y_test, y_test_p)}")
print(classification_report(y_test, y_test_p))

# Adding the parameter body_mass decreases model performance; adding bill_depth increases it, as does adding both parameters.