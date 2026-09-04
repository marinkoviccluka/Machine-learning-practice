# Tasks

## Task 6.5.1
The script `zadatak_1.py` loads the `Social_Network_Ads.csv` dataset. This dataset contains information about users who did or did not make a purchase for a displayed ad. User data includes gender, age, and estimated salary. A binary classification problem is considered where age and estimated salary are input features, while purchase (0 or 1) is the output feature. To visualize data examples and the decision boundary, the function `plot_decision_region` is available in the script. The data is split into a training set and a testing set in an 80%-20% ratio and standardized. A logistic regression model has been built, and its accuracy on the training and testing datasets has been calculated. It is required to:
  1. Build the KNN algorithm on the training dataset (with $K=5$). Calculate the classification accuracy on the training dataset and the testing dataset. Compare the obtained results with the results of logistic regression. What do you notice regarding the obtained decision boundary of the KNN model?
  2. What does the decision boundary look like when $K=1$ and when $K=100$?


## Task 6.5.2
Using cross-validation, determine the optimal value of the hyperparameter $K$ for the KNN algorithm on the data from Task 1.


## Task 6.5.3
Apply an SVM model using the RBF kernel function to the data from Task 1 and display the obtained decision boundary. Vary the values of hyperparameters $C$ and $\gamma$. How does changing these hyperparameters affect the decision boundary and the error on the testing dataset? Change the type of kernel being used. What do you notice?


## Task 6.5.4
Using cross-validation, determine the optimal values of hyperparameters $C$ and $\gamma$ for the SVM algorithm on the problem from Task 1.