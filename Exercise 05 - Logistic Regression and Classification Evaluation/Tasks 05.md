# Tasks

## Task 5.5.1
The script `zadatak_1.py` generates an artificial binary classification problem with two input features. The data is split into a training set and a testing set for the model.
  a) Display the training data in the $x_1-x_2$ plane using the `matplotlib` library, coloring the data points according to their class. Also display the testing data, but use a different marker for them (e.g., `'x'`). Use the `scatter` function, which accepts the data along with parameters `c` and `cmap` to define the color of each class.
  b) Build a logistic regression model using the `scikit-learn` library based on the training dataset.
  c) Find the model parameters within the attributes of the built model. Display the decision boundary of the learned model in the $x_1-x_2$ plane together with the training data. Note: the decision boundary in the $x_1-x_2$ plane is defined as the curve: $\theta_0 + \theta_1 x_1 + \theta_2 x_2 = 0$.
  d) Perform classification on the testing dataset using the built logistic regression model. Calculate and display the confusion matrix on the test data. Calculate accuracy, precision, and recall on the testing dataset.
  e) Display the testing set in the $x_1-x_2$ plane. Mark correctly classified examples in green and incorrectly classified examples in black.


## Task 5.5.2
The script `zadatak_2.py` loads the Palmer Penguins dataset. This dataset contains measurements conducted on three different penguin species ('Adelie', 'Chinstrap', 'Gentoo') across three different islands in the Palmer Station area, Antarctica. The penguin species is selected as the output variable, with classes labeled as integer values 0, 1, and 2. The input features are bill length (`'bill_length_mm'`) and flipper length in mm (`'flipper_length_mm'`). To visualize data examples and the decision boundary, the function `plot_decision_region` is available in the script.
  a) Using a bar plot, display how many samples exist for each class (penguin species) in the training dataset and the testing dataset. Use the NumPy function `unique`.
  b) Build a logistic regression model using the `scikit-learn` library based on the training dataset.
  c) Find the model parameters within the attributes of the built model. What is the difference compared to the binary classification problem from the first task?
  d) Call the `plot_decision_region` function by passing the training data and the built logistic regression model to it. How do you comment on the obtained results?
  e) Perform classification on the testing dataset using the built logistic regression model. Calculate and display the confusion matrix on the test data. Calculate accuracy. Using the `classification_report` function, calculate the values of the four main metrics on the testing dataset.
  f) Add more input features to the model. What happens to the classification results on the testing dataset?