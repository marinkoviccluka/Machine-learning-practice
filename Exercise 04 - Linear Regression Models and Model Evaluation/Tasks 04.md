# Tasks

## Task 4.5.1
The script `zadatak_1.py` loads the dataset from `data_C02_emission.csv`. It is necessary to build and evaluate a model that estimates CO2 gas emissions based on other numerical input features. Details regarding this dataset can be found in the 3rd laboratory exercise.
  a) Select the desired numerical features by specifying a list with column names. Split the data into a training set and a testing set in an 80%-20% ratio.
  b) Using the `matplotlib` library and a scatter plot, display the dependence of CO2 gas emissions on one numerical feature. In doing so, mark the data belonging to the training set in blue, and the data belonging to the testing set in red.
  c) Perform standardization of the input features of the training set. Display a histogram of the values of one input feature before and after scaling. Based on the obtained scaling parameters, transform the input features of the testing dataset.
  d) Build a linear regression model. Print the obtained model parameters to the terminal and relate them to expression 4.6.
  e) Perform estimation of the output feature based on the input features of the testing set. Using a scatter plot, display the relationship between the actual values of the output feature and the estimates obtained by the model.
  f) Perform model evaluation by calculating the values of regression metrics on the testing dataset.
  g) What happens to the values of the evaluation metrics on the test set when you change the number of input features?


## Task 4.5.2
Based on the solution to the previous task, create a model that also uses the categorical variable "Fuel Type" as an input feature. Use 1-of-K (one-hot) encoding for categorical variables. For simplicity, do not scale the input features. Comment on the obtained results. What is the maximum error in estimating CO2 gas emissions in g/km? Which vehicle model is it?