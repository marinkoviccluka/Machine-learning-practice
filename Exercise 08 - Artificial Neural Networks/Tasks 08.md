# Tasks

## Task 8.4.1
The MNIST dataset for building a handwritten digit classifier is available within Keras. The script `zadatak_1.py` loads the MNIST dataset and prepares the data for training a fully connected neural network.
  1. Familiarize yourself with the loaded data. How many samples does the training set contain, and how many does the test set contain? How are the input data (i.e., images) scaled? How is the output variable encoded?
  2. Using the `matplotlib` library, display one image from the training dataset and print its label to the terminal.
  3. Using the `Sequential` class, build the network shown in Figure 8.5. Using the `summary` method, print information about the network to the terminal.
  4. Using the `.compile` method, configure the network training process.
  5. Start training the network (define the number of epochs and batch size independently). Monitor the training progress in the terminal.
  6. Perform evaluation of the network on the test dataset using the `evaluate` method.
  7. Calculate the network predictions for the test dataset. Using the `scikit-learn` library, display the confusion matrix for the test dataset.
  8. Save the model to the hard disk.


## Task 8.4.2
Write a script that will load the trained network from Task 1 and the MNIST dataset. Using the `matplotlib` library, display several misclassified images from the test dataset. In the title of each image, write the true label and the label predicted by the network.


## Task 8.4.3
Write a script that will load the trained network from Task 1. Furthermore, the script should load the image `test.png` from disk. Add code to the script that will adapt the image for the network, classify the image using the built network, and print the result to the terminal. Modify the image using a graphics tool (e.g., draw the number 2 using Windows Paint) and run the script again. Comment on the obtained results for different handwritten digits.