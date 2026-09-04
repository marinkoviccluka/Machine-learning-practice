# Tasks

## Task 9.4.1
The script `zadatak_1.py` loads the CIFAR-10 dataset. This dataset contains 50,000 images in the training set and 10,000 images in the testing set. The images are RGB with a resolution of 32x32. Each image is assigned one of 10 classes depending on which object is shown in the image. It is required to:
  1. Study the available code. Which layers make up the CNN network? How many parameters does the network have?
  2. Start training the network. Monitor the training process using the TensorBoard tool in the following way: Run TensorBoard in the terminal using the command `tensorboard --logdir logs` and then open the address `http://localhost:6006/` in a web browser.
  3. Study the curves showing classification accuracy and the average loss function value on the training dataset and the validation dataset. What happened during the training of the network? Record the accuracy achieved on the test dataset.


## Task 9.4.2
Modify the script from the previous task by adding dropout layers at appropriate places in the network. Before starting training, change the TensorBoard callback function so that it writes information to a new directory (e.g., `logs/cnn_dropout`). Monitor the training progress. How do you comment on the impact of dropout layers on the network's performance?


## Task 9.4.3
Add an early stopping callback function that will stop the training process after the average loss function value on the validation set does not decrease for 5 consecutive epochs.


## Task 9.4.4
What happens to the training process:
  1. if a very large or very small batch size is used?
  2. if you use a very small or very large learning rate value?
  3. if you remove certain layers from the network to get a smaller network?
  4. if you reduce the size of the training dataset by 50%?