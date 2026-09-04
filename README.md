# Machine learning exercises

Weekly exercises from a machine learning course, covering Python basics through convolutional neural networks. Each folder holds the task sheet and the solution scripts for that week.

## Contents

- Exercise 01: Introduction to Python
- Exercise 02: NumPy and Matplotlib
- Exercise 03: Pandas, data preprocessing, exploratory data analysis
- Exercise 04: Linear regression and model evaluation
- Exercise 05: Logistic regression and classification evaluation
- Exercise 06: KNN and support vector machines
- Exercise 07: K-means clustering
- Exercise 08: Artificial neural networks
- Exercise 09: Convolutional neural networks

## What's in each exercise

**Exercise 01** covers basic Python: functions, input validation, loops, and file handling. Includes two text-processing tasks, one that counts word frequency in a song lyric file and one that parses the SMS Spam Collection dataset to compare word counts and punctuation between spam and ham messages.

**Exercise 02** works with a body measurement dataset (`data.csv`, gender/height/weight) using NumPy for array operations and Matplotlib for plotting, plus basic image manipulation on `road.jpg`.

**Exercise 03** uses `dataC02emission.csv` for data cleaning and exploratory analysis with pandas: filtering, grouping, and summary statistics on vehicle CO2 emissions data.

**Exercise 04** builds linear regression models on the same CO2 emissions dataset and evaluates them with standard regression metrics.

**Exercise 05** applies logistic regression to the Palmer Penguins dataset (`penguins.csv`) for species classification, with accuracy, precision, and recall as evaluation metrics.

**Exercise 06** compares KNN and SVM classifiers on `SocialNetworkAds.csv`, a binary purchase-prediction dataset. Includes decision boundary plots, cross-validation for hyperparameter tuning (K for KNN, C and gamma for SVM with RBF kernel), and a comparison against logistic regression.

**Exercise 07** covers K-means clustering, first on synthetic 2D data (blobs, circles, moons) to see how cluster shape affects results, then on real images (`imgs/test1.jpg` through `test6.jpg`) for color quantization, reducing an image's color palette by clustering RGB values.

**Exercise 08** builds fully connected neural networks with Keras on MNIST digit classification, including a script that tests the trained model against hand-drawn digit images (`test1.png` through `test5.png`).

**Exercise 09** builds convolutional neural networks with Keras on CIFAR-10, monitored through TensorBoard. Covers the effect of dropout layers, early stopping, and how batch size, learning rate, and network depth affect training.

## Structure

```text
Exercise 01 - Introduction to Python/
├── Tasks 01.md
├── zad141.py ... zad145.py
├── example.txt
├── song.txt
└── SMSSpamCollection.txt

Exercise 02 - Working with NumPy and Matplotlib/
├── Tasks 02.md
├── zad241.py ... zad244.py
├── data.csv
└── road.jpg

Exercise 03 - Pandas, Data Preprocessing and Exploratory Data Analysis/
├── Tasks 03.md
├── zad341.py, zad342.py
└── dataC02emission.csv

Exercise 04 - Linear Regression Models and Model Evaluation/
├── Tasks 04.md
├── zad441.py, zad442.py
└── dataC02emission.csv

Exercise 05 - Logistic Regression and Classification Evaluation/
├── Tasks 05.md
├── zad541.py, zad542.py
└── penguins.csv

Exercise 06 - KNN and Support Vector Machines/
├── Tasks 06.md
├── zad641.py
└── SocialNetworkAds.csv

Exercise 07 - Data Clustering with K-Means/
├── Tasks 07.md
├── zad741.py, zad742.py
└── imgs/test1.jpg ... test6.jpg

Exercise 08 - Artificial Neural Networks/
├── Tasks 08.md
├── zad841.py, zad842.py, zad843.py
└── test1.png ... test5.png

Exercise 09 - Convolutional Neural Networks/
├── Tasks 09.md
└── zad941.py ... zad944.py
```

Each `Tasks 0X.md` file has the assignment questions as given in class. The `zadXXX.py` scripts (zadatak is Croatian for "task" or "exercise") are the solutions, numbered to match.

## Requirements

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow scipy
```

TensorFlow is only needed from Exercise 08 onward. For Exercise 09, you also need TensorBoard, which ships with TensorFlow.

## Running the scripts

Each script expects to run from inside its own exercise folder, since paths to CSV, image, and text files are relative.

```bash
cd "Exercise 06 - KNN and Support Vector Machines"
python zad641.py
```

For the CNN exercise, start TensorBoard separately after launching training:

```bash
tensorboard --logdir logs
```

Then open `http://localhost:6006` to watch accuracy and loss curves update.

## Notes

A few scripts reference file paths that only make sense on the original machine, such as a hardcoded folder in `zad843.py`. Fix these paths before running that script on a different setup.

The `SMSSpamCollection.txt` dataset is the classic UCI SMS Spam Collection, 5,574 messages labeled spam or ham. `song.txt` in Exercise 01 is Bill Withers' "Ain't No Sunshine," used only as sample text for word counting, not for any musical analysis.
