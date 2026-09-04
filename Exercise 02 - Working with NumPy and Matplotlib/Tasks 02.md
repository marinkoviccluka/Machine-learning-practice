# Tasks

## Task 2.4.1
Using `numpy.array` and `matplotlib.pyplot`, attempt to draw the image from Figure 2.3.
Experiment with the image: change the line color, line thickness, markers, etc.

Plot Schema (Figure 2.3):
    y-axis
     4.0 +-----------------------------------+
         |                                   |
     3.5 |                                   |
         |                                   |
     3.0 |                                   |
         |                                   |
     2.5 |                                   |
         |           (2,2)       (3,2)       |
     2.0 |             *-----------*         |
         |            /            |         |
     1.5 |           /             |         |
         |          /              |         |
     1.0 |         *---------------*         |
         |       (1,1)           (3,1)       |
     0.5 |                                   |
         |                                   |
     0.0 +---------+-----+-----+-----+-------+
        0.0       1.0   2.0   3.0   4.0     x-axis

* Points: (1, 1), (2, 2), (3, 2), and (3, 1) forming a closed shape.
* Axis boundaries: [0.0, 4.0] for both x and y.


## Task 2.4.2
The file `data.csv` contains measurements of height and mass carried out on men and women.
The script loads the data into a NumPy array `data` where:
  - Column 1: Gender label (1 = male, 0 = female)
  - Column 2: Height in cm
  - Column 3: Mass in kg

Requirements:
  a) Based on the size of the `data` array, on how many people were measurements performed?
  b) Display the relationship between height and mass using `matplotlib.pyplot.scatter`.
  c) Repeat the previous task, but display measurements for every 50th person in the plot.
  d) Calculate and print to the terminal the minimum, maximum, and mean value of height in this dataset.
  e) Repeat subtask (d), but separately for men, i.e., women.
     (e.g., to filter men, create a boolean array and use it as a row index: `ind = (data[:, 0] == 1)`).



## Task 2.4.3
The script loads the image `'road.jpg'`. By manipulating the corresponding NumPy matrix, try to:
  a) Brighten the image.
  b) Display only the second quarter of the image across its width.
  c) Rotate the image 90 degrees clockwise.
  d) Mirror (flip horizontally) the image.



## Task 2.4.4
Write a program that creates an image containing four squares of black and white color (see Figure 2.4).
Requirements:
  - Use `numpy.zeros` and `numpy.ones` to create black and white arrays of dimensions 50x50 pixels.
  - Use `numpy.hstack` and `numpy.vstack` to arrange them into the final 2x2 layout.

Checkerboard Layout (Figure 2.4):
       0            50           100 (x / width)
     0 +------------+------------+
       |            |            |
       |   BLACK    |   WHITE    |
       |  (zeros)   |   (ones)   |
       |  [50x50]   |  [50x50]   |
       |            |            |
    50 +------------+------------+
       |            |            |
       |   WHITE    |   BLACK    |
       |   (ones)   |  (zeros)   |
       |  [50x50]   |  [50x50]   |
       |            |            |
   100 +------------+------------+
     (y / height)