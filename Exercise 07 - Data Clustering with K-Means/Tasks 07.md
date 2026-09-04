# Tasks

## Task 7.5.1
The script `zadatak_1.py` contains the function `generate_data`, which is used to generate artificial data samples to demonstrate clustering. The function accepts an integer that defines the desired number of samples in the dataset and an integer from 1 to 5 that defines how the data will be generated, returning the generated dataset as a NumPy array where the first and second columns are the values of the first and second input features for each sample. The script generates 500 data samples and displays them as a scatter plot.
  1. Run the script. Can you recognize how many groups/clusters there are in the generated data? Change the data generation mode.
  2. Apply the K-means method and display the samples again, but color each sample according to its cluster membership. Run the program code several times. Change the number $K$. What do you notice?
  3. Change the mode of defining artificial samples and observe the clustering results (use the optimal number of groups). How do you comment on the obtained results?


## Task 7.5.2
Color quantization is the process of reducing the number of distinct colors in a digital image while ensuring that the resulting image visually resembles the original image as closely as possible. A simple approach to color quantization can be achieved by applying the K-means algorithm to the RGB values of the original image elements. Quantization is then performed by replacing the value of each element in the original image with its closest cluster center. Figure 7.3a shows an example of an original image containing a total of 106,276 colors, while Figure 7.3b shows the resulting image after quantization containing only 5 colors determined by the K-means algorithm.
  1. Open the script `zadatak_2.py`. This script loads the original RGB image `test_1.jpg` and transforms it into a dataset whose dimensions correspond to expression (7.2), where $n$ is the number of image elements and $m$ equals 3. How many distinct colors are present in this image?
  2. Apply the K-means algorithm to find clusters in the RGB values of the original image elements.
  3. Replace the value of each element of the original image with its corresponding center.
  4. Compare the resulting image with the original. Change the number of clusters $K$. Comment on the obtained results.
  5. Apply the procedure to the other available images as well.
  6. Graphically display the dependence of $J$ on the number of clusters $K$. Use the `inertia_` attribute of the `KMeans` class object. Can you spot an elbow indicating the optimal number of clusters?
  7. Display the image elements belonging to a single cluster as a separate binary image. What do you notice?