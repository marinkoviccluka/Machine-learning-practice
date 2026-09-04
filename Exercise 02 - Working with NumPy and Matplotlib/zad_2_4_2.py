import numpy as np
import matplotlib.pyplot as plt

def GetNumberOfPeople(data):
    print({len(data)})

def GetHeightAndWeightRelationship(data):
    men = (data[:, 0] == 1)
    women = (data[:, 0] == 0)
    weight_m = data[men, 2]
    height_m = data[men, 1]
    weight_w = data[women, 2]
    height_w = data[women, 1]
    plt.scatter(weight_m, height_m, alpha=0.3, s=10, c='blue', label='Men')
    plt.scatter(weight_w, height_w, alpha=0.5, s=10, c='red', label='Women')
    plt.show()

def GetHeightAndWeightForEveryFiftiethPerson(data):
    weight = data[::50, 2]
    height = data[::50, 1]
    plt.scatter(weight, height, alpha=0.5)
    plt.show()

def GetMinMaxAvgHeight(data):
    height = data[:, 1]
    print(f"Min:{np.min(height)} Max:{np.max(height)} Avg:{np.mean(height)}")

def GetMinMaxAvgHeightMF(data):
    men = (data[:, 0] == 1)
    women = (data[:, 0] == 0)
    print(f"Men: Min:{np.min(data[men, 1])} Max:{np.max(data[men, 1])} Avg:{np.mean(data[men, 1])}")
    print(f"Women: Min:{np.min(data[women, 1])} Max:{np.max(data[women, 1])} Avg:{np.mean(data[women, 1])}")


data = np.loadtxt("data.csv", delimiter=",", skiprows=1)
GetNumberOfPeople(data)
GetHeightAndWeightRelationship(data)
GetHeightAndWeightForEveryFiftiethPerson(data)
GetMinMaxAvgHeight(data)
GetMinMaxAvgHeightMF(data)