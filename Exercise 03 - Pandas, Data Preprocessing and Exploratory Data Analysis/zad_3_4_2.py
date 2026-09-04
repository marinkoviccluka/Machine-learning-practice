import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("LV3/data_C02_emission.csv")

fig, axs = plt.subplots(3, 2, figsize=(12, 12))

"""
Use a histogram to show CO2 emissions. Comment on the resulting visualization.
"""
data["CO2 Emissions (g/km)"].plot(kind="hist", ax=axs[0, 0], bins=30)
axs[0, 0].set_xlabel("CO2 Emissions")
axs[0, 0].set_ylabel("Number of vehicles")

# The vast majority of vehicles fall into the CO2 emission range between 200 and 300 g/km

"""
Use a scatter plot to show the relationship between city fuel consumption and CO2 emissions. 
Comment on the resulting visualization. To better understand the relationships between variables, 
color the points in the scatter plot according to fuel type.
"""
colors = {"X": "blue", "Z": "red", "D": "green", "E": "yellow", "N": "black"}
data.plot.scatter(
    x="Fuel Consumption City (L/100km)", 
    y="CO2 Emissions (g/km)",
    c=data["Fuel Type"].map(colors),
    ax=axs[0, 1]
)

# Visible linear correlation - higher fuel consumption leads to higher CO2 emissions
# X - regular gasoline - linear relationship - medium consumption
# Z - premium gasoline - higher city consumption suggests high-performance/expensive cars - the last outlier point is likely a supercar...
# D - diesel - although diesels consume fewer liters, they emit more CO2 per liter compared to gasoline (X)
# E - ethanol - high fuel consumption volume, but notably lower emissions compared to others

"""
Use a box plot to display the distribution of highway fuel consumption with respect to fuel type. 
Do you notice a gross measurement error / major outlier in the data?
"""
data.boxplot(
    column="Fuel Consumption Hwy (L/100km)",
    by="Fuel Type",
    ax=axs[1, 0]
)

# A prominent outlier is visible for fuel type 'Z' where one point deviates drastically from the rest; 
# however, given it is premium fuel, it could be a vehicle like a Bugatti consuming ~30.3 L/100km.

"""
Use a bar chart to show the number of vehicles by fuel type. 
Use the groupby method.
"""
data.groupby("Fuel Type").size().plot(kind="bar", ax=axs[1, 1])

"""
Use a bar chart on the same figure to show the average CO2 emissions 
of vehicles with respect to the number of cylinders.
"""
data.groupby("Cylinders")["CO2 Emissions (g/km)"].mean().plot(kind="bar", ax=axs[2, 0])
axs[2, 0].set_ylabel("CO2 emissions")

plt.tight_layout()
plt.show()