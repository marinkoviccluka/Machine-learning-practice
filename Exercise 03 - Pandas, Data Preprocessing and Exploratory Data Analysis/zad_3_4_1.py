import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("LV3/data_C02_emission.csv")

"""
A) How many measurements does the DataFrame contain? What is the data type of each feature? 
Are there missing or duplicate values? Drop them if they exist. 
Convert categorical features to type category.
"""

print('A')
print("\nNumber of measurements: ", len(data)) 
print("\nData types: ", data.dtypes) 
print("\nNumber of missing values:\n", data.isnull().sum()) 
data = data.dropna() 
print("Number of duplicate values: ", data.duplicated().sum())
data = data.drop_duplicates() 

data = data.reset_index(drop=True)  # Resetting index in case of dropped rows

data["Make"] = data["Make"].astype("category")
data["Model"] = data["Model"].astype("category")
data["Vehicle Class"] = data["Vehicle Class"].astype("category")
data["Transmission"] = data["Transmission"].astype("category")
data["Fuel Type"] = data["Fuel Type"].astype("category")

print(data.info())

"""
B) Which three cars have the highest and lowest city fuel consumption? 
Print to the terminal: manufacturer name, vehicle model, and city fuel consumption.
"""
print('B')
largest = data.sort_values(by="Fuel Consumption City (L/100km)", ascending=False).head(3)
smallest = data.sort_values(by="Fuel Consumption City (L/100km)", ascending=False).tail(3)
print(largest[["Make", "Model", "Fuel Consumption City (L/100km)"]])
print(smallest[["Make", "Model", "Fuel Consumption City (L/100km)"]])

"""
C) How many vehicles have an engine size between 2.5 and 3.5 L? 
What is the average CO2 emission for these vehicles?
"""
print('C')
engine = data[(data['Engine Size (L)'] >= 2.5) & (data['Engine Size (L)'] <= 3.5)]
print('Number of vehicles with engine size between 2.5 and 3.5 L: ', len(engine))
print('Average CO2 emission: ', round(engine['CO2 Emissions (g/km)'].mean(), 2))

"""
D) How many measurements refer to vehicles made by Audi? 
What is the average CO2 emission for 4-cylinder Audi cars?
"""
print('D')
audi = data[data['Make'] == 'Audi']
print('Audi measurements: ', len(audi))
audi_cylinders = audi[audi['Cylinders'] == 4]
print('Average CO2 emission for Audi: ', round(audi_cylinders['CO2 Emissions (g/km)'].mean(), 2))

"""
E) How many vehicles have 4, 6, 8... cylinders? 
What is the average CO2 emission with respect to the number of cylinders?
"""
print('E')
print(data.groupby('Cylinders').size())
print(round(data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean(), 2))

"""
F) What is the average city fuel consumption for vehicles that use diesel, and for vehicles 
that use regular gasoline? What are the median values?
"""
print('F')
print('Diesel')
print(round(data[data["Fuel Type"] == "D"]["Fuel Consumption City (L/100km)"].mean(), 2))
print(round(data[data["Fuel Type"] == "D"]["Fuel Consumption City (L/100km)"].median(), 2))
print('Gasoline')
print(round(data[data["Fuel Type"] == "X"]["Fuel Consumption City (L/100km)"].mean(), 2))
print(round(data[data["Fuel Type"] == "X"]["Fuel Consumption City (L/100km)"].median(), 2))

"""
G) Which 4-cylinder diesel vehicle has the highest city fuel consumption?
"""
print('G')
max_diesel_car = (data[(data["Fuel Type"] == "D") & (data["Cylinders"] == 4)]).sort_values(
    by="Fuel Consumption City (L/100km)",
    ascending=False
).head(1)
print(max_diesel_car[["Make", "Model", "Fuel Consumption City (L/100km)"]])

"""
H) How many vehicles have a manual transmission (regardless of number of gears)?
"""
print('H')
manual = data[data["Transmission"].str.startswith("M")]
print('Number of cars with manual transmission: ', len(manual))

"""
I) Calculate the correlation between numerical features. Comment on the obtained result.
"""
print('I')
corr = round(data.corr(numeric_only=True), 2)
print(corr)