import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the cleaned dataset with scaling
df_scaled = pd.read_csv(r'D:\Capstone projects\cardekho\car_dekho_cleaned_dataset.csv')

#Distribution Plot of Car Prices
# plt.figure(figsize=(10, 6))
# sns.violinplot(x=df_scaled['price'])
# plt.title('Violin Plot of Car Prices')
# plt.xlabel('Price')
# plt.show()

# #Distribution Plot of Kilometers Driven
# plt.figure(figsize=(10, 6))
# sns.histplot(df_scaled['km'], kde=True)
# plt.title('Distribution of Kilometers Driven')
# plt.xlabel('Kilometers Driven')
# plt.ylabel('Frequency')
# plt.show()

# #Distribution Plot of Model Year
# plt.figure(figsize=(10, 6))
# sns.ecdfplot(df_scaled['modelYear'])
# plt.title('Distribution of Model Year')
# plt.xlabel('Model Year')
# plt.ylabel('Frequency')
# plt.show()

# #Distribution Plot of Mileage
# plt.figure(figsize=(10, 6))
# sns.histplot(df_scaled['mileage'], kde=True)
# plt.title('Distribution of Mileage')
# plt.xlabel('Mileage')
# plt.ylabel('Frequency')
# plt.show()

# #Distribution Plot for Seats
# plt.figure(figsize=(10, 6))
# sns.histplot(df_scaled['Seats'], kde=True)
# plt.title('Distribution of Seats')
# plt.xlabel('Seats')
# plt.ylabel('Frequency')
# plt.show()

# #Correlation Matrix for the scaled datas
# df_numeric = df_scaled.select_dtypes(include=[np.number])
# plt.figure(figsize=(12, 8))
# sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', fmt='.2f')
# plt.title('Correlation Matrix')
# plt.show()

# #PairPlot between km and price
# sns.pairplot(df_scaled[['km', 'price']])
# plt.show()

# #PairPlot between modelyear and price
# sns.pairplot(df_scaled[['modelYear', 'price']])
# plt.show()

# #Scatter Plot between mileage and price
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x='mileage', y='price', data=df_scaled)
# plt.title('Price vs Mileage')
# plt.xlabel('Mileage')
# plt.ylabel('Price')
# plt.show()

#Scatter Plot between seats and price
plt.figure(figsize=(12, 6))
sns.boxplot(x='Seats', y='price', data=df_scaled)
plt.title('Price vs Seats')
plt.xlabel('Seats')
plt.ylabel('Price')
plt.xticks(rotation=45)
plt.show()