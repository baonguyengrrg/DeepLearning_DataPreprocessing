import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

data_main = {
    'CustomerID': [1, 2, 3, 4, 5],
    'Age': [25, 400, 22, 25, 45],
    'Income': [50000, 60000, -3, 50000, 80000]
}
df = pd.DataFrame(data_main)

print("Original Data")
print(df, "\n")

#Cleaning
df['Age'] = np.where((df['Age'] < 0) | (df['Age'] > 100), np.nan, df['Age'])
df['Income'] = np.where(df['Income'] < 0, np.nan, df['Income'])
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Income'] = df['Income'].fillna(df['Income'].median())

print("After cleaning")
print(df, "\n")

#Integration
data_sales = {
    'CustomerID': [1, 2, 3, 4, 5],
    'Country': ['VN', 'US', 'JP', 'VN', 'JP'],
    'Total_Spent': [100, 250, 300, 150, 400]
}
df_sales = pd.DataFrame(data_sales)
df = pd.merge(df, df_sales, on='CustomerID', how='inner')

print("After integration (merge with sales data):")
print(df, "\n")

#Discretization
bins = [0, 18, 35, 55, 120]
labels = ['Child', 'Young Adult', 'Middle-Aged', 'Senior']
df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels)

print("After discretization (Age -> Age_Group):")
print(df[['Age', 'Age_Group']], "\n")

#Transformation
df = pd.get_dummies(df, columns=['Country', 'Age_Group'], dtype=int)
scaler = MinMaxScaler()
df[['Income', 'Total_Spent']] = scaler.fit_transform(df[['Income', 'Total_Spent']])

print("After transformation (One-hot encoding và Min-Max Scaling):")
print(df, "\n")

#Reduction
features = df.drop(columns=['CustomerID', 'Age'])
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(features)
df_final = pd.DataFrame(reduced_data, columns=['PC1', 'PC2'])
df_final['CustomerID'] = df['CustomerID']

print("After PCA reduction")
print(df_final)
