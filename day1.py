# Importing Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# Step 1: Explore the dataset
print(df.head())
print(df.info())
print(df.isnull().sum())

# Step 2: Handle missing values
# Fill Age with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill Embarked with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin (too many missing values)
df.drop('Cabin', axis=1, inplace=True)

# Step 3: Convert categorical to numerical
# Convert 'Sex' using Label Encoding
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# One-hot encode 'Embarked'
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Step 4: Normalize numerical features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

numerical_cols = ['Age', 'Fare']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Step 5: Visualize outliers
plt.figure(figsize=(12,5))
sns.boxplot(data=df[numerical_cols])
plt.title("Boxplot to detect outliers")
plt.show()

# Optional: Remove outliers using IQR
Q1 = df[numerical_cols].quantile(0.25)
Q3 = df[numerical_cols].quantile(0.75)
IQR = Q3 - Q1

df_cleaned = df[~((df[numerical_cols] < (Q1 - 1.5 * IQR)) | (df[numerical_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Final check
print(df_cleaned.head())
print(df_cleaned.info())
