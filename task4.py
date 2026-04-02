import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

data = pd.read_csv(r"D:\Movies\Downloads\cleaned_student_dataset.csv")

print("First 5 rows:\n", data.head())

print("\nDataset Info:\n")
print(data.info())

print("\nStatistical Summary:\n")
print(data.describe())

print("\nMissing Values:\n")
print(data.isnull().sum())

data['Name'] = data['Name'].fillna("Unknown")

le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])
data['Course'] = le.fit_transform(data['Course'])
data['City'] = le.fit_transform(data['City'])

sns.histplot(data['Marks'], kde=True)
plt.title("Marks Distribution")
plt.show()

sns.scatterplot(x='Attendance (%)', y='Marks', data=data)
plt.title("Attendance vs Marks")
plt.show()

sns.countplot(x='Course', data=data)
plt.title("Students per Course")
plt.show()

corr = data.corr(numeric_only=True)
sns.heatmap(corr, annot=True)
plt.title("Correlation Heatmap")
plt.show()

X = data[['Attendance (%)', 'Age', 'Gender', 'Course', 'City']]
y = data['Marks']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nPredicted Marks:", y_pred[:5])
print("Actual Marks:", y_test.values[:5])

print("\nModel Error (MAE):", mean_absolute_error(y_test, y_pred))