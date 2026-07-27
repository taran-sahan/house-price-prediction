import pandas as pd

df = pd.read_csv("housing.csv") 
print(df.shape)
print(df.head())
print(df.columns)

print(df.describe())
print(df.isnull().sum())
print(df["ocean_proximity"].value_counts())

import matplotlib.pyplot as plt

df["median_house_value"].hist(bins=50)
plt.xlabel("Median House Value ($)")
plt.ylabel("Count")
plt.title("Distribution of House Prices")
plt.show()

import seaborn as sns

numeric_df = df.select_dtypes(include="number")
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

plt.scatter(df["median_income"], df["median_house_value"], alpha=0.2)
plt.xlabel("Median Income")
plt.ylabel("Median House Value ($)")
plt.title("Price vs Median Income")
plt.show()

df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())
print(df.isnull().sum())
df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)
print(df.columns)

df["rooms_per_household"] = df["total_rooms"] / df["households"]
df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
df["population_per_household"] = df["population"] / df["households"]
print(df.columns)
print(df[["rooms_per_household", "bedrooms_per_room", "population_per_household"]].head())

X = df.drop(columns=["median_house_value"])
y = df["median_house_value"]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(X_train_scaled[:2])

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train_scaled, y_train)

import pandas as pd

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})
print(coefficients.sort_values(by="Coefficient", key=abs, ascending=False))

y_pred = model.predict(X_test_scaled)
 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)
print("R2 Score:", r2)