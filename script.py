import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("/Users/alexlevesque/Library/CloudStorage/OneDrive-Queen\'sUniversity/golf-analytics/golf_modeling_dataset.csv")

features = [
    'WindSpeed_mph',
    'Temperature_F',
    'Precipitation_in',
    'DrivingAccuracy_pct',
    'ShortGameRating',
    'AggressivenessScore'
]

X = df[features]
y = df['RoundScore']

df = df.dropna()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(mse)
print(r2)

for feature, coef in zip(features, model.coef_):
    print(f"{feature}: {coef:.3f}")

residuals = y_test - y_pred

plt.figure(figsize=(8, 5))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted Round Score")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Values")
plt.grid(True)
plt.show()