# -------------------------------------------------------
# 🧠 TrafficTelligence: Traffic Volume Estimation using ML
# -------------------------------------------------------

# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings('ignore')

# Step 2: Load or Simulate Dataset
# Replace this with your real dataset using: pd.read_csv("your_dataset.csv")
np.random.seed(42)
n = 1000  # number of rows

df = pd.DataFrame({
    'hour': np.random.randint(0, 24, n),
    'day_of_week': np.random.randint(0, 7, n),  # 0 = Monday, ..., 6 = Sunday
    'temperature': np.random.uniform(15, 35, n),  # in Celsius
    'rain_mm': np.random.uniform(0, 10, n),  # rainfall in millimeters
    'vehicle_count': np.random.randint(100, 1500, n)  # target variable
})

# Preview data
print("Sample Data:\n", df.head())

# Step 3: Feature Selection and Data Splitting
features = ['hour', 'day_of_week', 'temperature', 'rain_mm']
target = 'vehicle_count'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("\nData split complete.")
print("Training Set:", X_train.shape)
print("Testing Set:", X_test.shape)

# Step 4: Model Training using Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Step 5: Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Evaluation:")
print(f"🔹 MAE  : {mae:.2f}")
print(f"🔹 RMSE : {rmse:.2f}")
print(f"🔹 R²    : {r2:.2f}")

# Step 6: Visualization – Actual vs Predicted
plt.figure(figsize=(10, 5))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color="teal")
plt.xlabel("Actual Vehicle Count")
plt.ylabel("Predicted Vehicle Count")
plt.title("Actual vs Predicted Traffic Volume")
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 7: Feature Importance Plot
importances = pd.Series(model.feature_importances_, index=features)
plt.figure(figsize=(8, 4))
importances.sort_values().plot(kind='barh', color='orange')
plt.title("Feature Importance in Traffic Prediction")
plt.xlabel("Importance Score")
plt.grid(True)
plt.tight_layout()
plt.show()
