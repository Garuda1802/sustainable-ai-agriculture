import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Simulated dataset with environmental and soil parameters
data = {
    'soil_moisture': np.random.uniform(10, 50, 100),
    'soil_nitrogen': np.random.uniform(1, 10, 100),
    'temperature': np.random.uniform(20, 35, 100),
    'humidity': np.random.uniform(40, 90, 100),
    'rainfall_mm': np.random.uniform(50, 300, 100),
    'ndvi_index': np.random.uniform(0.3, 0.9, 100),  # Simulated satellite vegetation index
    'crop_yield': np.random.uniform(1.5, 5.5, 100)   # Target variable (tons/ha)
}

df = pd.DataFrame(data)

# Features and target
X = df.drop('crop_yield', axis=1)
y = df['crop_yield']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Root Mean Square Error: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Visualization
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Yield (tons/ha)")
plt.ylabel("Predicted Yield")
plt.title("Actual vs Predicted Crop Yield")
plt.grid(True)
plt.show()
