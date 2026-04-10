import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Ensure plots folder exists
# -------------------------
os.makedirs("plots", exist_ok=True)

# -------------------------
# 1. Load historical data (2019-2023)
# -------------------------
df = pd.read_csv("crop_yield_climate_soil_data_2019_2023.csv")
df.columns = df.columns.str.strip()

# Create date if needed for plots
if 'year' in df.columns and 'month' in df.columns:
    df['date'] = pd.to_datetime(
        df['year'].astype(int).astype(str) + '-' +
        df['month'].astype(int).astype(str) + '-01'
    )

# -------------------------
# 2. Select relevant features and target
# -------------------------
features = [
    'NDVI',
    'Rainfall',
    'Temperature',
    'Soil_PH',
    'Sunlight',
    'month',
    'year',
    'CNN_Feature'
]
target = 'Combined_Crop_Yield'

X = df[features]
y = df[target]

# -------------------------
# 3. Split & scale
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# -------------------------
# 4. Train Ridge Regression model
# -------------------------
model = Ridge(alpha=0.1)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
print(f"R² Score (Test Set): {r2:.5f}")

# -------------------------
# 5. Save model & scaler
# -------------------------
joblib.dump(model, "crop_yield_model.pkl")
joblib.dump(scaler, "crop_yield_scaler.pkl")
print("Model and scaler saved!")

# -------------------------
# 6. Feature importance
# -------------------------
importance = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_,
    'AbsCoefficient': np.abs(model.coef_)
}).sort_values('AbsCoefficient', ascending=False)

print("\nTop Influential Features:")
print(importance)

# -------------------------
# 7. Visualization examples (optional)
# -------------------------
plt.figure(figsize=(6,4))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Predicted vs Actual Crop Yield")
plt.savefig("plots/pred_vs_actual.png", bbox_inches="tight")
plt.show()

# You can add more plots like correlation heatmap, NDVI vs yield, rainfall vs yield, etc.