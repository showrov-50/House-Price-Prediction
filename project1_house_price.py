# ============================================================
# PROJECT 1: House Price Prediction (Linear Regression)
# Dataset: Synthetic Housing Data (Boston-style features)
# Author: Mohiuddin Showrov
# ============================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

np.random.seed(42)
n = 500

# Simulate Boston-like features
crime_rate   = np.random.exponential(3, n)
avg_rooms    = np.random.normal(6, 1, n).clip(3, 9)
age          = np.random.uniform(5, 100, n)
dist_work    = np.random.uniform(1, 12, n)
tax_rate     = np.random.uniform(180, 700, n)
pupil_teacher= np.random.normal(18, 2, n).clip(12, 22)
low_income_pct = np.random.uniform(2, 38, n)

price = (
    30
    + avg_rooms     * 5.5
    - crime_rate    * 0.2
    - age           * 0.05
    - dist_work     * 0.8
    - tax_rate      * 0.01
    - pupil_teacher * 0.4
    - low_income_pct* 0.5
    + np.random.normal(0, 3, n)
).clip(5, 60)

df = pd.DataFrame({
    'CrimeRate': crime_rate, 'AvgRooms': avg_rooms,
    'Age': age, 'DistToWork': dist_work,
    'TaxRate': tax_rate, 'PupilTeacher': pupil_teacher,
    'LowIncomePct': low_income_pct, 'HousePrice': price
})

print("=" * 55)
print("  PROJECT 1: HOUSE PRICE PREDICTION")
print("=" * 55)
print(f"Dataset Shape : {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nBasic Stats:\n{df.describe().round(2)}")
print(f"\nMissing Values: {df.isnull().sum().sum()}")

# ── EDA ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("House Price EDA", fontsize=16, fontweight='bold')

axes[0,0].hist(df['HousePrice'], bins=40, color='steelblue', edgecolor='white')
axes[0,0].set_title('Distribution of House Prices')
axes[0,0].set_xlabel('Price ($000s)')
axes[0,0].set_ylabel('Frequency')

corr = df.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[0,1], square=True)
axes[0,1].set_title('Feature Correlation Heatmap')

axes[1,0].scatter(df['AvgRooms'], df['HousePrice'], alpha=0.5, color='coral', s=15)
axes[1,0].set_title('Avg Rooms vs House Price')
axes[1,0].set_xlabel('Average Rooms')
axes[1,0].set_ylabel('House Price')

feat_corr = corr['HousePrice'].drop('HousePrice').sort_values()
feat_corr.plot(kind='barh', ax=axes[1,1], color='teal')
axes[1,1].set_title('Feature Correlation with Price')
axes[1,1].set_xlabel('Correlation Coefficient')

plt.tight_layout()
plt.savefig('/home/claude/p1_eda.png', dpi=120, bbox_inches='tight')
plt.close()
print("\n[Saved] EDA plot → p1_eda.png")

# ── Model ────────────────────────────────────────────────────
X = df.drop('HousePrice', axis=1)
y = df['HousePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression":  Ridge(alpha=1.0)
}

print("\n" + "─" * 55)
print("  MODEL EVALUATION")
print("─" * 55)

best_r2, best_name, best_pred = 0, "", None
for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"\n{name}:")
    print(f"  R² Score : {r2:.4f}")
    print(f"  MAE      : {mae:.4f}")
    print(f"  RMSE     : {rmse:.4f}")
    if r2 > best_r2:
        best_r2, best_name, best_pred = r2, name, y_pred

# ── Results Plot ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Linear Regression — Results ({best_name})", fontsize=14, fontweight='bold')

axes[0].scatter(y_test, best_pred, alpha=0.5, color='steelblue', s=20)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_title('Actual vs Predicted')
axes[0].set_xlabel('Actual Price')
axes[0].set_ylabel('Predicted Price')

residuals = y_test - best_pred
axes[1].hist(residuals, bins=30, color='coral', edgecolor='white')
axes[1].axvline(0, color='black', linestyle='--')
axes[1].set_title('Residuals Distribution')
axes[1].set_xlabel('Residual')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('/home/claude/p1_results.png', dpi=120, bbox_inches='tight')
plt.close()
print("\n[Saved] Results plot → p1_results.png")

lr = models["Linear Regression"]
lr.fit(X_train_sc, y_train)
coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': lr.coef_})
coef_df = coef_df.reindex(coef_df['Coefficient'].abs().sort_values(ascending=False).index)
print(f"\nFeature Coefficients:\n{coef_df.to_string(index=False)}")
print(f"\n✅ Project 1 Complete! Best R²: {best_r2:.4f}")
