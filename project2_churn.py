# ============================================================
# PROJECT 2: Customer Churn Prediction (Classification)
# Dataset: Synthetic Telecom Churn Data
# Author: Mohiuddin Showrov
# ============================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score)

# ── 1. Generate Realistic Dataset ────────────────────────────
np.random.seed(42)
n = 2000

tenure        = np.random.randint(1, 72, n)
monthly_bill  = np.round(np.random.uniform(20, 120, n), 2)
total_charges = np.round(tenure * monthly_bill + np.random.normal(0, 50, n), 2)
num_services  = np.random.randint(1, 8, n)
support_calls = np.random.randint(0, 10, n)
contract      = np.random.choice(['Month-to-Month', 'One Year', 'Two Year'], n, p=[0.5, 0.3, 0.2])
internet      = np.random.choice(['DSL', 'Fiber', 'No'], n)
senior        = np.random.choice([0, 1], n, p=[0.84, 0.16])

# Churn logic: short tenure + high bill + month-to-month = higher churn
churn_prob = (
    0.3 * (tenure < 12) +
    0.2 * (monthly_bill > 80) +
    0.25 * (contract == 'Month-to-Month') +
    0.15 * (support_calls > 5) +
    0.1  * (senior == 1)
)
churn_prob = np.clip(churn_prob + np.random.normal(0, 0.05, n), 0, 1)
churn = (churn_prob > 0.4).astype(int)

df = pd.DataFrame({
    'Tenure': tenure, 'MonthlyBill': monthly_bill,
    'TotalCharges': total_charges, 'NumServices': num_services,
    'SupportCalls': support_calls, 'Contract': contract,
    'InternetService': internet, 'SeniorCitizen': senior, 'Churn': churn
})

print("=" * 55)
print("  PROJECT 2: CUSTOMER CHURN PREDICTION")
print("=" * 55)
print(f"\nDataset Shape : {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nChurn Distribution:\n{df['Churn'].value_counts()}")
print(f"Churn Rate     : {df['Churn'].mean()*100:.1f}%")

# ── 2. EDA ───────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Customer Churn EDA", fontsize=16, fontweight='bold')

# Churn distribution
churn_counts = df['Churn'].value_counts()
axes[0,0].bar(['No Churn', 'Churned'], churn_counts, color=['steelblue', 'coral'])
axes[0,0].set_title('Churn Distribution')
axes[0,0].set_ylabel('Count')
for i, v in enumerate(churn_counts):
    axes[0,0].text(i, v + 10, str(v), ha='center', fontweight='bold')

# Tenure by churn
df.boxplot(column='Tenure', by='Churn', ax=axes[0,1], patch_artist=True)
axes[0,1].set_title('Tenure by Churn Status')
axes[0,1].set_xlabel('Churn (0=No, 1=Yes)')
plt.sca(axes[0,1])
plt.title('Tenure by Churn Status')

# Monthly bill by churn
df.boxplot(column='MonthlyBill', by='Churn', ax=axes[0,2], patch_artist=True)
axes[0,2].set_title('Monthly Bill by Churn')
axes[0,2].set_xlabel('Churn (0=No, 1=Yes)')
plt.sca(axes[0,2])
plt.title('Monthly Bill by Churn')

# Contract type vs churn
contract_churn = df.groupby('Contract')['Churn'].mean() * 100
axes[1,0].bar(contract_churn.index, contract_churn, color='teal')
axes[1,0].set_title('Churn Rate by Contract Type')
axes[1,0].set_ylabel('Churn Rate (%)')
axes[1,0].tick_params(axis='x', rotation=15)

# Support calls vs churn
support_churn = df.groupby('SupportCalls')['Churn'].mean() * 100
axes[1,1].plot(support_churn.index, support_churn, marker='o', color='coral')
axes[1,1].set_title('Churn Rate vs Support Calls')
axes[1,1].set_xlabel('Number of Support Calls')
axes[1,1].set_ylabel('Churn Rate (%)')

# Correlation heatmap
num_df = df.select_dtypes(include=[np.number])
sns.heatmap(num_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1,2])
axes[1,2].set_title('Correlation Heatmap')

plt.tight_layout()
plt.savefig('/home/claude/p2_eda.png', dpi=120, bbox_inches='tight')
plt.close()
print("\n[Saved] EDA plot → p2_eda.png")

# ── 3. Preprocessing ─────────────────────────────────────────
le = LabelEncoder()
df['Contract_enc']  = le.fit_transform(df['Contract'])
df['Internet_enc']  = le.fit_transform(df['InternetService'])

features = ['Tenure', 'MonthlyBill', 'TotalCharges', 'NumServices',
            'SupportCalls', 'SeniorCitizen', 'Contract_enc', 'Internet_enc']
X = df[features]
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                     random_state=42, stratify=y)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 4. Train & Evaluate Models ───────────────────────────────
models = {
    "Logistic Regression":    LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":          RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":      GradientBoostingClassifier(n_estimators=100, random_state=42),
}

print("\n" + "─" * 55)
print("  MODEL EVALUATION RESULTS")
print("─" * 55)

best_auc, best_name, best_model, best_pred = 0, "", None, None
all_results = {}

for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred  = model.predict(X_test_sc)
    y_prob  = model.predict_proba(X_test_sc)[:, 1]
    acc     = accuracy_score(y_test, y_pred)
    auc     = roc_auc_score(y_test, y_prob)
    all_results[name] = (y_prob, auc)
    print(f"\n{name}:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['No Churn','Churned']))
    if auc > best_auc:
        best_auc, best_name, best_model, best_pred = auc, name, model, y_pred

# ── 5. Visualization ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle(f"Model Performance — Best: {best_name}", fontsize=14, fontweight='bold')

# Confusion matrix
cm = confusion_matrix(y_test, best_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['No Churn','Churned'], yticklabels=['No Churn','Churned'])
axes[0].set_title(f'Confusion Matrix\n({best_name})')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

# ROC curves
for name, (y_prob, auc) in all_results.items():
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[1].plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
axes[1].plot([0,1], [0,1], 'k--')
axes[1].set_title('ROC Curves — All Models')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend(fontsize=8)

# Feature importance
rf_model = models["Random Forest"]
feat_imp = pd.Series(rf_model.feature_importances_, index=features).sort_values()
feat_imp.plot(kind='barh', ax=axes[2], color='teal')
axes[2].set_title('Feature Importance\n(Random Forest)')
axes[2].set_xlabel('Importance Score')

plt.tight_layout()
plt.savefig('/home/claude/p2_results.png', dpi=120, bbox_inches='tight')
plt.close()
print(f"\n[Saved] Results plot → p2_results.png")
print(f"\n✅ Project 2 Complete!")
print(f"   Best Model: {best_name} | ROC-AUC: {best_auc:.4f}")
