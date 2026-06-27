# ============================================================
# PROJECT 3: Titanic Dataset — Exploratory Data Analysis (EDA)
# Dataset: Built-in Seaborn Titanic
# Author: Mohiuddin Showrov
# ============================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# ── 1. Load Dataset ──────────────────────────────────────────
df = sns.load_dataset('titanic')

print("=" * 55)
print("  PROJECT 3: TITANIC EDA & SURVIVAL ANALYSIS")
print("=" * 55)
print(f"\nDataset Shape  : {df.shape}")
print(f"Columns        : {list(df.columns)}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nSurvival Rate  : {df['survived'].mean()*100:.1f}%")

# ── 2. Data Cleaning ─────────────────────────────────────────
df['age'].fillna(df['age'].median(), inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
df.drop(columns=['deck', 'embark_town', 'alive', 'who', 'adult_male',
                 'class', 'alone'], inplace=True)

print(f"\nAfter Cleaning — Missing Values:\n{df.isnull().sum()}")

# ── 3. Feature Engineering ────────────────────────────────────
df['family_size'] = df['sibsp'] + df['parch'] + 1
df['is_alone']    = (df['family_size'] == 1).astype(int)
df['age_group']   = pd.cut(df['age'],
                            bins=[0, 12, 18, 35, 60, 100],
                            labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])

# ── 4. EDA — Plot 1: Overview ────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Titanic EDA — Overview", fontsize=16, fontweight='bold')

# Survival count
survival_counts = df['survived'].value_counts()
axes[0,0].bar(['Died', 'Survived'], survival_counts, color=['#E74C3C', '#27AE60'])
axes[0,0].set_title('Overall Survival')
axes[0,0].set_ylabel('Count')
for i, v in enumerate(survival_counts):
    axes[0,0].text(i, v + 5, f'{v}\n({v/len(df)*100:.1f}%)', ha='center', fontweight='bold')

# Survival by gender
gender_surv = df.groupby('sex')['survived'].mean() * 100
axes[0,1].bar(gender_surv.index, gender_surv, color=['#3498DB', '#E91E8C'])
axes[0,1].set_title('Survival Rate by Gender')
axes[0,1].set_ylabel('Survival Rate (%)')
for i, (idx, v) in enumerate(gender_surv.items()):
    axes[0,1].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')

# Survival by class
class_surv = df.groupby('pclass')['survived'].mean() * 100
axes[0,2].bar(['1st Class', '2nd Class', '3rd Class'], class_surv,
              color=['gold', 'silver', '#CD7F32'])
axes[0,2].set_title('Survival Rate by Passenger Class')
axes[0,2].set_ylabel('Survival Rate (%)')
for i, v in enumerate(class_surv):
    axes[0,2].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')

# Age distribution
axes[1,0].hist(df[df['survived']==1]['age'], bins=30, alpha=0.6, color='#27AE60', label='Survived')
axes[1,0].hist(df[df['survived']==0]['age'], bins=30, alpha=0.6, color='#E74C3C', label='Died')
axes[1,0].set_title('Age Distribution by Survival')
axes[1,0].set_xlabel('Age')
axes[1,0].set_ylabel('Count')
axes[1,0].legend()

# Fare distribution
axes[1,1].hist(df[df['survived']==1]['fare'], bins=40, alpha=0.6, color='#27AE60', label='Survived')
axes[1,1].hist(df[df['survived']==0]['fare'], bins=40, alpha=0.6, color='#E74C3C', label='Died')
axes[1,1].set_title('Fare Distribution by Survival')
axes[1,1].set_xlabel('Fare (£)')
axes[1,1].set_ylabel('Count')
axes[1,1].legend()

# Embark port vs survival
port_surv = df.groupby('embarked')['survived'].mean() * 100
axes[1,2].bar(['Cherbourg (C)', 'Queenstown (Q)', 'Southampton (S)'],
              port_surv, color='teal')
axes[1,2].set_title('Survival Rate by Embarkation Port')
axes[1,2].set_ylabel('Survival Rate (%)')

plt.tight_layout()
plt.savefig('/home/claude/p3_eda1.png', dpi=120, bbox_inches='tight')
plt.close()
print("\n[Saved] EDA overview → p3_eda1.png")

# ── 5. EDA — Plot 2: Deep Dive ───────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Titanic EDA — Deep Dive Analysis", fontsize=16, fontweight='bold')

# Heatmap: Gender × Class survival
pivot = df.pivot_table(values='survived', index='sex', columns='pclass', aggfunc='mean') * 100
sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', ax=axes[0,0], vmin=0, vmax=100)
axes[0,0].set_title('Survival Rate (%) by Gender & Class')

# Age group survival
age_surv = df.groupby('age_group', observed=True)['survived'].mean() * 100
axes[0,1].bar(age_surv.index, age_surv,
              color=['#3498DB','#9B59B6','#E67E22','#E74C3C','#95A5A6'])
axes[0,1].set_title('Survival Rate by Age Group')
axes[0,1].set_ylabel('Survival Rate (%)')
axes[0,1].tick_params(axis='x', rotation=20)

# Family size vs survival
fam_surv = df.groupby('family_size')['survived'].mean() * 100
axes[0,2].bar(fam_surv.index, fam_surv, color='steelblue')
axes[0,2].set_title('Survival Rate by Family Size')
axes[0,2].set_xlabel('Family Size')
axes[0,2].set_ylabel('Survival Rate (%)')

# Fare boxplot by class
df_box = df[['pclass', 'fare']].copy()
survived_groups = [df[df['pclass']==c]['fare'].dropna() for c in [1,2,3]]
axes[1,0].boxplot(survived_groups, labels=['1st', '2nd', '3rd'], patch_artist=True,
                  boxprops=dict(facecolor='lightblue'))
axes[1,0].set_title('Fare Distribution by Class')
axes[1,0].set_xlabel('Passenger Class')
axes[1,0].set_ylabel('Fare (£)')

# Siblings/Spouses vs survival
sibsp_surv = df.groupby('sibsp')['survived'].mean() * 100
axes[1,1].bar(sibsp_surv.index, sibsp_surv, color='coral')
axes[1,1].set_title('Survival Rate by Siblings/Spouses')
axes[1,1].set_xlabel('# Siblings/Spouses')
axes[1,1].set_ylabel('Survival Rate (%)')

# Correlation heatmap (numerical)
num_df = df.select_dtypes(include=[np.number])
sns.heatmap(num_df.corr(), annot=True, fmt='.2f', cmap='coolwarm',
            ax=axes[1,2], square=True)
axes[1,2].set_title('Numerical Feature Correlations')

plt.tight_layout()
plt.savefig('/home/claude/p3_eda2.png', dpi=120, bbox_inches='tight')
plt.close()
print("[Saved] Deep dive EDA → p3_eda2.png")

# ── 6. Bonus: Quick Prediction Model ─────────────────────────
print("\n" + "─" * 55)
print("  BONUS: Survival Prediction Model")
print("─" * 55)

df_model = df.copy()
le = LabelEncoder()
df_model['sex_enc']      = le.fit_transform(df_model['sex'])
df_model['embarked_enc'] = le.fit_transform(df_model['embarked'].fillna('S'))

feats = ['pclass', 'sex_enc', 'age', 'sibsp', 'parch', 'fare',
         'embarked_enc', 'family_size', 'is_alone']
X = df_model[feats].fillna(0)
y = df_model['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
acc    = accuracy_score(y_test, y_pred)

print(f"\nRandom Forest Accuracy: {acc:.4f} ({acc*100:.1f}%)")
print(classification_report(y_test, y_pred, target_names=['Died','Survived']))

feat_imp = pd.Series(rf.feature_importances_, index=feats).sort_values(ascending=False)
print(f"\nTop Features by Importance:\n{feat_imp.round(4).to_string()}")

# Feature importance plot
fig, ax = plt.subplots(figsize=(9, 5))
feat_imp.sort_values().plot(kind='barh', ax=ax, color='steelblue')
ax.set_title('Feature Importance for Survival Prediction', fontweight='bold')
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig('/home/claude/p3_feat_imp.png', dpi=120, bbox_inches='tight')
plt.close()
print("\n[Saved] Feature importance → p3_feat_imp.png")

print("\n✅ Project 3 Complete!")
print("\n📌 KEY INSIGHTS:")
print("   • Women had ~74% survival vs ~19% for men")
print("   • 1st class passengers survived at 63% vs 24% in 3rd class")
print("   • Children (<12) had the highest survival among age groups")
print("   • Traveling alone or in large families reduced survival odds")
print("   • Higher fare strongly correlated with survival")
