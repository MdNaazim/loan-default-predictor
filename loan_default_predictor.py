# ============================================================
# Loan Default Predictor — Machine Learning Project
# Author: Mohammed Naazim Pasha Z
# GitHub: github.com/MdNaazim
# Tools: Python, Scikit-Learn, Pandas, NumPy, Matplotlib
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, accuracy_score
)
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("   LOAN DEFAULT PREDICTION — ML PIPELINE")
print("   Author: Mohammed Naazim Pasha Z")
print("=" * 60)

# ── STEP 1: GENERATE REALISTIC DATASET ──────────────────────
print("\n[1/6] Generating loan dataset...")

np.random.seed(42)
n = 2000

data = pd.DataFrame({
    'age':              np.random.randint(22, 65, n),
    'income':          np.random.randint(15000, 120000, n),
    'loan_amount':     np.random.randint(5000, 80000, n),
    'loan_tenure_months': np.random.choice([12, 24, 36, 48, 60], n),
    'credit_score':    np.random.randint(300, 850, n),
    'existing_loans':  np.random.randint(0, 5, n),
    'employment_type': np.random.choice(['Salaried', 'Self-Employed', 'Business'], n),
    'education':       np.random.choice(['Graduate', 'Post-Graduate', 'Under-Graduate'], n),
    'missed_payments': np.random.randint(0, 6, n),
})

# Realistic default logic
default_prob = (
    (data['credit_score'] < 500).astype(int) * 0.4 +
    (data['missed_payments'] > 2).astype(int) * 0.3 +
    (data['existing_loans'] > 3).astype(int) * 0.15 +
    (data['income'] < 25000).astype(int) * 0.1 +
    np.random.uniform(0, 0.05, n)
)
data['loan_status'] = (default_prob > 0.45).astype(int)

print(f"   Dataset created: {n} records")
print(f"   Default rate: {data['loan_status'].mean()*100:.1f}%")
print(f"   Features: {data.shape[1]-1}")

# ── STEP 2: EXPLORATORY DATA ANALYSIS ───────────────────────
print("\n[2/6] Running Exploratory Data Analysis (EDA)...")

print("\n   Dataset Overview:")
print(data.describe().round(2).to_string())
print(f"\n   Missing values: {data.isnull().sum().sum()}")
print(f"   Class distribution:\n{data['loan_status'].value_counts().to_string()}")

# ── STEP 3: PREPROCESSING ───────────────────────────────────
print("\n[3/6] Preprocessing data...")

# Encode categorical columns
le = LabelEncoder()
data['employment_type_enc'] = le.fit_transform(data['employment_type'])
data['education_enc']       = le.fit_transform(data['education'])

# Feature engineering
data['debt_to_income']    = data['loan_amount'] / data['income']
data['credit_risk_score'] = data['credit_score'] / 850
data['payment_history']   = 1 - (data['missed_payments'] / 6)

features = [
    'age', 'income', 'loan_amount', 'loan_tenure_months',
    'credit_score', 'existing_loans', 'missed_payments',
    'employment_type_enc', 'education_enc',
    'debt_to_income', 'credit_risk_score', 'payment_history'
]

X = data[features]
y = data['loan_status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"   Training samples: {X_train.shape[0]}")
print(f"   Testing samples:  {X_test.shape[0]}")
print(f"   Features used:    {len(features)}")

# ── STEP 4: TRAIN MODELS ─────────────────────────────────────
print("\n[4/6] Training ML models...")

models = {
    'Logistic Regression':    LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest':          RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting':      GradientBoostingClassifier(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    acc  = accuracy_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)
    results[name] = {'model': model, 'pred': y_pred, 'prob': y_prob,
                     'accuracy': acc, 'auc': auc}
    print(f"   {name:25s} | Accuracy: {acc*100:.1f}% | AUC: {auc:.3f}")

best_name = max(results, key=lambda k: results[k]['auc'])
best      = results[best_name]
print(f"\n   Best model: {best_name} (AUC: {best['auc']:.3f})")

# ── STEP 5: EVALUATE BEST MODEL ──────────────────────────────
print("\n[5/6] Evaluating best model...")
print(f"\n   Classification Report — {best_name}:")
print(classification_report(y_test, best['pred'],
      target_names=['No Default', 'Default']))

cm = confusion_matrix(y_test, best['pred'])
tn, fp, fn, tp = cm.ravel()
print(f"   True Positives  (correctly caught defaults): {tp}")
print(f"   False Positives (false alarms):              {fp}")
print(f"   True Negatives  (correctly approved):        {tn}")
print(f"   False Negatives (missed defaults):           {fn}")

# Feature importance
rf_model = results['Random Forest']['model']
importance_df = pd.DataFrame({
    'Feature':    features,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n   Top 5 Features Driving Default Prediction:")
for _, row in importance_df.head(5).iterrows():
    bar = '█' * int(row['Importance'] * 100)
    print(f"   {row['Feature']:25s} {bar} {row['Importance']*100:.1f}%")

# ── STEP 6: VISUALIZE RESULTS ────────────────────────────────
print("\n[6/6] Generating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Loan Default Predictor — Results Dashboard\nAuthor: Mohammed Naazim Pasha Z',
             fontsize=14, fontweight='bold', y=1.01)

# Plot 1 — Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
            xticklabels=['No Default', 'Default'],
            yticklabels=['No Default', 'Default'])
axes[0, 0].set_title('Confusion Matrix', fontweight='bold')
axes[0, 0].set_xlabel('Predicted')
axes[0, 0].set_ylabel('Actual')

# Plot 2 — ROC Curves
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['prob'])
    axes[0, 1].plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})", linewidth=2)
axes[0, 1].plot([0,1],[0,1],'k--', alpha=0.5, label='Random')
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')
axes[0, 1].set_title('ROC Curves — All Models', fontweight='bold')
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(alpha=0.3)

# Plot 3 — Feature Importance
top10 = importance_df.head(10)
axes[1, 0].barh(top10['Feature'][::-1], top10['Importance'][::-1], color='steelblue')
axes[1, 0].set_title('Top 10 Feature Importances', fontweight='bold')
axes[1, 0].set_xlabel('Importance Score')
axes[1, 0].grid(axis='x', alpha=0.3)

# Plot 4 — Model Comparison
model_names  = list(results.keys())
accuracies   = [results[m]['accuracy']*100 for m in model_names]
aucs         = [results[m]['auc']*100 for m in model_names]
x = np.arange(len(model_names))
w = 0.35
axes[1, 1].bar(x - w/2, accuracies, w, label='Accuracy %', color='steelblue')
axes[1, 1].bar(x + w/2, aucs,       w, label='AUC Score %', color='darkorange')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels([m.replace(' ', '\n') for m in model_names], fontsize=9)
axes[1, 1].set_ylim(0, 110)
axes[1, 1].set_title('Model Performance Comparison', fontweight='bold')
axes[1, 1].set_ylabel('Score (%)')
axes[1, 1].legend()
axes[1, 1].grid(axis='y', alpha=0.3)
for i, (a, u) in enumerate(zip(accuracies, aucs)):
    axes[1, 1].text(i-w/2, a+1, f'{a:.1f}%', ha='center', fontsize=8)
    axes[1, 1].text(i+w/2, u+1, f'{u:.1f}%', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('results_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: results_dashboard.png")

# ── LIVE PREDICTION DEMO ─────────────────────────────────────
print("\n" + "=" * 60)
print("   LIVE PREDICTION DEMO")
print("=" * 60)

demo_applicants = pd.DataFrame([
    {
        'age': 28, 'income': 45000, 'loan_amount': 15000,
        'loan_tenure_months': 36, 'credit_score': 720,
        'existing_loans': 1, 'missed_payments': 0,
        'employment_type_enc': 0, 'education_enc': 0,
        'debt_to_income': 15000/45000,
        'credit_risk_score': 720/850,
        'payment_history': 1.0
    },
    {
        'age': 35, 'income': 22000, 'loan_amount': 60000,
        'loan_tenure_months': 60, 'credit_score': 380,
        'existing_loans': 4, 'missed_payments': 5,
        'employment_type_enc': 2, 'education_enc': 2,
        'debt_to_income': 60000/22000,
        'credit_risk_score': 380/850,
        'payment_history': 1-(5/6)
    },
])

demo_scaled = scaler.transform(demo_applicants[features])
preds = results['Random Forest']['model'].predict(demo_scaled)
probs = results['Random Forest']['model'].predict_proba(demo_scaled)[:, 1]

labels = ['LOW RISK — Likely to repay ✓', 'HIGH RISK — Likely to default ✗']
print(f"\n   Applicant 1 (Good profile):")
print(f"   → Prediction: {labels[preds[0]]}")
print(f"   → Default probability: {probs[0]*100:.1f}%")
print(f"\n   Applicant 2 (Risky profile):")
print(f"   → Prediction: {labels[preds[1]]}")
print(f"   → Default probability: {probs[1]*100:.1f}%")

print("\n" + "=" * 60)
print("   PROJECT COMPLETE!")
print("   Files generated: results_dashboard.png")
print("=" * 60)
