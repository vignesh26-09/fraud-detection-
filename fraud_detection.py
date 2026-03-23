# ==============================
# 🚀 FRAUD DETECTION PROJECT
# ==============================

# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE

# ------------------------------

# 2. LOAD DATASET
df = pd.read_csv(r'D:\python\project_rise\credit_card_fraud_10k.csv')

print("Dataset Shape:", df.shape)
print(df.head())
print("\nColumns:", df.columns)

# ------------------------------

# 3. DATA PREPROCESSING

# Drop ID
if 'transaction_id' in df.columns:
    df = df.drop('transaction_id', axis=1)

# Scale amount
scaler = StandardScaler()
df['amount'] = scaler.fit_transform(df[['amount']])

# Convert categorical to numeric
if 'merchant_category' in df.columns:
    df = pd.get_dummies(df, columns=['merchant_category'], drop_first=True)

# ------------------------------

# 4. CHECK CLASS DISTRIBUTION

print("\nClass Distribution:")
print(df['is_fraud'].value_counts())

sns.countplot(x='is_fraud', data=df)
plt.title("Fraud vs Normal Transactions")
plt.show()

# ------------------------------

# 5. FEATURE & TARGET SPLIT

X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

# ------------------------------

# 6. TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------

# 7. HANDLE IMBALANCE USING SMOTE

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("\nBefore SMOTE:")
print(y_train.value_counts())

print("\nAfter SMOTE:")
print(pd.Series(y_train_res).value_counts())

# ------------------------------

# 8. TRAIN MODELS

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_res, y_train_res)

# Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train_res, y_train_res)

# ------------------------------

# 9. PREDICTIONS

lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)

# ------------------------------

# 10. EVALUATION FUNCTION

def evaluate_model(y_true, y_pred, model_name):
    print(f"\n===== {model_name} =====")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d')
    plt.title(f"{model_name} - Confusion Matrix")
    plt.show()

# Evaluate both models
evaluate_model(y_test, lr_pred, "Logistic Regression")
evaluate_model(y_test, rf_pred, "Random Forest")

# ------------------------------

# 11. ROC CURVE

lr_prob = lr.predict_proba(X_test)[:,1]
rf_prob = rf.predict_proba(X_test)[:,1]

lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_prob)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_prob)

plt.plot(lr_fpr, lr_tpr, label="Logistic Regression")
plt.plot(rf_fpr, rf_tpr, label="Random Forest")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# ------------------------------

# 12. FEATURE IMPORTANCE

importances = rf.feature_importances_
features = X.columns

feat_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,5))
sns.barplot(x='Importance', y='Feature', data=feat_df.head(10))
plt.title("Top 10 Important Features")
plt.show()

# ------------------------------

# 13. FRAUD RISK SCORING

df_test = X_test.copy()
df_test['Fraud_Probability'] = rf_prob

print("\nFraud Risk Scores:")
print(df_test[['Fraud_Probability']].head())

# ------------------------------

# 14. INSIGHTS

print("\n=== INSIGHTS ===")
print("1. Fraud transactions are rare compared to normal transactions.")
print("2. SMOTE improves model performance on minority class.")
print("3. Random Forest generally performs better.")
print("4. Some features strongly influence fraud detection.")
print("5. Fraud probability scoring helps in real-world risk detection.")

# ==============================
# END OF PROJECT
# ==============================