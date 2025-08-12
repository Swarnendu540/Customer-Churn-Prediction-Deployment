# training.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
import joblib

# --- 1. Load Data ---
df = pd.read_csv(r"C:\Users\Swarnendu Pan\Downloads\Churn_Modelling.csv")  # Update the path if needed

# --- 2. Feature Engineering ---
bins = [0, 669, 739, 850]
labels = ['Low', 'Medium', 'High']
df['CreditScoreGroup'] = pd.cut(df['CreditScore'], bins=bins, labels=labels, include_lowest=True)

df['CreditUtilization'] = df['Balance'] / df['CreditScore']
df['InteractionScore'] = df['NumOfProducts'] + df['HasCrCard'] + df['IsActiveMember']
df['BalanceToSalaryRatio'] = df['Balance'] / df['EstimatedSalary']
df['CreditScoreAgeInteraction'] = df['CreditScore'] * df['Age']

# --- 3. Encode Categorical Columns ---
cat_cols = ['Geography', 'Gender', 'CreditScoreGroup']
encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# --- 4. Prepare Feature and Target ---
col_drop = ['Exited', 'RowNumber', 'CustomerId', 'Surname']
X = df.drop(col_drop, axis=1)
y = df['Exited']

# --- 5. Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- 6. Scale Numerical Features ---
scaling_columns = ['Age', 'CreditScore', 'Balance', 'EstimatedSalary',
                   'CreditUtilization', 'BalanceToSalaryRatio', 'CreditScoreAgeInteraction']

scaler = StandardScaler()
X_train[scaling_columns] = scaler.fit_transform(X_train[scaling_columns])
X_test[scaling_columns] = scaler.transform(X_test[scaling_columns])

# --- 7. Train Model (XGBoost) ---
model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),
    random_state=42
)

model.fit(X_train, y_train)

# --- 8. Evaluate Model (Optional) ---
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

# --- 9. Save Model, Scaler, and Encoders ---
joblib.dump(model, 'churn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(encoders, 'label_encoders.pkl')

print("\n✅ Model, scaler, and encoders saved successfully.")
