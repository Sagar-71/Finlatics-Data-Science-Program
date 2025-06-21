# telemarketing_prediction.py

import pandas as pd
import numpy as np
import gdown
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Download dataset from Google Drive
url = 'https://drive.google.com/uc?id=1luWiMorf4NjS-0NU5uW-HWECUS_nYDjr'
output = 'telemarketing_dataset.csv'
gdown.download(url, output, quiet=False)

# Load dataset
df = pd.read_csv(output)

# Data preprocessing
def preprocess_data(df):
    # Convert target variable to binary
    df['y'] = df['y'].map({'yes': 1, 'no': 0})

    # Feature engineering
    df['contacted_previously'] = df['pdays'].apply(lambda x: 0 if x == -1 else 1)
    df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 70, 100], 
                             labels=['<30', '30-50', '50-70', '70+'])

    # Drop duration column (not available before call)
    if 'duration' in df.columns:
        df = df.drop('duration', axis=1)

    return df

df = preprocess_data(df)

# Split data
X = df.drop('y', axis=1)
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42, stratify=y)

# Define preprocessing pipeline
numeric_features = ['age', 'balance', 'campaign', 'pdays', 'previous']
categorical_features = ['job', 'marital', 'education', 'default', 'housing', 
                        'loan', 'contact', 'month', 'poutcome', 'age_group',
                        'contacted_previously']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Model pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
])

# Hyperparameter tuning
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Predictions
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

# Print results
print(f"\nBest Model: {grid_search.best_params_}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Deposit', 'Deposit'],
            yticklabels=['No Deposit', 'Deposit'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

# Feature importance for tree-based models
if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
    ohe_columns = list(best_model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features))
    all_features = numeric_features + ohe_columns
    importances = best_model.named_steps['classifier'].feature_importances_
    feature_importance = pd.DataFrame({'Feature': all_features, 'Importance': importances})
    feature_importance = feature_importance.sort_values('Importance', ascending=False).head(20)
    plt.figure(figsize=(12,8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Top 20 Feature Importances')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()

# Save the best model
joblib.dump(best_model, 'bank_telemarketing_model.pkl')
print("Best model saved as 'bank_telemarketing_model.pkl'")

# Example prediction
sample_data = X_test.iloc[[0]]
prediction = best_model.predict(sample_data)
probability = best_model.predict_proba(sample_data)[0][1]
print(f"\nSample Prediction: {'Deposit' if prediction[0] == 1 else 'No Deposit'} with {probability:.2f} probability")
