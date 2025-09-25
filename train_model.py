import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib
import numpy as np

print("🚀 Starting Diabetes Risk Prediction Model Training...")

# Load the data
df = pd.read_csv('diabetes_data.csv')
print(f"✅ Data loaded successfully! Shape: {df.shape}")

# Prepare the data
print("\\n📊 Preparing data...")
feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'BMI', 'Age', 
                  'PhysicalActivity', 'FamilyHistory', 'Smoking', 'GeneticRisk']
X = df[feature_columns]
y = df['Diabetes']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Train models
print("\\n🤖 Training models...")

# Logistic Regression
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)
lr_auc = roc_auc_score(y_test, lr_model.predict_proba(X_test)[:, 1])

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])

# Compare models
print("\\n📈 Model Performance:")
print(f"Logistic Regression - Accuracy: {lr_accuracy:.3f}, AUC: {lr_auc:.3f}")
print(f"Random Forest - Accuracy: {rf_accuracy:.3f}, AUC: {rf_auc:.3f}")

# Choose best model
if rf_auc >= lr_auc:
    best_model = rf_model
    best_name = "Random Forest"
    best_accuracy = rf_accuracy
    best_auc = rf_auc
else:
    best_model = lr_model
    best_name = "Logistic Regression"
    best_accuracy = lr_accuracy
    best_auc = lr_auc

print(f"\\n🏆 Best Model: {best_name}")
print(f"Final Accuracy: {best_accuracy:.3f}")
print(f"Final AUC: {best_auc:.3f}")

# Feature importance
if best_name == "Random Forest":
    print("\\n🔍 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, row in feature_importance.head(5).iterrows():
        print(f"{row['feature']}: {row['importance']:.3f}")

# Save model
joblib.dump(best_model, 'diabetes_model.pkl')
joblib.dump(feature_columns, 'feature_names.pkl')

print("\\n💾 Model saved successfully!")
print("✅ Training complete! Ready to build web app.")