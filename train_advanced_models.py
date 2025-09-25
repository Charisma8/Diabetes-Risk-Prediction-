import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import warnings
warnings.filterwarnings('ignore')

print("🚀 Training Advanced Multi-Omics Diabetes Prediction Models...")

# Load engineered data
df = pd.read_csv('diabetes_engineered_dataset.csv')

# Prepare features and target
target_col = 'Diabetes'
exclude_cols = [target_col, 'Diabetes_Risk_Score', 'HbA1c_Category', 'BMI_Category', 'Genetic_Risk_Category']
feature_cols = [col for col in df.columns if col not in exclude_cols]

X = df[feature_cols]
y = df[target_col]

# Handle any remaining missing values
X = X.fillna(X.median())

print(f"📊 Dataset shape: {X.shape}")
print(f"🎯 Target distribution: {y.value_counts().to_dict()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature selection
print("\n🔍 Performing feature selection...")
selector = SelectKBest(f_classif, k=25)  # Select top 25 features
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

selected_features = X.columns[selector.get_support()].tolist()
print(f"✅ Selected {len(selected_features)} features")

class AdvancedDiabetesPredictionSystem:
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.feature_selector = None
        self.ensemble_model = None
        
    def initialize_models(self):
        """Initialize advanced ML models"""
        self.models = {
            'Logistic_Regression': LogisticRegression(
                random_state=42, max_iter=1000, class_weight='balanced'
            ),
            
            'Random_Forest': RandomForestClassifier(
                n_estimators=200, max_depth=10, min_samples_split=5,
                random_state=42, class_weight='balanced'
            ),
            
            'XGBoost': xgb.XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, eval_metric='logloss'
            ),
            
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbose=-1
            ),
            
            'Neural_Network': MLPClassifier(
                hidden_layer_sizes=(100, 50, 25), max_iter=1000,
                random_state=42, early_stopping=True,
                validation_fraction=0.1
            ),
            
            'SVM': SVC(
                kernel='rbf', probability=True, class_weight='balanced',
                random_state=42
            ),
            
            'Gradient_Boosting': GradientBoostingClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                random_state=42
            )
        }
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train all models and evaluate performance"""
        self.initialize_models()
        
        results = {}
        trained_models = {}
        
        print("\n🤖 Training advanced models...")
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            results[name] = {
                'Accuracy': accuracy,
                'AUC': auc,
                'Model': model
            }
            
            trained_models[name] = model
            
            print(f"✅ {name}: Accuracy={accuracy:.4f}, AUC={auc:.4f}")
        
        self.trained_models = trained_models
        return results
    
    def create_ensemble(self, results, X_train, y_train):
        """Create ensemble of best models"""
        print("\n🎭 Creating ensemble model...")
        
        # Select top 3 models by AUC
        top_models = sorted(results.items(), key=lambda x: x[1]['AUC'], reverse=True)[:3]
        
        ensemble_estimators = [(name, results[name]['Model']) for name, _ in top_models]
        
        self.ensemble_model = VotingClassifier(
            estimators=ensemble_estimators,
            voting='soft'
        )
        
        self.ensemble_model.fit(X_train, y_train)
        
        print(f"✅ Ensemble created with: {[name for name, _ in ensemble_estimators]}")
        
        return self.ensemble_model
    
    def evaluate_ensemble(self, X_test, y_test):
        """Evaluate ensemble performance"""
        y_pred = self.ensemble_model.predict(X_test)
        y_pred_proba = self.ensemble_model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\n🏆 Ensemble Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC: {auc:.4f}")
        
        return accuracy, auc

# Train the advanced system
system = AdvancedDiabetesPredictionSystem()
results = system.train_models(X_train_selected, y_train, X_test_selected, y_test)

# Create ensemble
ensemble = system.create_ensemble(results, X_train_selected, y_train)
ensemble_accuracy, ensemble_auc = system.evaluate_ensemble(X_test_selected, y_test)

# Feature importance analysis
print("\n🔍 Top 10 Most Important Features:")
best_model_name = max(results.keys(), key=lambda k: results[k]['AUC'])
best_model = results[best_model_name]['Model']

if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': selected_features,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nUsing {best_model_name} for feature importance:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"{row['feature']}: {row['importance']:.4f}")

# Save models and preprocessing objects
joblib.dump(ensemble, 'diabetes_ensemble_model.pkl')
joblib.dump(scaler, 'feature_scaler.pkl')
joblib.dump(selector, 'feature_selector.pkl')
joblib.dump(selected_features, 'selected_features.pkl')

# Save results
results_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[k]['Accuracy'] for k in results.keys()],
    'AUC': [results[k]['AUC'] for k in results.keys()]
})
results_df.loc[len(results_df)] = ['Ensemble', ensemble_accuracy, ensemble_auc]
results_df.to_csv('model_comparison_results.csv', index=False)

print(f"\n💾 All models and preprocessing objects saved!")
print(f"📊 Model comparison saved to 'model_comparison_results.csv'")
print("✅ Advanced training complete!")
