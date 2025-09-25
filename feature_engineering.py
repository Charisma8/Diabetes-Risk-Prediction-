import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier

print("🔧 Advanced Feature Engineering...")

# Load data
df = pd.read_csv('diabetes_advanced_dataset.csv')

class AdvancedFeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_selector = None
        
    def handle_missing_values(self, df):
        """Advanced missing value imputation"""
        # For clinical values, use median
        clinical_cols = ['Sleep_Quality_Score', 'Depression_Score']
        for col in clinical_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        # For categorical, use mode
        categorical_cols = ['Income_Level']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0])
        
        return df
    
    def create_interaction_features(self, df):
        """Create meaningful feature interactions"""
        # BMI * Genetic Risk
        df['BMI_Genetic_Risk'] = df['BMI'] * (1 + df['Polygenic_Risk_Score'])
        
        # Metabolic syndrome indicator
        df['Metabolic_Syndrome_Score'] = (
            (df['WaistCircumference'] > 88).astype(int) +
            (df['HDL_Cholesterol'] < 40).astype(int) +
            (df['Triglycerides'] > 150).astype(int) +
            (df['BloodPressure_Systolic'] > 130).astype(int) +
            (df['Glucose'] > 100).astype(int)
        )
        
        # Lifestyle health score
        df['Lifestyle_Health_Score'] = (
            df['Mediterranean_Diet_Score'] * 0.3 +
            df['PhysicalActivity_Hours'] * 0.4 +
            (11 - df['Stress_Level']) * 0.2 +
            df['Sleep_Quality_Score'] * 0.1
        )
        
        # Age-adjusted BMI
        df['Age_Adjusted_BMI'] = df['BMI'] * (1 + (df['Age'] - 40) / 100)
        
        return df
    
    def create_risk_categories(self, df):
        """Create categorical risk features"""
        # HbA1c categories
        df['HbA1c_Category'] = pd.cut(df['HbA1c'], 
                                     bins=[0, 5.7, 6.5, float('inf')],
                                     labels=['Normal', 'Prediabetes', 'Diabetes'])
        
        # BMI categories
        df['BMI_Category'] = pd.cut(df['BMI'],
                                   bins=[0, 18.5, 25, 30, float('inf')],
                                   labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        
        # Genetic risk categories
        df['Genetic_Risk_Category'] = pd.cut(df['Polygenic_Risk_Score'],
                                           bins=[-float('inf'), -0.5, 0.5, float('inf')],
                                           labels=['Low', 'Medium', 'High'])
        
        return df
    
    def engineer_features(self, df):
        """Complete feature engineering pipeline"""
        print("📊 Starting feature engineering...")
        
        # Handle missing values
        df = self.handle_missing_values(df)
        print(f"✅ Missing values handled")
        
        # Create interactions
        df = self.create_interaction_features(df)
        print(f"✅ Interaction features created")
        
        # Create risk categories
        df = self.create_risk_categories(df)
        print(f"✅ Risk categories created")
        
        # Encode categorical variables
        categorical_cols = ['HbA1c_Category', 'BMI_Category', 'Genetic_Risk_Category', 
                          'Marital_Status', 'Employment_Status', 'Ethnicity_Risk']
        
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col + '_Encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        print(f"✅ Categorical encoding completed")
        print(f"📈 Final feature count: {len(df.columns)}")
        
        return df

# Apply feature engineering
engineer = AdvancedFeatureEngineer()
df_engineered = engineer.engineer_features(df)

# Save engineered dataset
df_engineered.to_csv('diabetes_engineered_dataset.csv', index=False)
print("💾 Engineered dataset saved!")
