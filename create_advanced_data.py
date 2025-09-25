import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

print("🧬 Creating Advanced Multi-Omics Diabetes Dataset...")

np.random.seed(42)
n_samples = 2000  # Increased sample size

# === CLINICAL BIOMARKERS ===
clinical_data = {
    # Basic clinical (existing)
    'Age': np.random.randint(20, 80, n_samples),
    'BMI': np.random.normal(28, 8, n_samples),
    'Glucose': np.random.normal(120, 30, n_samples),
    'BloodPressure_Systolic': np.random.normal(130, 20, n_samples),
    'BloodPressure_Diastolic': np.random.normal(85, 15, n_samples),
    
    # Advanced clinical biomarkers
    'HbA1c': np.random.normal(5.8, 1.2, n_samples),  # Most important predictor
    'WaistCircumference': np.random.normal(95, 15, n_samples),
    'WaistHipRatio': np.random.normal(0.85, 0.1, n_samples),
    'HDL_Cholesterol': np.random.normal(50, 15, n_samples),
    'LDL_Cholesterol': np.random.normal(130, 30, n_samples),
    'Triglycerides': np.random.normal(150, 50, n_samples),
    'FastingInsulin': np.random.lognormal(2.5, 0.8, n_samples),
    'CReactiveProtein': np.random.lognormal(0.5, 1.0, n_samples),
    'GGT': np.random.gamma(2, 15, n_samples),  # Gamma-glutamyl transferase
}

# === LIFESTYLE FACTORS ===
lifestyle_data = {
    # Basic lifestyle
    'PhysicalActivity_Hours': np.random.gamma(2, 2, n_samples),
    'Smoking': np.random.choice([0, 1], n_samples, p=[0.75, 0.25]),
    
    # Advanced lifestyle factors
    'Mediterranean_Diet_Score': np.random.randint(0, 15, n_samples),
    'Processed_Food_Intake': np.random.randint(0, 10, n_samples),
    'Sleep_Duration': np.random.normal(7, 1.5, n_samples),
    'Sleep_Quality_Score': np.random.randint(1, 11, n_samples),
    'Stress_Level': np.random.randint(1, 11, n_samples),
    'Depression_Score': np.random.randint(0, 21, n_samples),  # PHQ-9 scale
    'Alcohol_Drinks_Per_Week': np.random.exponential(3, n_samples),
    'Sedentary_Hours_Daily': np.random.normal(8, 3, n_samples),
    
    # Social determinants
    'Education_Years': np.random.randint(8, 20, n_samples),
    'Income_Level': np.random.randint(1, 6, n_samples),  # 1-5 scale
    'Marital_Status': np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.5, 0.2]),  # Single, Married, Divorced
    'Employment_Status': np.random.choice([0, 1, 2], n_samples, p=[0.1, 0.8, 0.1]),  # Unemployed, Employed, Retired
}

# === GENETIC RISK FACTORS ===
genetic_data = {
    'Polygenic_Risk_Score': np.random.normal(0, 1, n_samples),  # Standardized PRS
    'TCF7L2_Risk_Alleles': np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.35, 0.05]),
    'PPARG_Risk_Alleles': np.random.choice([0, 1, 2], n_samples, p=[0.8, 0.18, 0.02]),
    'KCNJ11_Risk_Alleles': np.random.choice([0, 1, 2], n_samples, p=[0.85, 0.14, 0.01]),
    'CAPN10_Risk_Alleles': np.random.choice([0, 1, 2], n_samples, p=[0.7, 0.27, 0.03]),
    'Ethnicity_Risk': np.random.choice([0, 1, 2, 3], n_samples, p=[0.6, 0.15, 0.15, 0.1]),  # White, Hispanic, African, Asian
    'Family_History_T1D': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
    'Family_History_T2D': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
}

# === TEMPORAL FEATURES ===
temporal_data = {
    'Years_Since_Last_Checkup': np.random.exponential(2, n_samples),
    'Weight_Change_5Years': np.random.normal(0, 10, n_samples),  # kg change
    'BMI_Trajectory': np.random.choice([-1, 0, 1], n_samples, p=[0.3, 0.4, 0.3]),  # Decreasing, stable, increasing
}

# Combine all data
all_data = {**clinical_data, **lifestyle_data, **genetic_data, **temporal_data}

# === CREATE REALISTIC DIABETES TARGET ===
def calculate_diabetes_risk(data):
    # Advanced risk calculation incorporating all factors
    risk_components = {
        # Clinical factors (highest weight)
        'hba1c_risk': np.where(data['HbA1c'] >= 6.5, 3.0, 
                      np.where(data['HbA1c'] >= 5.7, 1.5, 0)),
        'glucose_risk': (data['Glucose'] - 100) * 0.015,
        'bmi_risk': np.maximum(data['BMI'] - 25, 0) * 0.08,
        'waist_risk': np.maximum(data['WaistCircumference'] - 88, 0) * 0.02,
        'insulin_risk': np.log(data['FastingInsulin']) * 0.3,
        
        # Lifestyle factors
        'diet_protection': (data['Mediterranean_Diet_Score'] - 7) * -0.1,
        'processed_food_risk': data['Processed_Food_Intake'] * 0.1,
        'activity_protection': data['PhysicalActivity_Hours'] * -0.15,
        'sleep_risk': np.abs(data['Sleep_Duration'] - 7) * 0.1,
        'stress_risk': data['Stress_Level'] * 0.05,
        'smoking_risk': data['Smoking'] * 0.4,
        
        # Genetic factors
        'prs_risk': data['Polygenic_Risk_Score'] * 0.8,
        'tcf7l2_risk': data['TCF7L2_Risk_Alleles'] * 0.3,
        'family_t2d_risk': data['Family_History_T2D'] * 0.6,
        
        # Age factor
        'age_risk': np.maximum(data['Age'] - 40, 0) * 0.03,
    }
    
    # Sum all risk components
    total_risk = sum(risk_components.values())
    
    # Convert to probability using logistic function
    probability = 1 / (1 + np.exp(-total_risk + 1.5))
    
    return probability

# Calculate realistic diabetes outcome
df = pd.DataFrame(all_data)
diabetes_probability = calculate_diabetes_risk(df)
df['Diabetes_Risk_Score'] = diabetes_probability
df['Diabetes'] = (np.random.random(n_samples) < diabetes_probability).astype(int)

# Add some data quality issues (realistic)
missing_percentage = 0.05
for col in ['Sleep_Quality_Score', 'Depression_Score', 'Income_Level']:
    missing_indices = np.random.choice(n_samples, int(n_samples * missing_percentage), replace=False)
    df.loc[missing_indices, col] = np.nan

# Save enhanced dataset
df.to_csv('diabetes_advanced_dataset.csv', index=False)

print(f"✅ Advanced dataset created successfully!")
print(f"📊 Total samples: {len(df)}")
print(f"🔴 Diabetes cases: {df['Diabetes'].sum()}")
print(f"📈 Diabetes rate: {df['Diabetes'].mean():.2%}")
print(f"🧬 Features: {len(df.columns)-1}")
print(f"📈 Risk score range: {df['Diabetes_Risk_Score'].min():.3f} - {df['Diabetes_Risk_Score'].max():.3f}")
