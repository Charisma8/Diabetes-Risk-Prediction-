import pandas as pd
import numpy as np

print("Creating diabetes dataset...")

# Create sample diabetes data
np.random.seed(42)
n_samples = 1000

data = {
    'Pregnancies': np.random.randint(0, 10, n_samples),
    'Glucose': np.random.normal(120, 30, n_samples),
    'BloodPressure': np.random.normal(80, 15, n_samples),
    'BMI': np.random.normal(28, 8, n_samples),
    'Age': np.random.randint(20, 70, n_samples),
    'PhysicalActivity': np.random.randint(0, 5, n_samples),  # hours per week
    'FamilyHistory': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    'Smoking': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
    'GeneticRisk': np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.3, 0.2])
}

# Create target variable (diabetes or not)
risk_score = (
    np.abs(data['Glucose'] - 100) * 0.02 +
    (data['BMI'] - 25) * 0.1 +
    data['Age'] * 0.02 +
    data['FamilyHistory'] * 0.8 +
    data['GeneticRisk'] * 0.5 -
    data['PhysicalActivity'] * 0.2 +
    data['Smoking'] * 0.3
)

# Convert to probability
probability = 1 / (1 + np.exp(-risk_score + 2))
data['Diabetes'] = (np.random.random(n_samples) < probability).astype(int)

# Create DataFrame and save
df = pd.DataFrame(data)
df.to_csv('diabetes_data.csv', index=False)

print(f"✅ Data created successfully!")
print(f"📊 Total samples: {len(df)}")
print(f"🔴 Diabetes cases: {df['Diabetes'].sum()}")
print(f"📈 Diabetes rate: {df['Diabetes'].mean():.2%}")