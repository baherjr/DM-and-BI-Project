import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import FunctionSet


heart_df = pd.read_csv('../DM-and-BI-Project/Dataset/heart.csv')

heart_df = heart_df.dropna()

heart_df = heart_df.astype({
    'age': 'int8',
    'sex': 'int8',
    'cp': 'int8',
    'trestbps': 'int16',
    'chol': 'int16',
    'fbs': 'int8',
    'restecg': 'int8',
    'thalach': 'int16',
    'exang': 'int8',
    'oldpeak': 'float32',
    'slope': 'int8',
    'ca': 'int8',
    'thal': 'int8',
    'target': 'int8'
})

print(heart_df.info())


with open('heart_df_description.txt', 'w') as f:
    f.write(heart_df.describe().to_string())


print("\nDisease Presence Distribution:")
print(heart_df['target'].value_counts(normalize=True))

plt.figure(figsize=(12, 10))
sns.heatmap(heart_df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Heart Disease Features')
plt.tight_layout()
plt.show()


numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
for feature in numeric_features:
    FunctionSet.analyze_outlier_impact(feature, heart_df,method='zscore')
