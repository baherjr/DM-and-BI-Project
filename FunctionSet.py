import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def analyze_outlier_impact(feature, df, output_dir='outliers'):
    os.makedirs(output_dir, exist_ok=True)

    disease = df[df['target'] == 1]
    no_disease = df[df['target'] == 0]

    def get_outliers(df):
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]

    disease_outliers = get_outliers(disease)
    no_disease_outliers = get_outliers(no_disease)

    print(f"\nOutlier Analysis for '{feature}':")
    print(f"Disease Group: {len(disease_outliers)} outliers out of {len(disease)} records "
          f"({len(disease_outliers) / len(disease) * 100:.2f}%)")
    print(f"No Disease Group: {len(no_disease_outliers)} outliers out of {len(no_disease)} records "
          f"({len(no_disease_outliers) / len(no_disease) * 100:.2f}%)")

    def save_outliers(outliers, group_name):
        if not outliers.empty:
            file_path = os.path.join(output_dir, f'{feature}_{group_name}_outliers.csv')
            outliers.to_csv(file_path, index=False)
            print(f"{group_name.capitalize()} group outliers saved to '{file_path}'")
        else:
            print(f"No outliers found in {group_name} group.")

    save_outliers(disease_outliers, 'disease')
    save_outliers(no_disease_outliers, 'no_disease')

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='target', y=feature, data=df, hue='target', palette='Set2', dodge=False)
    plt.title(f"'{feature}' Distribution by Target")
    plt.xlabel('Target (0: No Disease, 1: Disease)')
    plt.ylabel(feature)
    plt.tight_layout()
    plt.show()

    return disease_outliers, no_disease_outliers
