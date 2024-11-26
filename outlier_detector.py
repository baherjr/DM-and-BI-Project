import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def analyze_outlier_impact(feature, df, output_dir, method='iqr'):

    os.makedirs(output_dir, exist_ok=True)

    disease = df[df['target'] == 1]
    no_disease = df[df['target'] == 0]

    def get_outliers_iqr(df):
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]

    def get_outliers_zscore(df, threshold=3):
        z_scores = np.abs((df[feature] - df[feature].mean()) / df[feature].std())
        return df[z_scores > threshold]

    if method.lower() == 'iqr':
        disease_outliers = get_outliers_iqr(disease)
        no_disease_outliers = get_outliers_iqr(no_disease)
        method_name = 'IQR'
    elif method.lower() == 'zscore':
        disease_outliers = get_outliers_zscore(disease)
        no_disease_outliers = get_outliers_zscore(no_disease)
        method_name = 'Z-Score'
    else:
        raise ValueError("Method must be either 'iqr' or 'zscore'")

    print(f"\nOutlier Analysis for '{feature}' using {method_name} method:")
    print(f"Disease Group: {len(disease_outliers)} outliers out of {len(disease)} records "
          f"({len(disease_outliers) / len(disease) * 100:.2f}%)")
    print(f"No Disease Group: {len(no_disease_outliers)} outliers out of {len(no_disease)} records "
          f"({len(no_disease_outliers) / len(no_disease) * 100:.2f}%)")

    def save_outliers(outliers, group_name):
        if not outliers.empty:
            file_path = os.path.join(output_dir, f'{feature}_{method}_{group_name}_outliers.csv')
            outliers.to_csv(file_path, index=False)
            print(f"{group_name.capitalize()} group outliers saved to '{file_path}'")
        else:
            print(f"No outliers found in {group_name} group.")

    save_outliers(disease_outliers, 'disease')
    save_outliers(no_disease_outliers, 'no_disease')

    # Create visualization
    plt.figure(figsize=(12, 6))

    # Create subplot for boxplot
    plt.subplot(1, 2, 1)
    sns.boxplot(x='target', y=feature, data=df, hue='target', palette='Set2', dodge=False)
    plt.title(f"'{feature}' Distribution by Target\n(Boxplot)")
    plt.xlabel('Target (0: No Disease, 1: Disease)')
    plt.ylabel(feature)

    # Create subplot for distribution plot
    plt.subplot(1, 2, 2)
    sns.histplot(data=df, x=feature, hue='target', multiple="layer", alpha=0.5)
    if method.lower() == 'zscore':
        mean = df[feature].mean()
        std = df[feature].std()
        plt.axvline(mean - 3*std, color='r', linestyle='--', alpha=0.5, label='±3 SD')
        plt.axvline(mean + 3*std, color='r', linestyle='--', alpha=0.5)
    plt.title(f"'{feature}' Distribution by Target\n(Histogram)")
    plt.xlabel(feature)
    plt.ylabel('Count')

    plt.tight_layout()
    plt.show()

    return disease_outliers, no_disease_outliers