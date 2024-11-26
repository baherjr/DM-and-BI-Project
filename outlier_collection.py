import pandas as pd
import os


def outlier_collection(method):
    global first_file
    directory = os.scandir(f'outliers/{method.upper()}')
    if method == 'iqr':
        first_file = pd.read_csv(f'outliers/{method.upper()}/age_iqr_no_disease_outliers.csv')
    elif method == 'zscore':
        first_file = pd.read_csv(f'outliers/{method.upper()}/chol_zscore_disease_outliers.csv')
    Outlier_dataframe = first_file
    i = 0

    for file in directory:
        if i == 0:
            i += 1
            continue
        if file.name.endswith('.csv'):
            print(file.name)
            df = pd.read_csv(f'outliers/{method.upper()}/' + file.name)
            Outlier_dataframe = pd.concat([Outlier_dataframe, df], ignore_index=True)

    Outlier_dataframe.to_csv(f'../DM-and-Bi-Project/Dataset/{method}_outlier.csv', index=False)
