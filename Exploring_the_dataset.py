import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    logging.info("Reading heart.csv file...")
    heart_df = pd.read_csv('../DM-and-BI-Project/Dataset/heart.csv')

    logging.info("Dropping rows with missing values...")
    heart_df = heart_df.dropna()

    logging.info("Changing data types of columns...")
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

    logging.info("Printing dataframe info:")
    print(heart_df.info())
    print()

    logging.info("Saving full dataframe description to file...")
    with open('heart_df_description.txt', 'w') as f:
        f.write(heart_df.describe().to_string())

    logging.info("Dataframe description saved to 'heart_df_description.txt'")

except Exception as e:
    logging.error(f"An error occurred: {e}")