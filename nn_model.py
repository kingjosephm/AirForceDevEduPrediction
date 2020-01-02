import pandas as pd
import json


if __name__ == '__main__':

    df = pd.read_csv(r'\\pii_zippy\d\USAF PME Board Evaluations\Processed data\combined_data.csv')


    X = df.loc[:,~df.columns.isin(['SSN', 'Final rank'])]
    y = df['Final rank']
