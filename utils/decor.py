import pandas as pd
import numpy as np
import os

def build_decor():

    '''
    Creates dataframe of individuals' decorations per year. Does so by using file names in a directory containing PDFs
    and TIFs of individual decorations. File names contain SSN and decoration type, so images themselves not necessary.
    :returns: pandas dataframe of SSN x year decoration counts by type of decoration (these as columns).
    '''

    path = r'\\pii_zippy\d\USAF PME Board Evaluations\Download 20191211\decorations'

    files = [file[:-4] for file in os.listdir(path) if ".TIF" in file or ".PDF" in file] # get all TIF & PDF files in directory

    df = []
    for index, item in enumerate(files):
        ssn = int(files[index][:9])
        decoration = files[index][10:-9]
        date = files[index][-8:]
        df.append([ssn, decoration, date])
    df = pd.DataFrame(df, columns=['SSN', 'DECOR', 'DATE'])
    df['DATE'] = pd.to_datetime(df['DATE'], format='%Y%m%d', errors='raise')

    # Create decorations specific to year in which applicant was reviewed by board (don't want to include future decorations)
    maxdates = pd.read_csv(os.path.join(os.path.join(r'\\pii_zippy\d\USAF PME Board Evaluations\Processed data', 'ssn_board_date.csv')))
    maxdates['Board date'] = pd.to_datetime(maxdates['Board date'], format='%Y%m%d', errors='raise')
    maxdates['Year'] = maxdates['Board date'].dt.year

    decor = pd.DataFrame()
    for yr in sorted(maxdates['Year'].unique()):
        temp = df.merge(maxdates[maxdates['Year']==yr], on=['SSN'], how='right') # keep SSNs in each year
        temp = temp.loc[temp['DATE'] < temp['Board date']] # no decorations beyond board date this year
        temp = temp.groupby(['SSN', 'DECOR'])['DATE'].count().reset_index() # count number of decorations per SSN
        temp = temp.pivot(index='SSN', columns='DECOR', values='DATE').reset_index() # long to wide pivot
        temp['Year'] = yr
        decor = pd.concat([decor, temp], axis=0, ignore_index=True, sort=True)
    decor.replace(np.NaN, 0, inplace=True)
    return decor