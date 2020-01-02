import pandas as pd
import os

def build_SDE():

    '''
    :return: Appended Pandas df of annual SDE results
    '''
    directory = r'\\pii_zippy\d\USAF PME Board Evaluations\Download 20191122\copies_useful_docs'

    #################################################
    #####               2019 SDE                #####
    #################################################

    df19 = pd.read_excel(os.path.join(directory, '2019_SDE_Final.xlsx'))
    df19.drop(columns=['NOTES', 'STATUS', '1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7', '1.8', '1.9', '1.10',
                       'Xtra_Info', 'TOTAL SCORE', 'AVERAGE SCORE', 'PC Rank', 'Original Ties', 'Sel/Cad', 'DT',
                       'Avg Pct', 'Total Score (num)', 'Avg Score (num)'],
              inplace=True)  # computer-generated, duplicated in other cols
    df19.rename(columns={'BOARD SEQ NO.': 'Order', 'SOCIAL SECURITY NO.': 'SSN'}, inplace=True)
    df19 = df19.iloc[:, :-6]

    df19.rename(columns={'1.1.1': 'Toth', '1.2.1': 'McGee', '1.3.1': 'Hackbarth', '1.4.1': 'Solomon', '1.5.1': 'Onufer',
                         '1.6.1': 'Farmer', '1.7.1': 'Graham', '1.8.1': 'Ebner', '1.9.1': 'Collins', '1.10.1': 'Katzer',
                         'Manual/Final Order of Merit': 'Final Rank', 'New Avg': 'Final Score'},
                inplace=True)  # names from '\\pii_zippy\d\USAF PME Board Evaluations\Download 20191122\2018-19\Final Export 18 Apr 19_board mbr sheets.xlsx'

    for num in range(1, 11):
        df19.columns = df19.columns.str.replace(r'adj2.' + str(num), 'adj' + str(num))
        df19.columns = df19.columns.str.replace(r'pct2.' + str(num), 'pct' + str(num))

    df19 = df19.dropna()

    df19['Board Date'] = '20190408'
    df19['year'] = 2019

    # Standardize reviewer scores across each other
    reviewers = ['Toth', 'McGee', 'Hackbarth', 'Solomon', 'Onufer', 'Farmer', 'Graham', 'Ebner', 'Collins', 'Katzer']
    df19[reviewers] = df19[reviewers].apply(lambda x: (x - df19[reviewers].mean(axis=1)) / df19[reviewers].std(axis=1))

    #################################################
    #####               2018 SDE                #####
    #################################################

    df18 = pd.read_excel(os.path.join(directory, '2018_SDE_Final.xlsx'))
    df18.rename(columns={'CANDIDATE -- SDE': 'CANDIDATE', 'Final rank': 'Final Rank'}, inplace=True)
    # Get scores from individual raters
    temp = pd.read_excel(os.path.join(directory, 'DSY Tie breaker sheet 2018.xlsx'))
    for _ in ['CLUFF', 'JOHNSON', 'RICHARDSON', 'TOWNSEND', 'BASS', 'CULLEN', 'CANTWELL', 'ASHLEY', 'MINEAU', 'BELZ']:
        temp[_] = temp[_].str[:-12].astype(float)
    temp.drop(columns=['CLUFF.1', 'JOHNSON.1', 'RICHARDSON.1', 'TOWNSEND.1', 'BASS.1', 'CULLEN.1',
                       'CANTWELL.1', 'ASHLEY.1', 'MINEAU.1', 'BELZ.1', 'AVERAGE SCORE', 'TOTAL SCORE', 'COMPUTER',
                       'Original Ties', 'TOTAL SCORE.1', 'AVERAGE SCORE.1', 'DT', 'Avg PCT', 'Total Score (num)',
                       'Avg Score (Num)'], inplace=True)
    temp = temp.iloc[:, :-5]
    temp.rename(columns={'#': 'Order', 'Final': 'Final Rank', 'Avg ADJ': 'Avg Adj', 'New Avg': 'Final Score'},
                inplace=True)
    temp = temp.dropna()
    for num in range(1, 11):
        temp.columns = temp.columns.str.replace(r'ADJ 2.' + str(num), 'adj' + str(num))
        temp.columns = temp.columns.str.replace(r'PCT 2.' + str(num), 'pct' + str(num))

    df18 = df18[['SSN']].merge(temp, on='SSN', how='left')
    del temp, _, num

    df18 = df18.dropna()

    df18['Board Date'] = '20180514'
    df18['year'] = 2018

    # Standardize reviewer scores across each other
    reviewers = ['CLUFF', 'JOHNSON', 'RICHARDSON', 'TOWNSEND', 'BASS', 'CULLEN', 'CANTWELL', 'ASHLEY', 'MINEAU', 'BELZ']
    df18[reviewers] = df18[reviewers].apply(lambda x: (x - df18[reviewers].mean(axis=1)) / df18[reviewers].std(axis=1))

    #################################################
    #####               2017 SDE                #####
    #################################################

    df17 = pd.read_excel(os.path.join(directory, '2017_SDE_Final.xlsx'), skiprows=[0, 1])
    df17.drop(columns=['Unnamed: 16', 'Pct Score', 'Average Score'], inplace=True)
    df17.rename(columns={'SOCIAL SECURITY NO.': 'SSN', 'BOARD SEQ NO.': 'Order', 'Core AFSC': 'AFSC',
                         'Rank (with Ties)': 'Ballot Rank', 'New Rank (no ties)': 'Final Rank',
                         'New Average': 'Final Score'}, inplace=True)
    df17 = df17.iloc[:, :-2]
    for num in range(1, 11):
        df17.columns = df17.columns.str.replace(r'Adj 2.' + str(num), 'adj' + str(num))
        df17.columns = df17.columns.str.replace(r'Pct 2.' + str(num), 'pct' + str(num))

    # col namees compared with 'SDE Board Member Scores 2017.xlsx'
    df17.rename(columns={'1.1': 'Leavitt', '1.2': 'Roberson', '1.3': 'Cooper', '1.4': 'Purdy', '1.5': 'Krishna',
                         '1.6': 'Delgado', '1.7': 'Spain', '1.8': 'Farrar', '1.9': 'McDaniel', '1.10': 'McCray',
                         'Adj Score': 'Avg Adj'}, inplace=True)

    df17 = df17.dropna()

    df17['Board Date'] = '20170403'
    df17['year'] = 2017

    # Standardize reviewer scores across each other
    reviewers = ['Leavitt', 'Roberson', 'Cooper', 'Purdy', 'Krishna', 'Delgado', 'Spain', 'Farrar', 'McDaniel', 'McCray']
    df17[reviewers] = df17[reviewers].apply(lambda x: (x - df17[reviewers].mean(axis=1)) / df17[reviewers].std(axis=1))

    '''

    #################################################
    #####               2016 SDE                #####
    #################################################

    df16 = pd.read_excel(os.path.join(directory, '2016_SDE_Final.xlsx'), skiprows=[0, 1])
    df16.drop(columns=['new score', 'STATUS', 'Unnamed: 17', 'new score.1', 'New Rank (no ties)'], inplace=True)
    # only 2016 there are two final-looking columns: 'New Rank (no ties)' and 'Final rank after board broke ties'
    # I use latter. They're corr at 0.999973
    df16.rename(columns={'SOCIAL SECURITY NO.': 'SSN', 'Rank (with Ties)': 'Ballot Rank',
                         'Final rank after board broke ties': 'Final Rank', 'New Average': 'New Avg',
                         'Average Score': 'Avg Score', 'CORE': 'AFSC'}, inplace=True) # should CORE be coded to AFSC or DT?

    df16['Order'] = np.NaN # missing this year

    for num in range(1, 11):
        df16.columns = df16.columns.str.replace(r'Adj 2.'+str(num), 'adj'+str(num))
        df16.columns = df16.columns.str.replace(r'Pct 2.' + str(num), 'pct' + str(num))
        df16.columns = df16.columns.str.replace(r'Member 2.' + str(num), 'r' + str(num))

    # deduced names from '//pii_zippy/d/USAF PME Board Evaluations/Download 20191122/2016-17/Central Board Panel Members v2 2016.xlsx'
    df16.rename(columns={'r1': 'Bunnell', 'r2': 'Dunn', 'r3': 'Howard', 'r4': 'Rodriguez', 'r5': 'Tromba', 'r6': 'Norman',
                         'r7': 'Marshall', 'r8': 'Olyniec', 'r9': 'Craycraft', 'r10': 'Schaefer'}, inplace=True)

    df16['Board Date'] = '20160411'
    df16['year'] = 2016

    '''

    #################################################
    #####             Append SDEs               #####
    #################################################

    df = pd.concat([df19, df18, df17], axis=0, sort=True)
    df.columns = [i.capitalize() for i in df.columns]
    df.rename(columns={'Ssn': 'SSN', 'Afsc': 'AFSC'}, inplace=True)
    df.sort_values(by=['SSN', 'Year'], inplace=True)
    # Verify no duplicate records within year
    assert(df.duplicated(subset=['SSN', 'Year']).all()==False), "\nThere are duplicate records within the same year!"

    cols = ['SSN', 'Year', 'Final rank', 'Final score', 'Ballot rank',
            'Adj1', 'Adj2', 'Adj3', 'Adj4', 'Adj5', 'Adj6', 'Adj7', 'Adj8', 'Adj9', 'Adj10', 'Avg adj',
            'Pct1', 'Pct2', 'Pct3', 'Pct4', 'Pct5', 'Pct6', 'Pct7', 'Pct8', 'Pct9', 'Pct10',
            'Order', 'AFSC', 'Board date']

    reviewers = [i for i in df.columns if i not in cols and "Candidate" not in i]
    columns = cols + reviewers
    df = df[columns]

    df = df.sort_values(by=['Year', 'Final rank']).reset_index(drop=True)

    # Output df of SSNs and board dates for other scripts
    df[['SSN', 'Board date']].to_csv(os.path.join(r'\\pii_zippy\d\USAF PME Board Evaluations\Processed data', 'ssn_board_date.csv'), index=False)
    del df['Board date']

    # Remove adjustment and percentage adjustment scores
    cols = [i for i in df.columns if "Adj" not in i and "Pct" not in i and "Avg adj" not in i]
    df = df[cols]

    return df