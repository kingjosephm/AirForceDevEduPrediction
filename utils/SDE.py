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

    # Board composition
    # Source: "\\pii_zippy\d\USAF PME Board Evaluations\Download 20191122\2018-19\2019 Central Board_IDE-SDEv5 (FINAL).xlsx"
    # "SDE 2019 board composition.pdf" in same folder does not contain most board member names
    df19['race_Toth'] = 'WHITE'
    df19['gender_Toth'] = 'MALE'
    df19['hisp_Toth'] = 'NOT HISPANIC OR LATINO'
    df19['afsc_Toth'] = '11F'
    #
    df19['race_McGee'] = 'WHITE'
    df19['gender_McGee'] = 'FEMALE'
    df19['hisp_McGee'] = 'HISPANIC OR LATINO'
    df19['afsc_McGee'] = '13S'
    #
    df19['race_Hackbarth'] = 'WHITE'
    df19['gender_Hackbarth'] = 'MALE'
    df19['hisp_Hackbarth'] = 'NOT HISPANIC OR LATINO'
    df19['afsc_Hackbarth'] = '11M'
    #
    df19['race_Solomon'] = 'WHITE'
    df19['gender_Solomon'] = 'MALE'
    df19['hisp_Solomon'] = 'NOT HISPANIC OR LATINO'
    df19['afsc_Solomon'] = '17D'
    #
    df19['race_Onufer'] = 'WHITE'
    df19['gender_Onufer'] = 'FEMALE'
    df19['hisp_Onufer'] = 'NOT HISPANIC OR LATINO'
    df19['afsc_Onufer'] = '12S'
    #
    df19['race_Farmer'] = 'WHITE'
    df19['gender_Farmer'] = 'MALE'
    df19['hisp_Farmer'] = 'NOT HISPANIC OR LATINO'
    df19['afsc_Farmer'] = '11B'
    #
    df19['race_Graham'] = 'WHITE'
    df19['gender_Graham'] = 'MALE'
    df19['hisp_Graham'] = 'NOT HISPANIC OR LATINO'
    df19['afsc_Graham'] = '21R'
    #
    df19['race_Ebner'] = 'WHITE'
    df19['gender_Ebner'] = 'FEMALE'
    df19['hisp_Ebner'] = 'NOT HISPANIC OR LATINO'
    df19['afsc_Ebner'] = '11F'
    #
    df19['race_Collins'] = 'WHITE'
    df19['gender_Collins'] = 'MALE'
    df19['hisp_Collins'] = 'NOT HISPANIC OR LATINO'
    df19['afsc_Collins'] = '63A'
    #
    df19['race_Katzer'] = 'WHITE'
    df19['gender_Katzer'] = 'MALE'
    df19['hisp_Katzer'] = 'NOT HISPANIC OR LATINO'
    df19['afsc_Katzer'] = '32E'

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

    # Board composition
    # Source: "\\pii_zippy\d\USAF PME Board Evaluations\Download 20191122\2018-19\2018 BOARD WORKSHEET (IDE-SDE)v3_6 apr 18.xlsx"
    # "SDE 2018 board composition.pdf" in this same directory missing some board member names that were in original data
    df19['race_Cluff'] = 'WHITE'
    df19['gender_Cluff'] = 'MALE'
    df19['hisp_Cluff'] = 'NOT HISPANIC OR LATINO'
    df19['afsc_Cluff'] = '11F'
    #
    df19['race_Johnson'] = 'WHITE'
    df19['gender_Johnson'] = 'FEMALE'
    df19['hisp_Johnson'] = 'NOT HISPANIC OR LATINO'
    df19['afsc_Johnson'] = '17D'
    #
    df19['race_Richardson'] = 'WHITE'
    df19['gender_Richardson'] = 'MALE'
    df19['hisp_Richardson'] = 'NOT HISPANIC OR LATINO'
    df19['afsc_Richardson'] = '12M'
    #
    df19['race_Townsend'] = 'BLACK OR AFRICAN AMERICAN'
    df19['gender_Townsend'] = 'MALE'
    df19['hisp_Townsend'] = 'NOT HISPANIC OR LATINO'
    df19['afsc_Townsend'] = '13N'
    #
    df19['race_Bass'] = 'WHITE'
    df19['gender_Bass'] = 'MALE'
    df19['hisp_Bass'] = 'NOT HISPANIC OR LATINO'
    df19['afsc_Bass'] = '13B'
    #
    df19['race_Cullen'] = 'WHITE'
    df19['gender_Cullen'] = 'MALE'
    df19['hisp_Cullen'] = 'NOT HISPANIC OR LATINO'
    df19['afsc_Cullen'] = '31P'
    #
    df19['race_Cantwell'] = 'ASIAN'
    df19['gender_Cantwell'] = 'MALE'
    df19['hisp_Cantwell'] = 'NOT HISPANIC OR LATINO'
    df19['afsc_Cantwell'] = '11F'
    #
    df19['race_Ashley'] = 'WHITE'
    df19['gender_Ashley'] = 'MALE'
    df19['hisp_Ashley'] = 'NOT HISPANIC OR LATINO'
    df19['afsc_Ashley'] = '62E'
    #
    df19['race_Mineau'] = 'WHITE'
    df19['gender_Mineau'] = 'MALE'
    df19['hisp_Mineau'] = 'NOT HISPANIC OR LATINO'
    df19['afsc_Mineau'] = '11F'
    #
    df19['race_Belz'] = 'WHITE'
    df19['gender_Belz'] = 'MALE'
    df19['hisp_Belz'] = 'NOT HISPANIC OR LATINO'
    df19['afsc_Belz'] = '21A'

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

    # Board composition
    # Source: "\\pii_zippy\d\USAF PME Board Evaluations\Download 20191122\2016-17\CENTRAL PME BOARD WORKSHEET_IDE_SDEV2 - PRIMARIES ONLY 2017.xlsx"
    #
    df19['race_Leavitt'] = 'WHITE'
    df19['gender_Leavitt'] = 'FEMALE'
    df19['hisp_Leavitt'] = 'NOT HISPANIC OR LATINO'
    df19['afsc_Leavitt'] = '11F'
    #
    df19['race_Roberson'] = 'WHITE'
    df19['gender_Roberson'] = 'MALE'
    df19['hisp_Roberson'] = 'NOT HISPANIC OR LATINO'
    df19['afsc_Roberson'] = '13S'
    #
    df19['race_Cooper'] = 'WHITE'
    df19['gender_Cooper'] = 'MALE'
    df19['hisp_Cooper'] = 'NOT HISPANIC OR LATINO'
    df19['afsc_Cooper'] = '11R'
    #
    df19['race_Purdy'] = 'WHITE'
    df19['gender_Purdy'] = 'MALE'
    df19['hisp_Purdy'] = 'NOT HISPANIC OR LATINO'
    df19['afsc_Purdy'] = '63A'
    #
    df19['race_Krishna'] = 'WHITE'
    df19['gender_Krishna'] = 'MALE'
    df19['hisp_Krishna'] = 'NOT HISPANIC OR LATINO'
    df19['afsc_Krishna'] = '12R'
    #
    df19['race_Delgado'] = 'WHITE'
    df19['gender_Delgado'] = 'MALE'
    df19['hisp_Delgado'] = 'HISPANIC OR LATINO'
    df19['afsc_Delgado'] = '17D'
    #
    df19['race_Spain'] = 'BLACK OR AFRICAN AMERICAN'
    df19['gender_Spain'] = 'MALE'
    df19['hisp_Spain'] = 'NOT HISPANIC OR LATINO'
    df19['afsc_Spain'] = '11F'
    #
    df19['race_Farrar'] = 'WHITE'
    df19['gender_Farrar'] = 'MALE'
    df19['hisp_Farrar'] = 'NOT HISPANIC OR LATINO'
    df19['afsc_Farrar'] = '31P'
    #
    df19['race_McDaniel'] = 'WHITE'
    df19['gender_McDaniel'] = 'MALE'
    df19['hisp_McDaniel'] = 'NOT HISPANIC OR LATINO'
    df19['afsc_McDaniel'] = '11M'
    #
    df19['race_McCray'] = 'WHITE'
    df19['gender_McCray'] = 'MALE'
    df19['hisp_McCray'] = 'NOT HISPANIC OR LATINO'
    df19['afsc_McCray'] = '21A'

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

    cols = ['SSN', 'Year', 'Final rank', 'Final score', 'Ballot rank', 'Order', 'AFSC', 'Board date']\
           +[i for i in df.columns if "Race_" in i or "Gender_" in i or "Afsc_" in i or "Hisp_" in i]

    # Note - excluded reviewers and adjustments
    df = df[cols]

    df = df.sort_values(by=['Year', 'Final rank']).reset_index(drop=True)

    # Output df of SSNs and board dates for other scripts
    df[['SSN', 'Board date']].to_csv(os.path.join(r'\\pii_zippy\d\USAF PME Board Evaluations\Processed data', 'ssn_board_date.csv'), index=False)
    del df['Board date']

    return df