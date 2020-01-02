import pandas as pd
import numpy as np
import os
from utils.utils import convert_elapsed_time, del_invar_miss_col

def build_SURF(view_delete_cols=False):

    '''
    :print_delete_cols: bool, whether to print columns with more than 95% missing
    :return: Appended Pandas df of SURFs
    '''
    path = r'D:\USAF PME Board Evaluations\Download 20191211'
    surf = pd.DataFrame()
    for _ in sorted([file for file in os.listdir(path) if 'xlsx' in file and not '~$' in file], reverse=True)[:-1]:
        temp = pd.read_excel(os.path.join(path, _), skiprows=[0, 1, 2])
        temp['Year'] = int(_[10:14])  # year identifier for merge
        cols = [i for i in temp.columns if
                'Unnamed' not in i and 'NAME - PERSON' not in i and "SOCIAL SECURITY NUMBER" not in i]  # drop extra cols, SSN twice in data
        temp = temp[cols]
        temp.rename(columns={'SSAN': 'SSN'}, inplace=True)
        # shorten names of cols
        temp.columns = [i.replace('\n', ' ').replace(' - ', ' ').replace('LANGUAGE', 'LANG').replace('INDICATOR', '') \
                            .replace('HISTORY', 'HIST').replace('ACQUISITION', 'ACQ').replace('EXPERIENCE', 'EXP') \
                            .replace('AIRCRAFT', 'AC').replace('DUTY', 'DTY').replace('TITLE ', 'TITL') \
                            .replace('MOST RECENT', '').replace('COMMAND', 'CMD').replace('CREDIT', '') \
                            .replace('LEVEL', '').replace('MILITARY', 'MIL').replace('CALC YRS IN AF ', '') \
                            .replace('ASSIGNMENT', 'ASSN').replace('SERVICE', 'SVC').replace('TRAINING', 'TRAIN') \
                            .replace('LOCATION', 'LOC').replace("NUMBER", 'NR').replace('EFFECTIVE', 'EFF').replace(
            '  ', ' ') \
                            .replace('AVAILABILITY', 'AVAIL').replace('GROUP', 'GRP').replace('REASON', 'RSN') \
                            .replace('SEPARATION', 'SEP').replace('INSTRUCTOR', 'INST').replace('STATION', 'STATN') \
                            .replace('DEPARTED', 'DEPT').replace('ARRIVED', 'ARR').replace("DEPARTURE", 'DEPT') \
                            .replace('PENDING', 'PEND').replace('HISPANIC LATINO DESIGNATION', 'HISP') \
                            .replace('DEPENDENTS IN HOUSEHOLD', 'DEPEND').replace('RELIGIOUS DENOMINATION', 'RELIG') \
                            .replace('TOTAL', 'TOT').replace('PROGRAM', 'PGM').replace('CURRENT', 'CURR') \
                            .replace('HOURS', 'HRS').replace('DATE OF BIRTH', 'BDAY') for i in temp.columns]
        surf = pd.concat([surf, temp], axis=0, ignore_index=True, sort=False)

    # Consolidate variables
    surf['LANG ID 1'] = np.where(surf['LANG ID 1'].notnull(), surf['LANG ID 1'], surf['LANG ID 1.1'])
    surf['LANG ID 2'] = np.where(surf['LANG ID 2'].notnull(), surf['LANG ID 2'], surf['LANG ID 2.1'])
    surf.drop(columns=['LANG ID 1.1', 'LANG ID 2.1', 'DTY PHONE'], inplace=True)

    surf.drop_duplicates(subset=['SSN', 'Year'], inplace=True)  # drop duplicates within SSN year

    # Convert to datetime format
    dates = [i for i in surf.columns if "DATE" in i or 'BDAY' in i or 'DEROS' in i] # convert to datetime format
    for col in dates:
        surf[col] = convert_elapsed_time(surf[col])

    # Drop mostly missing columns & invariant columns
    surf = del_invar_miss_col(surf, view=False)

    return surf