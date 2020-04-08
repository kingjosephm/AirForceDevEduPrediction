import pandas as pd
import numpy as np
import os
from utils.utils import convert_elapsed_time, del_invar_miss_col

def build_SURF():

    '''
    :return: Appended Pandas df of SURFs
    '''
    path = r'\\pii_zippy\d\USAF PME Board Evaluations\Download 20191211'
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
    surf.drop(columns=['LANG ID 1.1', 'LANG ID 2.1', 'DTY PHONE', 'PERS AEFI.1'], inplace=True)

    surf.drop_duplicates(subset=['SSN', 'Year'], inplace=True)  # drop duplicates within SSN year

    # Convert to datetime format
    dates = [i for i in surf.columns if "DATE" in i or i in ['BDAY', 'DEROS', 'GRADE CURR DOR', 'TAFCSD', 'TAFMSD',
            'TFCSD', 'ETO', 'ODSD']] # convert to datetime format
    for col in dates:
        surf[col] = convert_elapsed_time(surf[col])

    # Drop mostly missing columns & invariant columns
    surf = del_invar_miss_col(surf, view=False)

    # Exclude columns that board does not see
    exclude = [i for i in surf.columns if "PME DATE" in i][:5]+[i for i in surf.columns if "PME METHOD" in i][:5]+\
              [i for i in surf.columns if "ACAD VOC EDUC" in i]+[i for i in surf.columns if "ACAD EDUC METH" in i]
    surf = surf[[i for i in surf.columns if i not in exclude]]

    surf.drop(columns=['PERS AEFI',
                       'AEF START DATE',
                       'ASG ACT NR 1ST ASG',
                       'ASG REPT NLT DATE 1ST ASG',
                       'ASG SELECT DATE 1ST ASG',
                       'PAS 1ST ASG',
                       'ACP ELIG DATE',
                       'ACP EFF DATE',
                       'ACP STOP DATE',
                       'ADSCD 1ST',
                       'ADSCD 2ND',
                       'ADSCD 3RD',
                       'ADSCD 4TH',
                       'ADSCD 5TH',
                       'ADSCD 6TH',
                       'ADSCD RSN FOR 1ST',
                       'ADSCD RSN FOR 2ND',
                       'ADSCD RSN FOR 3RD',
                       'ADSCD RSN FOR 4TH',
                       'ADSCD RSN FOR 5TH',
                       'ADSCD RSN FOR 6TH',
                       'SPOUSE STATUS MIL',
                       'CORE FLAG',
                       'DTY STATUS EFF DATE',
                       'DTY STATUS EXP DATE',
                       'DTY STATUS EFF DATE PROJ',
                       'DTY STATUS EXP DATE PROJ',
                       'SCTY CLEAR ELIG DATE',
                       'DTY POSITION NR',
                       'PGM ELEMENT CODE',
                       'DEPEND',
                       'MARITAL STATUS',
                       'RELIG',
                       'SCTY INV BASIS',
                       'PERS SCTY CLEAR ELIG',
                       'CITIZENSHIP',
                       'ACAD INST NAME 1',
                       'ACAD INST NAME 2',
                       'PRP GRP',
                       'ADSC GRP',
                       'PME 1',
                       'PME 2',
                       'PME 3',
                       'PME 4',
                       'PME 5',
                       'PME GRP 1',
                       'PME GRP 2',
                       'PME GRP 3',
                       'JPME SCHOOL LABEL',
                       'JPME DATE',
                       'JPME SCHOOL',
                       'JPME METHOD',
                       'GENDER',
                       'RACE',
                       'HISP',
                       'PAS',
                       'AUTH ACQ CAREER ',
                       'TOT COMBAT HRS',
                       'RATED POSITION ID',
                       'AS OF DATE',
                       'ACADEMIC SPECIALTY 1ST',
                       'ACADEMIC SPECIALTY 2ND',
                       'TIME IN GRADE (YEARS)',
                       'TIME ON STATN (YEARS)',
                       'PROJECTED DTY LOC',
                       'OFFICE SYMBOL',
                       'ASSN AVAIL CODE GRP',
                       'ASSN LIMIT CODE GRP',
                       'PROJECTED DTY STATUS',
                       'DDA GRP',
                       'NEI GRP',
                       'SEI GEN GRP',
                       'SEI DTY GRP',
                       'AUTH_ACQ_POSN_TYPE_T'], inplace=True)

    return surf