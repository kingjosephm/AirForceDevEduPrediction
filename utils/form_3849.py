import pandas as pd
import numpy as np
from utils.utils import del_invar_miss_col
import os

def form_3849():

    path = r'\\pii_zippy\d\USAF PME Board Evaluations\Download 20191122\copies_useful_docs'
    files = [i for i in os.listdir(path) if "3849" in i]

    #####################
    ####    2017    #####
    #####################
    df17 = pd.read_excel(os.path.join(path, files[1]))
    df17 = df17[df17['Grade'] == 'LTC']  # keep only lieutenant coloniels
    df17.drop(columns=['MemberEMail', 'MemberCommPhone', 'DateSrRaterSigned', 'DateMemberSigned', 'DateMemberLogon',
                       'Name', 'ID', 'Grade', 'roleID', 'MembersDSN', 'ReviewerEmail', 'ReviewerPassword',
                       'ReviewerDate', 'SRID', 'SRIDNew', 'SRIDOld', 'MemberCommPhone', 'reviewID', 'DeletedBy',
                       'DeletedDate', 'DAFSCCoreID', 'MemberCommentsYN', 'PrefID1', 'PrefID2', 'PrefID3',
                       'PrefID4', 'PrefID5', 'MemberSignedYN', 'SRSignedYN', 'CandidateYN', 'GTestTotal',
                       'ASGPrefID2', 'ASGPrefID3', 'ASGPrefID4', 'IsDeleted', 'YearGroup', 'YearOfEligibility',
                        'Pref1', 'Pref2', 'Pref3', 'Pref4', 'Pref5', 'Location', 'SRVector', 'SRRationale'], inplace=True)

    # Drop mostly missing and invariant columns
    df17 = del_invar_miss_col(df17, view=False)

    df17['Year'] = 2017

    df17.rename(columns=lambda x: x.strip(), inplace=True) # strip trailing whitespace in col name

    df17.rename(columns={'GTestVerbal': 'GRE Verbal', 'GTestQuant': 'GRE Quant', 'GPAUnder': 'GPA Under',
                         'GPAGrad': 'GPA Grad', 'SRComments': 'SR Cmts', 'MemberComment': 'Mbr Cmts',
                         'NominatedYN': 'Nominated', 'SupervNominatedYN': 'Supv Nominated',
                         'SRReadyYN': 'SR Ready'}, inplace=True)


    #####################
    ####    2018    #####
    #####################
    df18 = pd.read_excel(os.path.join(path, files[2]))
    df18 = df18[df18['Rank'] == 'Lt Col']
    df18.drop(columns=['Level', 'Name', 'Jt Comment', '%#*@', 'Senior Rater ID', 'Spouse SSN', 'Spouse Name',
                       "COMMENTS ARE MANDATORY:\nProvide spouse's info:   (Spouse Rank, Name, component, Component POC: Name, Location DSN, Email)  \ni.e: Maj Brenda Brown, US Army, POC: Capt John Doe, Ft Hood, DSN123-4567, john.doe.mil@mail\n",
                       'Indicate which program in which have been selected for, are attending or completed in addition to the program start date and end date.',
                       'Did you choose the "SDE Deliberate Development for CURRENT or PREVIOUS SDE Equiv Pgm"? ', 'Alternate',
                       'Select/Candidate', 'Status', 'Please choose one of the following options.',
                       'DT', 'Core ID', 'Order of Merit', 'Race', 'Gender', 'Hispanic', 'Look', 'Join Spouse Intent',
                       'Vector Match', 'Schools Match ',  'TAFCSD YR', 'DEVector1', 'DEVector2', 'DEVector3',
                       'PME Pref 1', 'PME Pref 2', 'PME Pref 3', 'PME Pref 4', 'PME Pref 5', 'SR Vector 1', 'SR Vector 2',
                        'SR Vector 3', 'SR Vector 4', 'SR Vector 5', 'Join Spouse', 'DLPT Lang Dt 1', 'DLPT Lang Dt 2', 'RDTM'], inplace=True)

    # Drop mostly missing and invariant columns
    df18 = del_invar_miss_col(df18, view=False)

    df18['Year'] = 2018

    df18.rename(columns=lambda x: x.strip(), inplace=True) # strip trailing whitespace in col name

    df18.rename(columns={'SSAN': 'SSN', 'Additional Comments if desired': 'Other Comments',
                         'Is your Spouse eligible/competing for IDE/SDE this cycle?': 'Spouse compete SDE',
                         'Are you Join Spouse?': 'Join Spouse', 'Mbr DE Comments': 'Mbr Cmts'}, inplace=True)

    df18.replace(['no response.'], np.NaN, inplace=True)

    df18.columns = [i.replace('Stratification', 'Strat').replace('Method', 'Meth').replace('Language', 'Lang')\
                   .replace('Vector', 'Vec').replace('Recent', 'Rec').replace('Comments', 'Cmts')\
                    .replace('Location', 'Loc') for i in df18.columns]

    #####################
    ####    2019    #####
    #####################
    df19 = pd.read_excel(os.path.join(path, files[3]))
    df19 = df19[df19['Rank'] == 'Lt Col']
    df19.drop(columns=['Rank', 'Name',  'Senior Rater ID', 'Did you choose the "SDE Deliberate Development for CURRENT or PREVIOUS SDE Equiv Pgm"? ',
                        'Submission Status', 'Did you choose the "SDE Deliberate Development for CURRENT or PREVIOUS SDE Equiv Pgm"? ',
                        "COMMENTS ARE MANDATORY:  Provide spouse's info:   (Spouse Rank, Name, component, Component POC: Name, Location DSN, Email)  i.e: Maj Brenda Brown, US Army, POC: Capt John Doe, Ft Hood, DSN123-4567, john.doe.mil@mail ",
                        'Indicate which program in which have been selected for, are attending or completed in addition to the program start date and end date. ',
                        'Please choose one of the following options. ', 'Application Updated Date', 'DT Mgt Gp',
                        'Core ID', 'Race', 'Gender', 'Hispanic', 'Look', 'Join Spouse Intent', 'Board DT Mgt Gp', 'Promo Bd Status',
                       'PME Pref 1', 'PME Pref 2', 'PME Pref 3', 'PME Pref 4', 'PME Pref 5', 'SR Vector 1', 'SR Vector 2',
                       'SR Vector 3', 'SR Vector 4', 'SR Vector 5', 'TAFCSD YR', 'Comp Cat Gp', 'Confirmed DE Plan'], inplace=True)

    # Drop mostly missing and invariant columns
    df19 = del_invar_miss_col(df19, view=False)

    df19['Year'] = 2019

    df19.rename(columns=lambda x: x.strip(), inplace=True) # strip trailing whitespace in col name

    df19.rename(columns={'SSAN': 'SSN', 'Additional Comments if desired': 'Other Comments',
                         'Is your Spouse eligible/competing for IDE/SDE this cycle?': 'Spouse compete SDE',
                         'Are you Join Spouse?': 'Join Spouse', 'Mbr DE Comments': 'Mbr Cmts',
                         'Desires to Compete': 'Desire Compete'}, inplace=True)

    df19.replace(['no response.'], np.NaN, inplace=True)

    df19.columns = [i.replace('Stratification', 'Strat').replace('Method', 'Meth').replace('Language', 'Lang')\
                   .replace('Vector', 'Vec').replace('Recent', 'Rec').replace('Comments', 'Cmts')\
                    .replace('Location', 'Loc') for i in df19.columns]

    #######################
    ##### Concatenate #####
    #######################

    # List of all column names across all dfs
    cols = sorted(list(set().union(df17.columns, df18.columns, df19.columns)))

    # Add missing column names to each df - Note: pd.concat cannot handle some duplicate col names and new col names
    df17 = df17.reindex(columns=df17.columns.tolist() + [i for i in cols if i not in df17])
    df18 = df18.reindex(columns=df18.columns.tolist() + [i for i in cols if i not in df18])
    df19 = df19.reindex(columns=df19.columns.tolist() + [i for i in cols if i not in df19])

    df = pd.concat([df17, df18, df19], ignore_index=True, sort=True)

    #############################
    ##### Other Corrections #####
    #############################

    # Nominated or not
    df['Nominated'] = df['Nominated'].replace([True, 'Yes'], 1)
    df['Nominated'] = df['Nominated'].replace([False, 'No'], 0)
    df.loc[df['SR Cmts']=='Not Nominated.', 'Nominated'] = 0

    # If person wants to compete (i.e. go before board)
    df.loc[(df['SR Cmts'].str.contains('not compete|not desire to compete|not desire|no desire', case=False, regex=True)==True) \
           & (df['Desire Compete'].isnull()), 'Desire Compete'] = 0
    df.loc[(df['Mbr Cmts'].str.contains('not compete|not desire to compete|not desire|no desire', case=False, regex=True)==True) \
            & (df['Desire Compete'].isnull()), 'Desire Compete'] = 0
    df['Desire Compete'] = df['Desire Compete'].replace('Yes', 1).replace('No', 0)

    # Senior rater's stratification
    df['temp'] = df['SR Cmts'].str.findall(pat='#[0-9]+\/[0-9]+').str[1].str[1:2] # take numerator of first element of list (in case multiple ratings given)
    df.loc[(df['temp'].notnull()) & (df['SR Strat'].isnull()), 'SR Strat'] = df['temp']
    del df['temp']

    df = del_invar_miss_col(df, view=False) # globally drop cols that are mostly missing

    return df