import pandas as pd
import numpy as np
from keras import backend as K

def convert_elapsed_time(col):
    '''
    Function converts date column (either 8- or 4-digits) to pd datetime format, then
    to elapsed days since Unix date
    :col: pandas series
    Returns pandas series in elapsed days since Unix date
    '''
    # Convert to standard pandas datetime
    test = pd.to_datetime(col, format='%Y%m%d', errors='coerce') # YYYYMMDD
    if test.isnull().mean() == 1: # wrong format
        test = pd.to_datetime(col, format='%y%m', errors='coerce') # YYMM
        if test.isnull().mean() == 1:
            test = pd.to_datetime(col, format='%d%b%y', errors='coerce') # e.g. 30Jan10
            if test.isnull().mean() == 1:
                raise ValueError("\nDate formatting of {} unclear".format(col))
            else:
                col = test
        else:
            col = test
    else:
        col = test

    # Convert to days since Unix date
    col = (col - pd.to_datetime('1970-01-01', format='%Y-%m-%d')).astype('timedelta64[D]')
    return col

def del_invar_miss_col(df, thresh=0.95, view=False):
    '''
    Function drops columns more than 95% missing and invariant columns.
    :df: pandas data frame
    :thresh: float, proportion of column missing
    :view: bool, if true prints col name deleted
    Returns dataframe minus mostly missing or invariant columns.
    '''
    for col in df.columns:

        df[col].replace('', np.NaN, inplace=True)

        if df[col].isnull().mean() > thresh:
            if view:
                print(col, 'was >{}% missing and deleted'.format(thresh))
            del df[col]
        else: # verify not invariant if not mostly missing
            if len(df[df[col].notnull()][col].unique()) == 1:
                if view:
                    print(col, 'was invariant and deleted')
                del df[col]
    return df

def factorize_columns(df):
    '''
    Converts all object-type columns in pandas dataframe to intergers.
    :df: pandas dataframe
    returns: dictionary of integer mappings back to original strings and modified dataframe
    '''

    dictionary = {} # initialize dictionary for factor mappings

    cat_cols = [i for i in list(df.select_dtypes(include=['object']).columns) if
                'Candidate' not in i and 'Mbr Cmts' not in i and 'SR Cmts' not in i]
    for col in cat_cols:
        num_codes = df[col].astype('category').cat.codes.drop_duplicates().to_list() # NaN is -1 by default
        num_codes = [0 if i==-1 else i for i in num_codes] # replace missing (-1) with 0
        orig_codes = df[col].drop_duplicates().to_list()
        dictionary[col] = dict(zip(num_codes, orig_codes))
        df[col] = df[col].astype('category').cat.codes

    df = df.replace([np.NaN, -1], 0) # replace all missing values with 0
    return dictionary, df

def high_dimension(dictionary, nr=800):
    '''
    Returns list of high dimensional elements (keys) of a dictionary, where high dimensional specified by parameter 'nr'.
    :dictionary: dictionary
    :nr: int, number of unique values above which a particular dictionary element is considered high dimensional
    '''

    high_dim = []
    if nr != None:
        for key, values in dictionary.items():
            if len(dictionary[key]) > nr:
                high_dim.append(key)
    return high_dim

def r_square(y_true, y_pred):
    '''
    Computes coefficient of determination (R2) for train and valid sets
    :param y_true: observed y_value
    :param y_pred: predicted y_value
    :return: coefficient of determination for Keras deep learning model
    '''
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))