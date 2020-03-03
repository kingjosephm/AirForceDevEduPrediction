import pandas as pd
import numpy as np
from keras import backend as K
import json
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

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

    # Convert to days since 1900
    col = (col - pd.to_datetime('1900-01-01', format='%Y-%m-%d')).astype('timedelta64[D]')
    col = col + 1 # ensures no zero-duration dates; zero used as missing
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

    cat_cols = [i for i in list(df.select_dtypes(include=['object']).columns)]
    for col in cat_cols:
        df[col] = df[col].replace(r'^\s*$', np.NaN, regex=True)
        num_codes = df[col].astype('category').cat.codes.drop_duplicates().to_list() # NaN is -1 by default
        num_codes = [i+1 for i in num_codes] # shifts zero-based index to one-based index, with zero as missing
        orig_codes = df[col].drop_duplicates().to_list()
        dictionary[col] = dict(zip(num_codes, orig_codes))
        df[col] = df[col].astype('category').cat.codes +1

    df = df.replace([np.NaN], 0) # replace all missing values with 0
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

def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []

    for text in texts:
        try:
            text = text.replace('\t+|\r+', ' ')
        except AttributeError: # some rows are missing (np.NaN) or have a number
            text = ''
        text = tokenizer.tokenize(text)
        text = text[:max_len - 2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)

        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len

        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)


def read_data():
    '''
    Loads appended CSV dataset and json dictionary of factorized variable categorical codes, creates X dataframe
    that excludes high-dimensional and other columns.
    Returns:
        df_fac - dataframe of factorized categorical features (excluding string features) and numeric features
        categorical_mappings - dictionary of  mappings for each categorical feature in dataframe
        config file
    '''
    # Load unfactorized data
    df = pd.read_csv('../data/combined_data_unfactorized.csv', low_memory=False)

    # Get config
    with open('config.json') as j:
        config = json.load(j)

    # Remove excluded features
    if config['excluded_features']:
       df = df[[i for i in df.columns if i not in config['excluded_features']]]

    # Convert strings to factors and get mappings
    if config['string_features']:
        categorical_mappings, df_fac = factorize_columns(df[[i for i in df.columns if i not in config['string_features']]])
        df_fac = pd.concat([df_fac, df[config['string_features']]], axis=1) # reattach original string column(s)
    else:
        categorical_mappings, df_fac = factorize_columns(df)

    # Remove high-dimensional categorical features
    high_dim = high_dimension(categorical_mappings, nr=config['max_feature_categories'])  # exclude categorical features with > 'max_feature_categories' unique categories
    for x in high_dim:
        del df_fac[x], categorical_mappings[x]

    return df_fac, categorical_mappings, config

def get_vif():
    '''
    Calculates Variance Inflation Factor for each feature in a dataframe.
    :return: Pandas dataframe of VIF scores for each feature.
    '''
    df, categorical_mappings, config = read_data()
    y = df[config['outcome_feature']]
    X = df[[i for i in df.columns if i not in config['outcome_feature']]]
    X = sm.add_constant(X)
    sm.OLS(y, X).fit().summary()
    vif_scores = [vif(X.values, i) for i in range(X.shape[1])]
    return pd.concat([pd.Series(X.columns), pd.Series(vif_scores)], axis=1).rename(columns={0: 'column', 1: 'vif'})