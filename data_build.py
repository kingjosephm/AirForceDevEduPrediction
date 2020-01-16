import pandas as pd
import numpy as np
from utils.utils import del_invar_miss_col
from utils.SURF import build_SURF
from utils.SDE import build_SDE
from utils.decor import build_decor
from utils.form_3849 import form_3849
pd.set_option('display.max_rows', 600)
pd.set_option('display.max_columns', 550)


if __name__ == '__main__':

    # Get SDE reports - main board results
    sde = build_SDE()

    # Get SURFs
    surf = build_SURF()

    # Get decorations
    decor = build_decor()

    # Form 3849
    f3849 = form_3849()

    # Merge together
    df = pd.DataFrame()
    df = sde.merge(surf, on=['SSN', 'Year'], how='left', copy=False)
    df = df.merge(f3849, on=['SSN', 'Year'], how='left', copy=False)
    df = df.merge(decor, on=['SSN', 'Year'], how='left', copy=False)

    df = df.loc[:, ~df.columns.duplicated()] # ensure no col name duplicates

    # Features for similiarity between board member and applicant
    for col in [i for i in df.columns if "Gender_" in i]:
        temp = df[col]==df['GENDER']
        df[col] = np.where(df[col].notnull(), temp, np.NaN)
    for col in [i for i in df.columns if "Race_" in i]:
        temp = df[col]==df['RACE']
        df[col] = np.where(df[col].notnull(), temp, np.NaN)
    for col in [i for i in df.columns if "Hisp_" in i]:
        temp = df[col]==df['HISP']
        df[col] = np.where(df[col].notnull(), temp, np.NaN)
    for col in [i for i in df.columns if "Afsc_" in i]:
        temp = df[col]==df['AFSC']
        df[col] = np.where(df[col].notnull(), temp, np.NaN)

    # Change bool values to binary
    df.replace(['True', 'Yes'], 1, inplace=True)
    df.replace(['False', 'No'], 0, inplace=True)

    # Ensure cols <94% missing
    df = del_invar_miss_col(df, view=True)

    # Output
    df.to_csv(r'\\pii_zippy\d\USAF PME Board Evaluations\Processed data\combined_data_unfactorized.csv', index=False)