import pandas as pd
import json
from utils.utils import factorize_columns, del_invar_miss_col
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

    df.drop(columns=['PERS AEFI.1'], inplace=True) # merge above oddly duplicates this col
    df = df.loc[:, ~df.columns.duplicated()] # ensure no col name duplicates

    # Change bool values to binary
    df.replace(['True', 'Yes'], 1, inplace=True)
    df.replace(['False', 'No'], 0, inplace=True)

    # Ensure cols <94% missing
    df = del_invar_miss_col(df, view=False)

    # Convert strings to factors and get mappings
    dictionary, f = factorize_columns(df)

    # Output
    with open(r'\\pii_zippy\d\USAF PME Board Evaluations\Processed data\cat_codes.json', 'w') as j:
        json.dump(dictionary, j)

    f.to_csv(r'\\pii_zippy\d\USAF PME Board Evaluations\Processed data\combined_data.csv', index=False)
    df.to_csv(r'\\pii_zippy\d\USAF PME Board Evaluations\Processed data\combined_data_unfactorized.csv', index=False)