import pandas as pd
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

    # Change bool values to binary
    df.replace(['True', 'Yes'], 2, inplace=True)
    df.replace(['False', 'No'], 1, inplace=True)

    hrs = [i for i in df.columns if "HRS" in i]
    for col in hrs:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Ensure cols <94% missing
    df = del_invar_miss_col(df, view=True)

    # Output
    df.to_csv(r'\\pii_zippy\d\USAF PME Board Evaluations\Processed data\combined_data_unfactorized.csv', index=False)
    df.to_csv('../data/combined_data_unfactorized.csv', index=False)