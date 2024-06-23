import pytest
import pandas as pd
from functions import load_data, clean_data

def test_load_data():
    df = load_data('data/title.basics.tsv.gz')
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

def test_clean_data():
    df = pd.DataFrame({
        'col1': ['a', '\\N', 'c'],
        'col2': [1, 2, '\\N']
    })
    cleaned_df = clean_data(df)
    assert cleaned_df.isnull().sum().sum() == 2
    assert cleaned_df.loc[1, 'col1'] is pd.NA
    assert cleaned_df.loc[2, 'col2'] is pd.NA
