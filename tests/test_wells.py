"""Kuyu verisi yükleme testleri."""
import numpy as np
import pandas as pd
import pytest

from geoai.io.loaders import load_well_data


def _write(tmp_path, text, name='w.csv'):
    p = tmp_path / name
    p.write_text(text, encoding='utf-8')
    return str(p)


README_CSV = """X,Y,LABEL,TARGET_TYPE,DEPTH,NOTES
312000,4012000,1,mineral,150,Mineralized zone
306000,4010000,1,groundwater,80,Aquifer detected
320000,4018000,0,mineral,,Barren drillhole
"""


def test_load_basic_with_optional_columns(tmp_path):
    df = load_well_data(_write(tmp_path, README_CSV))
    assert list(df[['x', 'y', 'label']].itertuples(index=False, name=None))[0] == (312000, 4012000, 1)
    assert df['target_type'].tolist() == ['mineral', 'groundwater', 'mineral']
    assert 'depth' in df and 'notes' in df
    assert df['label'].dtype.kind == 'i'


def test_target_type_filter(tmp_path):
    df = load_well_data(_write(tmp_path, README_CSV), target_type='mineral')
    assert len(df) == 2 and set(df['target_type']) == {'mineral'}
    df = load_well_data(_write(tmp_path, README_CSV), target_type='Groundwater')
    assert len(df) == 1 and df['label'].tolist() == [1]


def test_label_column_not_confused_with_target_type(tmp_path):
    """TARGET_TYPE kolonu varken etiket 'HIT' gibi farklı isimde olabilir."""
    text = "Easting,Northing,TARGET_TYPE,HIT\n1,2,mineral,1\n3,4,mineral,0\n"
    df = load_well_data(_write(tmp_path, text))
    assert df['label'].tolist() == [1, 0]
    assert df['target_type'].tolist() == ['mineral', 'mineral']


def test_mineralized_label_not_taken_as_type(tmp_path):
    text = "x,y,mineralized\n1,2,yes\n3,4,no\n5,6,Evet\n"
    df = load_well_data(_write(tmp_path, text))
    assert df['label'].tolist() == [1, 0, 1]
    assert 'target_type' not in df


def test_invalid_labels_raise(tmp_path):
    with pytest.raises(ValueError):
        load_well_data(_write(tmp_path, "X,Y,LABEL\n1,1,2\n2,2,1\n"))
    with pytest.raises(ValueError):
        load_well_data(_write(tmp_path, "X,Y,LABEL\n1,1,maybe\n2,2,1\n"))


def test_missing_rows_are_dropped_with_warning(tmp_path):
    text = "X,Y,LABEL\n1,1,1\n,2,0\n3,3,\n4,4,0\n"
    with pytest.warns(UserWarning, match='2 satır'):
        df = load_well_data(_write(tmp_path, text))
    assert len(df) == 2


def test_semicolon_and_well_id(tmp_path):
    text = "HOLE_ID;X;Y;RESULT\nDH-1;10;20;1\nDH-2;30;40;0\n"
    df = load_well_data(_write(tmp_path, text))
    assert df['well_id'].tolist() == ['DH-1', 'DH-2']
    assert df['label'].tolist() == [1, 0]


def test_explicit_columns(tmp_path):
    text = "a,b,c,d\n1,2,0,9\n3,4,1,9\n"
    df = load_well_data(_write(tmp_path, text), x_col='b', y_col='a', label_col='c')
    np.testing.assert_array_equal(df['x'], [2, 4])
    np.testing.assert_array_equal(df['y'], [1, 3])
    with pytest.raises(KeyError):
        load_well_data(_write(tmp_path, text), label_col='nope')


def test_returns_dataframe_index_reset(tmp_path):
    text = "X,Y,LABEL,TARGET_TYPE\n1,1,1,water\n2,2,0,mineral\n3,3,1,mineral\n"
    df = load_well_data(_write(tmp_path, text), target_type='mineral')
    assert isinstance(df, pd.DataFrame)
    assert df.index.tolist() == [0, 1]
