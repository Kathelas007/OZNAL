import os
import pandas as pd
import zipfile
import rarfile, csv
import logging
import io
import numpy as np
import logging

# 1 rok 107 000 zaznamu
# 8000 na jeden kraj

cache_file_template = './data/accidents_{}.pkl'

date_column = [3]
string_columns = [64, 5, 51, 52, 53, 54, 55, 56, 57, 58, 59, 62]
int_columns = [0, 1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
               29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 60, 61, 63]
float_columns = [45, 46, 47, 48, 49, 50]

columns_oder = date_column + string_columns + int_columns + float_columns

column_original_names = [
    'p1', 'p36', 'p37', 'p2a', 'weekday(p2a)', 'p2b', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12', 'p13a',  # 14
    'p13b', 'p13c', 'p14', 'p15', 'p16', 'p17', 'p18', 'p19', 'p20', 'p21', 'p22', 'p23', 'p24', 'p27',
    'p28', 'p34', 'p35', 'p39', 'p44', 'p45a', 'p47', 'p48a', 'p49', 'p50a', 'p50b', 'p51', 'p52', 'p53',
    'p55a', 'p57', 'p58', 'a', 'b', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'n', 'o', 'p', 'q', 'r',
    's', 't', 'p5a']

column_description = ['date', 'reg', 'cas', 'h', 'i', 'j', 'k', 'l', 'n', 'o', 'p', 'q', 't', 'id',
                      'druh pozemni komunikace', 'cislo pozemni komunikace', 'den v tydnu', 'druh nehody',
                      'druh strazky s jedoucim vozidlem', 'druh pevne prekasky', 'charakter nehody',
                      'zavineni nehody', 'alkohol vinika', 'hlavni pricina', 'usmrceno', 'tezce zraneno',
                      'lehce zraneno', 'skoda', 'druh povrchu', 'stav povrchu', 'stav komunikace',
                      'povetrnostni podminky', 'viditelnost', 'rozhledove podminky', 'deleni komunikace',
                      'situovani nehody', 'rizeni provozu', 'mistni uprava prednosti', 'spec mista a objekty',
                      'smerove pomery', 'pocet zucatnenych vozidel', 'misto dopravni nehody',
                      'druh krizujici komunikace', 'druh vozidla', 'znacka mot vozidla', 'rok vyroby vozidla',
                      'charakter vozidla', 'smyk', 'vozidlo po nehode', 'unik hmot', 'zpusob vyprosteni',
                      'smer jizdy nebo pozastaveni voz', 'skoda na vozidle', 'kat ridice', 'stav ridice',
                      'vnejso ovlivneni', 'r', 's', 'lokalita nehody', 'a', 'b', 'd', 'e', 'f', 'g']

region_codes = {
    '00': 'PHA',
    '01': 'STC',
    '02': 'JHC',
    '03': 'PLK',
    '04': 'ULK',
    '05': 'HKK',
    '06': 'JHM',
    '07': 'MSK',
    '14': 'OLK',
    '15': 'ZLK',
    '16': 'VYS',
    '17': 'PAK',
    '18': 'LBK',
    '19': 'KVK',
}


# co je j, o

# ********** CLEANING ********************************************************


def hours_min_to_td(cas: str):
    if len(cas) != 4:
        logging.warning(f'Unknown time value {cas}. Interpreting as Nan')
        return np.nan

    hours = int(cas[:2])
    minutes = int(cas[2:])

    if hours == 25 or minutes == 60:
        return np.nan

    return np.timedelta64(hours, 'h') + np.timedelta64(minutes, 'm')


def clean_data(df: pd.DataFrame):
    # XX is unknown manufacture date
    df.loc[:, 'p47'] = df.loc[:, 'p47'].replace({'XX': np.nan, '': np.nan})

    # empty values to Nan
    df.iloc[:, int_columns] = df.iloc[:, int_columns].replace({'': np.nan})
    df.iloc[:, float_columns] = df.iloc[:, float_columns].replace({'': np.nan})

    # make float parseable
    df.iloc[:, float_columns] = df.iloc[:, float_columns].astype('str')
    for float_col in float_columns:
        df.iloc[:, float_col] = df.iloc[:, float_col].str.replace(',', '.')

    df.iloc[:, string_columns] = df.iloc[:, string_columns].astype(str)
    try:
        df.iloc[:, float_columns] = df.iloc[:, float_columns].astype(np.float64)
    except ValueError:
        # inconsistenci in 2018 dataset
        flt_res = {f_c: pd.to_numeric(df.iloc[:, f_c], errors='coerce') for f_c in float_columns}
        for name, res in flt_res.items():
            df.iloc[:, name] = res

    df.iloc[:, int_columns] = df.iloc[:, int_columns].astype('Int64')  # nullable int

    # set descriptive column names
    df = df.iloc[:, columns_oder]
    df.columns = column_description
    df = df.sort_values('date', axis=0)
    df = df.reset_index(drop=True)

    # set date times
    df['date'] = df['date'].astype(np.datetime64)
    df['time'] = df['cas'].apply(hours_min_to_td)

    return df


# ********** CACHING ********************************************************

def csv_to_df(rf, rf_csv_file):
    csv_desc = rf.open(rf_csv_file)
    csv_reader = csv.reader(io.TextIOWrapper(csv_desc, 'windows-1250'), delimiter=';', quotechar='"')

    df = pd.DataFrame(csv_reader, columns=column_original_names)
    csv_desc.close()

    df['reg'] = region_codes[rf_csv_file.filename[:2]]
    return df


def rar_to_df(archive_name):
    if not os.path.exists:
        logging.warning(f"Can not find {archive_name}. Skipping this dataset.")
        return pd.DataFrame()

    with rarfile.RarFile(archive_name) as rf:
        rf_csv_files = [f for f in rf.infolist()
                        if len(f.filename) == 6
                        and f.filename.endswith('.csv')
                        and f.filename[:2] in region_codes.keys()]

        dfs = [csv_to_df(rf, rf_csv) for rf_csv in rf_csv_files]

    return pd.concat(dfs)


def cache_data(year):
    archive_name = f'./data/data_GIS_{year}.rar'
    df = rar_to_df(archive_name)
    df = clean_data(df)

    df.to_pickle(cache_file_template.format(year))

    return df


# ******* LOAD *************************************************************

def load_data(start_year=2019, end_year=2021):
    years = list(range(start_year, end_year + 1))
    cached_years = list(filter(lambda x: os.path.exists(cache_file_template.format(x)), years))
    non_cached_years = list(filter(lambda x: not os.path.exists(cache_file_template.format(x)), years))

    new_df = [cache_data(year) for year in non_cached_years]
    old_df = [pd.read_pickle(cache_file_template.format(year)) for year in cached_years]
    dfs = new_df + old_df

    return pd.concat(dfs)


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    df = load_data(2018, 2018)
    print(df.head())
