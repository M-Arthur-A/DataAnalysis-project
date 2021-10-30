import pandas as pd

# paths
path_1 = r'./Initial_dataset/Банкротные.xlsx'
path_2 = r'./Initial_dataset/Большие.xlsx'
path_3 = r'./Initial_dataset/Живые.xlsx'

# constants
years = [2018, 2019, 2020]

def anonimize(_df):
    cols = _df.columns.to_list()
    cols.remove('Наименование')
    cols.remove('Код налогоплательщика')
    return _df[cols]

def numerize_features(_df):
    _df.loc[_df['Статус'] == 'В состоянии банкротства', 'Статус'] = 1
    _df.loc[_df['Статус'] != 1, 'Статус'] = 0
    _df.loc[_df['Сайт в сети Интернет'].isnull(), 'Сайт в сети Интернет'] = 0
    _df.loc[_df['Сайт в сети Интернет'] != 0, 'Сайт в сети Интернет'] = 1
    _df['Размер компании'] = _df['Размер компании'].factorize()[0]
    _df['Вид деятельности/отрасль'] = _df['Вид деятельности/отрасль'].factorize()[0]
    cols  = ['Статус',
            'Сайт в сети Интернет',
            'Возраст компании, лет',
            'ИДО',
            'ИФР',
            'ИПД',
            '2018, Налоги, млн RUB',
            '2019, Налоги, млн RUB',
            '2020, Налоги, млн RUB',
            '2018, Основные средства , млн RUB',
            '2019, Основные средства , млн RUB',
            '2020, Основные средства , млн RUB',
            '2018, Чистые активы, млн RUB',
            '2019, Чистые активы, млн RUB',
            '2020, Чистые активы, млн RUB',
            '2018, Активы  всего, млн RUB',
            '2019, Активы  всего, млн RUB',
            '2020, Активы  всего, млн RUB',
            '2018, Совокупный долг, млн RUB',
            '2019, Совокупный долг, млн RUB',
            '2020, Совокупный долг, млн RUB',
            '2018, Выручка, млн RUB',
            '2019, Выручка, млн RUB',
            '2020, Выручка, млн RUB',
            '2018, Прибыль (убыток) от продажи, млн RUB',
            '2019, Прибыль (убыток) от продажи, млн RUB',
            '2020, Прибыль (убыток) от продажи, млн RUB',
            '2018, Чистая прибыль (убыток), млн RUB',
            '2019, Чистая прибыль (убыток), млн RUB',
            '2020, Чистая прибыль (убыток), млн RUB']

    for col in cols:
        _df[col] = _df[col].astype('float64')

    for year in years:
        _df.loc[_df[f'{year}, Среднесписочная численность работников'].isnull(), f'{year}, Среднесписочная численность работников'] = 0

        _df.loc[_df[f'{year}, Среднесписочная численность работников']\
          .str.contains('-', na=False), \
          f'{year}, Среднесписочная численность работников'] = \
        _df.loc[_df[f'{year}, Среднесписочная численность работников']\
          .str.contains('-', na=False), \
          f'{year}, Среднесписочная численность работников'].str.split(' - ').str[0]

        _df[f'{year}, Среднесписочная численность работников'] = _df[f'{year}, Среднесписочная численность работников'].str.replace(' ', '')

        _df[f'{year}, Среднесписочная численность работников'] = pd.to_numeric(_df[f'{year}, Среднесписочная численность работников'], errors='coerce')
    return _df

df = pd.concat([pd.read_excel(path_1, header=3, dtype=str).iloc[:-2], 
                pd.read_excel(path_2, header=3, dtype=str).iloc[:-2],
                pd.read_excel(path_3, header=3, dtype=str).iloc[:-2]])
df = df.reset_index().iloc[:,2:]
df = anonimize(df)
df = numerize_features(df)

df.dtypes.to_csv('../data/schema.csv', sep='&')
df.to_csv('../data/dataset.csv', sep='&')

