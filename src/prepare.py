import pandas as pd

# paths
path_1 = r'./Initial_dataset/Банкротные.xlsx'
path_2 = r'./Initial_dataset/Большие.xlsx'
path_3 = r'./Initial_dataset/Живые.xlsx'
path_4 = r'./Initial_dataset/Банкроты_add.xlsx'

# constants
years = [2018, 2019, 2020]
years_b = [2013, 2014, 2015, 2016, 2017]

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

def get_bankruptsy_date(_df):
    def bdate_from_string(infos):
        # проверки в порядке важности вхождения
        check1 = ['Решение о признании должника банкротом',
                  'Юридическое лицо признано несостоятельным (банкротом)']
        check2 = ['наблюдение',
                  'наблюдении',
                  'наблюдения']
        check3 = ['внешнего управления',
                  'внешнее управление']
        check4 = ['о возобновлении производства по делу о несостоятельности',
                  'возбуждено производство']
        check5 = ['оздоровления',
                  'оздоровление']
        check6 = ['заявлением о банкротстве']
        if type(infos) == float:
            return 'NaN'
        for check in [check1, check2, check3, check4, check5, check6]:
            for mes in str(infos).split(', '):
                if any(ext in mes for ext in check):
                    return mes.split(' от ')[-1]
        # если эта графа заполнена совсем плохо - берем хотя бы дату ареста счетов ФНС
        for mes in str(infos).split(', '):
            if 'решения ФНС' in mes:
                return mes.split()[1]
        return 'Нет решения'

    _df['b_date'] = _df['Важная информация'].apply(bdate_from_string)
    print('Дата банкротства получена')
    return _df

def choose_bunkruptsy_financials(_df):
    def get_cols_by_year(year: int) -> list:
        col_financials = []
        for col in _df.columns.tolist():
            if ',' in col:
                col_year = col.split(',')[0]
                if str(col_year) == str(year):
                    col_financials.append(col)
        return col_financials

    # выбор финансовых данных за 2 года до банкротства
    _df['b_year'] = _df['b_date'].str.extract(r'(\d{4})')
    _df.loc[_df['b_year'].isnull(), 'b_year'] = 2013
    _df['b_year_threshold'] = _df['b_year'].astype('int16')-2
    _df.loc[_df['b_year_threshold']<2013, 'b_year_threshold'] = 2013
    # удаление "старых" банкротств, где не будет совсем никакой динамики
    _df = _df.drop(_df.loc[(_df['b_date'].notnull()) & (_df['b_year_threshold'] == 2013)].index)
    # добавление current и previos отчетности за 2 года до банкротства
    bankrupts_filter = _df['b_date'].notnull()
    thresholds = _df.loc[bankrupts_filter, 'b_year_threshold'].value_counts().index.tolist()
    for year in thresholds:
        year_filter = _df['b_year_threshold'] == year
        for col in get_cols_by_year(year):
            _df.loc[bankrupts_filter & year_filter, 'cur_' + col.split(', ')[1]] = _df.loc[bankrupts_filter & year_filter, col]
        for col in get_cols_by_year(year-1):
            _df.loc[bankrupts_filter & year_filter, 'prev_'+ col.split(', ')[1]] = _df.loc[bankrupts_filter & year_filter, col]
    return _df

def choose_financials(_df, training=False, years=years):
    def get_cols_by_year(year: int) -> list:
        col_financials = []
        for col in _df.columns.tolist():
            if ',' in col:
                col_year = col.split(',')[0]
                if str(col_year) == str(year):
                    col_financials.append(col)
        return col_financials
    if training:
        bankrupts_filter = _df['b_date'].notnull()
        filter_df = ~bankrupts_filter    # только живые компании
    else:
        filter_df = _df.index.notnull()  # все компании
    for year in years:
        for col in get_cols_by_year(year):
            _df.loc[filter_df, 'cur_' + col.split(', ')[1]] = _df.loc[filter_df, col]
        for col in get_cols_by_year(year-1):
            _df.loc[filter_df, 'prev_'+ col.split(', ')[1]] = _df.loc[filter_df, col]
    return _df

def prepare_train_dataset():
    df = pd.concat([pd.read_excel(path_1, header=3, dtype=str).iloc[:-2], 
                    pd.read_excel(path_2, header=3, dtype=str).iloc[:-2],
                    pd.read_excel(path_3, header=3, dtype=str).iloc[:-2]])
    df = df.reset_index().iloc[:,2:]
    b_df = pd.read_excel(path_4, header=3, dtype=str).iloc[:-2]
    b_df = get_bankruptsy_date(_df)
    cols_to_merge = ['Код налогоплательщика'] + b_df.columns.difference(df.columns).tolist()
    df = df.merge(b_df[cols_to_merge], on='Код налогоплательщика', how='left')
    df = choose_bunkruptsy_financials(df)
    df = choose_financials(df, training=True)
    df = anonimize(df)
    df = numerize_features(df)

if __name__ == '__main__':
    df = prepare_train_dataset()
    # обрезка ненужных колонок
    cols = df.columns.tolist()
    cols[1] = cols[1].replace(', лет', '')
    df.columns = cols
    cols_to_save = []
    check = [', ', 'b', '№', 'ИДО', 'ИФР', 'ИПД', 'Регистрационный номер', 'Мои списки', 'Реестры СПАРКа', 'Важная информация']
    for c in cols:
        if any(ext in c for ext in check):
            continue
        cols_to_save.append(c)
    df = df[cols_to_save]

    # сохранение схемы данных
    df.dtypes.to_csv('../data/schema.csv', sep='&')

    # сохранение самого датасета
    df.to_csv('../data/dataset.csv', sep='&', index=False)
