import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
import dill

# Утилиты
def col_name(short_name, _df):
    return [name for name in _df.columns if short_name in name]

# Основыне функции
def add_features(_df):
    """
    Добавление фичей:
    - ['O'] OIBDA (на основе "гипотетической амортизации" = ОС х (100% / СПИ)), СПИ = 10 лет
    - ['D-O'] Совокупный долг / OIBDA
    - ['O-R'] OIBDA / Выручка
    - ['D-R'] Совокупный долг / Выручка
    - ['E-A'] Основные средства / Активы всего
    """
    for year in ['cur', 'prev']:
        col = f'{year}_'
        _df[col+'O']   = _df[col+'Прибыль (убыток) от продажи'] - _df[col+'Основные средства ']*0.1
        _df[col+'D-0'] = _df[col+'Совокупный долг']   / _df[col+'O']
        _df[col+'O-R'] = _df[col+'O']                 / _df[col+'Выручка']
        _df[col+'D-R'] = _df[col+'Совокупный долг']   / _df[col+'Выручка']
        _df[col+'E-A'] = _df[col+'Основные средства '] / _df[col+'Активы  всего']

    _df = _df.replace([np.inf, -np.inf], np.nan)
    print('DONE - features created.')
    return _df

def grid_search_cv(_X_train, _y_train):
    xgb_model = xgb.XGBClassifier()
    xgb_params = {'nthread':[4, 6], #when use hyperthread, xgboost may become slower
                  'objective':['reg:squarederror'],
                  'learning_rate': [0.05, 0.03], # `eta` value
                  'max_depth': [3, 6, 12],
                  'min_child_weight': [3, 5, 11], # fighting against overfit
                  'subsample': [0.8],
                  'colsample_bytree': [0.7],
                  'n_estimators': [5, 10, 500, 1000], #number of trees, change it to 1000 for better results
                  'missing':[-999],
                  'seed': [46]}

    grid_search = GridSearchCV(xgb_model,
                              xgb_params,
                              cv = 2,
                              n_jobs = 5,
                              verbose=True)

    grid_search.fit(_X_train, _y_train)
    return grid_search.best_score_, grid_search.best_params_

def df_split(_df):
    y_train = _df['Статус']
    X_train = _df[[item for item in _df.columns.tolist() if item != 'Статус']]
    # доля банкротных компаний в каждой выборке составляет около 31%
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, shuffle=True, random_state=42)
    print('DONE df splitted.', y_train.value_counts()[1]/y_train.value_counts()[0], y_test.value_counts()[1]/y_test.value_counts()[0])
    return X_train, X_test, y_train, y_test

def train_model(_X_train, _y_train):
    xgb_model = xgb.XGBClassifier()
    xgb_params = {'colsample_bytree': 0.7,
                  'learning_rate': 0.05,
                  'max_depth': 6,
                  'min_child_weight': 3,
                  'missing': -999,
                  'n_estimators': 1000,
                  'nthread': 4,
                  'objective': 'reg:squarederror',
                  'seed': 46,
                  'subsample': 0.8}
    xgb_model.set_params(**xgb_params)
    xgb_model.fit(_X_train, _y_train)
    print('DONE - model created.')
    return xgb_model

def model_backup(_model):
    with open('./model/model.pkl', 'wb') as file:
        dill.dump(_model, file)
    print('DONE - model saved.')

def model_load():
    with open('./model/model.pkl', 'rb') as m:
        return dill.load(m)

if __name__ == '__main__':
     df = pd.read_csv(path_dataset, sep='&')
     df = add_features(df)
     X_train, X_test, y_train, y_test = df_split(df)
     model = train_model(X_train, y_train)
     model_backup(model)
     answers = model.predict_proba(X_test)[:,1]
     print(r2_score(y_test, answers))
