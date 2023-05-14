import pandas as pd
import numpy as np
import json
import sys
from joblib import dump
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import fbeta_score, make_scorer
from ml_models.utils import get_x
from microservice.utils.path_to import path_to, ensure_path
from ml_models.sumodel import SuperModel


def run_advanced_model(search_params=False):
    data = pd.read_json(path_to(__file__, 'data/processed/processed_data.jsonl'), lines=True)

    Y = data.iloc[:, -1].values # type:ignore
    X = get_x(data, 'buy_ended')

    ss_price = StandardScaler()
    ss_time = StandardScaler()
    X['price'] = ss_price.fit_transform(X['price'].values.reshape(-1, 1)) # type:ignore
    X['session_time'] = ss_time.fit_transform(X['session_time'].values.reshape(-1, 1)) # type:ignore

    # best score = 0.48638776833092623
    best_params = {
        'n_estimators': [80],
        'max_depth':[None],
        'min_samples_split':[2],
        'min_samples_leaf': [5],
        'criterion': ["gini"],
        'max_features': [None],
        'bootstrap': [True],
        'random_state': [42],
    }
    param_grid = {
        'n_estimators': [80],
        'max_depth':[None],
        'min_samples_split':[2],
        'criterion': ["gini"],
        'max_features': [None],
        'bootstrap': [True],
        'min_samples_leaf': [5],
        'random_state': [42]
    }
    if not search_params:
        param_grid = best_params

    scorer = make_scorer(fbeta_score, beta=0.5)
    model = RandomForestClassifier(n_jobs=-1)
    gscv = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scorer, n_jobs=-1, refit=True, verbose=3)
    gscv.fit(X, Y)
    results = {"best_params": gscv.best_params_, "best_score": gscv.best_score_}
    print("Best_score:", gscv.best_score_)
    print("Best params:", gscv.best_params_)

    ensure_path('logs')
    with open(path_to(__file__, 'logs/advanced_model_results.jsonl'), 'a') as file:
        json.dump(results, file)

    dump(SuperModel(gscv.best_estimator_, ss_time, ss_price), 'ml_models/advanced_model.joblib')

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        run_advanced_model(sys.argv[1] == 'search')
    else:
        run_advanced_model()
