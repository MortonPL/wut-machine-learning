import pandas as pd
import numpy as np
import json
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import fbeta_score
from ml_models.utils import get_x
from microservice.utils.path_to import path_to, ensure_path
from ml_models.sumodel import SuperModel


def run_basic_model():
    data = pd.read_json(path_to(__file__, 'data/processed/processed_data.jsonl'), lines=True)

    Y = data.iloc[:, -1].values # type:ignore
    X = get_x(data, 'buy_ended')

    ss_price = StandardScaler()
    ss_time = StandardScaler()
    X['price'] = ss_price.fit_transform(X['price'].values.reshape(-1, 1)) # type:ignore
    X['session_time'] = ss_time.fit_transform(X['session_time'].values.reshape(-1, 1)) # type:ignore

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

    model = SGDClassifier(n_jobs=-1)
    model.fit(X_train, Y_train)
    test_prediction = model.predict(X_test)

    f_beta = fbeta_score(Y_test, test_prediction, beta=0.5)

    results = {"f_beta": f_beta}
    print(f"f_beta: {f_beta}")

    ensure_path('logs')
    with open(path_to(__file__, 'logs/basic_model_results.jsonl'), 'a') as file:
        json.dump(results, file)

    dump(SuperModel(model, ss_time, ss_price), 'ml_models/basic_model.joblib')

if __name__ == "__main__":
    run_basic_model()
