import json
import pandas as pd
import numpy as np
import os


class Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(Encoder, self).default(obj)


class Exporter:
    @staticmethod
    def to_csv(prediction: int, params: pd.DataFrame, model_version='basic'):
        os.makedirs('logs', exist_ok=True)
        if model_version == 'basic':
            filename = 'logs/prediction_basic_model.jsonl'
        else:
            filename = 'logs/prediction_advanced_model.jsonl'

        result = {}
        for col, val in zip(list(params.columns), list(params.iloc[0])):
            result[col] = val
        result['product_bought'] = prediction
        with open(filename, 'a') as csvfile:
            csvfile.write(json.dumps(result, cls=Encoder))
            csvfile.write('\n')
