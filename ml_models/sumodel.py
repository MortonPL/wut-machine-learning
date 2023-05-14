from ml_models.process_data import process_category, process_city, process_discount, process_sex
from pandas import DataFrame

class SuperModel:
    def __init__(self, model, time_scaler, price_scaler):
        self.model = model
        self.time_scaler = time_scaler
        self.price_scaler = price_scaler

    def predict(self, params):
        dict_ = {
            'price': self.price_scaler.transform([[params.price]])[0, 0],
            'session_time': self.time_scaler.transform([[params.session_time]])[0, 0],
            'discount': process_discount(params.discount),
            'sex': process_sex(params.name)
        }
        fh_category = process_category(params.category)[0]
        for i, v in enumerate(fh_category):
            dict_[f'c{i}'] = v
        fh_city = process_city(params.city)[0]
        for i, v in enumerate(fh_city):
            dict_[f'p{i}'] = v
        ready_params = DataFrame(data=dict_, index=[0])
        return int(self.model.predict(ready_params)[0]), ready_params
