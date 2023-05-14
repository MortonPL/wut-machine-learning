import joblib
from microservice.utils.path_to import path_to


class ModelReader:

    @staticmethod
    def read_model(model='basic'):
        if model == 'basic':
            path_ = path_to(__file__, 'taught_models/basic_model.joblib', 2)
            return joblib.load(path_, mmap_mode='r')
        path_ = path_to(__file__, 'taught_models/advanced_model.joblib', 2)
        return joblib.load(path_)
