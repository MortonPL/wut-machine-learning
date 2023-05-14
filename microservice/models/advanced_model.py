from microservice.models.base_model import BaseModel
from microservice.utils.exporter import Exporter
from microservice.utils.model_reader import ModelReader
from microservice.data_models.model_params import ModelParams


class AdvancedModel(BaseModel):
    def __new__(cls, params: ModelParams):
        if not hasattr(cls, 'instance'):
            cls.instance = super(AdvancedModel, cls).__new__(cls)
        return cls.instance

    def init_once(self):
        self.sumodel = ModelReader.read_model('advanced')

    def __init__(self, params: ModelParams) -> None:
        self.params = params
        super().__init__(self.params)

    def predict(self):
        prediction, params = self.sumodel.predict(self.params)
        Exporter.to_csv(prediction, params, 'advanced') # type:ignore
        return prediction
