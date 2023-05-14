from abc import abstractmethod


class BaseModel:
    def __init__(self, params) -> None:
        self.params = params

    @abstractmethod
    def init_once(self):
        pass

    @abstractmethod
    def predict(self):
        pass
