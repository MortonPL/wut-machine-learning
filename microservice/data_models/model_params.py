from pydantic import BaseModel


class ModelParams(BaseModel):
    price: float
    session_time: float
    discount: int
    name: str
    category: str
    city: str
