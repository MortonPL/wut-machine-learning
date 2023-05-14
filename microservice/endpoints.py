from fastapi import FastAPI, Cookie, Response
from typing import Optional
from uuid import UUID, uuid4
from random import random
import json
from microservice.configuration.about import TITLE, VERSION, DESCRIPTION
from microservice.models.basic_model import BasicModel
from microservice.models.advanced_model import AdvancedModel
from microservice.data_models.model_params import ModelParams
from microservice.utils.path_to import path_to, ensure_path

app = FastAPI(title=TITLE, description=DESCRIPTION, version=VERSION)
user_dict = {}

CLIENT_SESSION = 365 * 24 * 60 * 60

# Singletons
bm = BasicModel(None) # type:ignore
bm.init_once()
am = AdvancedModel(None) # type:ignore
am.init_once()

@app.get("/", tags=["About"])
def project_information():
    return {"Endpoints:":
           ["Basic model: /model/basic",
           "Advanced model: /model/advanced",
           "AB test: /model"]}

@app.get("/model/basic", tags=["BasicModel"])
def basic_model_prediction(model_params: ModelParams):
    model = BasicModel(model_params)
    prediction = model.predict()
    return {'class':prediction}

@app.get("/model/advanced", tags=["AdvancedModel"])
def advanced_model_prediction(model_params: ModelParams):
    model = AdvancedModel(model_params)
    prediction = model.predict()
    return {'class':prediction}

@app.get("/model", tags=["ABTest"])
def ab_model_prediction(model_params: ModelParams, response: Response, user_id: Optional[UUID]=Cookie(None)):
    abstr = None
    if user_id is None or user_id not in user_dict:
        user_id = uuid4()
        ab = random() <= 0.5
        user_dict[user_id] = ab
        abstr = "ADVANCED" if ab else "BASIC"
        print("New user connected, delegating to the", abstr, "model.")
    else:
        ab = user_dict[user_id]
        abstr = "ADVANCED" if ab else "BASIC"

    response.set_cookie("user_id", str(user_id), max_age=CLIENT_SESSION, httponly=True, samesite="none")
    
    ensure_path('logs')
    with open(path_to(__file__, 'logs/ab_groups.jsonl'), 'a') as file:
        file.write(json.dumps({'group': abstr}))
        file.write('\n')

    model = AdvancedModel(model_params) if ab else BasicModel(model_params)
    prediction = model.predict()
    return {'class':prediction}
