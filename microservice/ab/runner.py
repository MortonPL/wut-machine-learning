import requests
import random
import sys
import json
from microservice.utils.path_to import path_to, ensure_path
from ml_models.process_data import N_FEATURES_PROVINCE
from ml_models.process_data import N_FEATURES_CATEGORY

def get_params(set):
    line = random.choice(set)
    json_: dict = json.loads(line)
    class_ = json_.pop('buy_ended')
    params = json_
    return params, class_


if __name__ == "__main__":
    users = 0

    try:
        users = int(sys.argv[1])
    except ValueError:
        print("Invalid user count!")

    set = list(open(path_to(__file__, "data/processed/client_data.jsonl", 2)))

    for u in range(users):
        params, class_ = get_params(set)
        res = requests.get("http://127.0.0.1:8000/model", data=json.dumps(params))

        result = {'model_class': json.loads(res.text)['class'], 'true_class': class_}
        ensure_path('logs')
        with open(path_to(__file__, 'logs/ab_truth.jsonl', 2), 'a') as file:
            file.write(json.dumps(result))
            file.write('\n')
        print(u)
        print("\u001b[1A", end='')

    print("Done")
