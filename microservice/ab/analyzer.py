import json
import numpy as np
import sys
from microservice.utils.path_to import path_to
from sklearn.metrics import fbeta_score, confusion_matrix

if __name__ == "__main__":
    path = 'logs'
    if len(sys.argv) > 1:
        path = sys.argv[1]
    A_predictions = []
    B_predictions = []
    A_reality = []
    B_reality = []

    with open(path_to(__file__, path + '/ab_truth.jsonl', 2), 'r') as truth_file,\
        open(path_to(__file__, path + '/ab_groups.jsonl', 2), 'r') as group_file:
        for truth, group in zip(truth_file,group_file):
            json_truth = json.loads(truth)
            if json.loads(group)['group'] == 'BASIC':
                A_predictions.append(json_truth['model_class'])
                A_reality.append(json_truth['true_class'])
            else:
                B_predictions.append(json_truth['model_class'])
                B_reality.append(json_truth['true_class'])                

    mA = confusion_matrix(A_reality, A_predictions)
    mB = confusion_matrix(B_reality, B_predictions)

    fA = fbeta_score(A_reality, A_predictions, beta=0.5)
    fB = fbeta_score(B_reality, B_predictions, beta=0.5)
    print("A / Basic Total:", np.sum(mA))
    print("A / Basic Fbeta:", fA)
    print("A / Basic Matrix:")
    print(mA)
    print("B / Advanced Total:", np.sum(mB))
    print("B / Advanced Fbeta:", fB)
    print("B / Advanced Matrix:")
    print(mB)
