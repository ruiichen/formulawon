import pandas as pd
import torch
import torch.nn.functional as F
from util.scaler import get_scaler
from services.quali_service import get_quali_session

def predict_winner_from_quali(season, round, model, data):
    model.eval()

    scaler = get_scaler(data)
    df = get_quali_session(season, round)

    conf = 1
    guess = 0
    for grid in range(20):
        X_value = df[(df['grid'] == grid + 1)]
        X_value = X_value.drop(['driver', 'country', 'url'], axis=1)
        X_value = pd.DataFrame(scaler.transform(X_value), columns=X_value.columns)
        X_value = torch.Tensor(X_value.to_numpy())

        prediction = model(X_value)
        prob = F.softmax(prediction, dim=1)
        prob, dummy = prob.topk(1, dim=1)

        if prediction.argmax().item():
            return grid + 1
        elif prob[0][0] < conf:
            conf = prob[0][0]
            guess = grid + 1

    return guess