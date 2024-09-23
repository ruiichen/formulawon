import pandas as pd
import torch
import torch.nn.functional as F
from util.scaler import get_scaler
from services.quali_service import get_quali_session

def predict_winner_from_quali(season, round, model, data, order):
    model.eval()

    scaler = get_scaler(data)
    df = get_quali_session(season, round, order)
    conf = 1
    guess = 0
    for grid in range(len(df.index)):
        X_value = df[(df['grid'] == grid + 1)]
        X_value = X_value.drop(['driver', 'country', 'url'], axis=1)
        X_value = pd.DataFrame(scaler.transform(X_value), columns=X_value.columns)
        X_value = torch.Tensor(X_value.to_numpy())
        with (torch.no_grad()):
            prediction = model(X_value)
            prob = F.softmax(prediction, dim=1)
            prob, dummy = prob.topk(1, dim=1)

            if prediction.argmax().item():
                return grid + 1
            elif prob[0][0] < conf:
                conf = prob[0][0]
                guess = grid + 1
    return guess

def predict_winner_from_quali_list(season, round, model, data, order):
    model.eval()

    scaler = get_scaler(data)
    df = get_quali_session(season, round, order)

    winners = []
    losers = []
    for grid in range(len(df.index)):
        X_value = df[(df['grid'] == grid + 1)]
        X_value = X_value.drop(['driver', 'country', 'url'], axis=1)
        X_value = pd.DataFrame(scaler.transform(X_value), columns=X_value.columns)
        X_value = torch.Tensor(X_value.to_numpy())
        with (torch.no_grad()):
            prediction = model(X_value)
            prob = F.softmax(prediction, dim=1)
            prob, dummy = prob.topk(1, dim=1)

            if prediction.argmax().item():
                winners.append((grid+1, prob[0][0]))
            else:
                losers.append((grid+1, prob[0][0]))
    winners.sort(key=lambda x: x[1], reverse=True)
    losers.sort(key=lambda x: x[1])
    if len(winners) >= 3:
        return winners[:3]
    else:
        while len(winners) < 3:
            winners.append(losers.pop(0))
        return winners

def predict_winner_from_pole(season, round, model, data, order):
    model.eval()

    scaler = get_scaler(data)
    df = get_quali_session(season, round, order)
    with (torch.no_grad()):
        X_value = df[(df['grid'] == 1)]
        X_value = X_value.drop(['driver', 'country', 'url'], axis=1)
        X_value = pd.DataFrame(scaler.transform(X_value), columns=X_value.columns)
        X_value = torch.Tensor(X_value.to_numpy())
        prediction = model(X_value)
        return prediction.argmax().item()
