import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from util.scaler import get_scaler
from util.predictor import predict_winner_from_quali, predict_winner_from_pole


_MODEL = None

class F1RacePrediction(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential (
        nn.Flatten(),
        nn.Linear(96, 75),
        nn.ReLU(),
        nn.Linear(75, 25),
        nn.ReLU(),
        nn.Linear(25,50),
        nn.ReLU(),
        nn.Linear(50,10)
    )

  def forward(self, x):
    return self.layers(x)

def get_model():
    global _MODEL

    if _MODEL is not None:
        return _MODEL
    else:
        _MODEL = torch.jit.load('racemodel.pth')
        _MODEL.eval()
        return _MODEL

moggle= F1RacePrediction()
moggle.load_state_dict(torch.load('racemodel.pth', weights_only=True))
data = pd.read_csv('data/processed_data.csv')

def scorecard_ts(season, model2):
    scaler = get_scaler(data)
    df = data.copy()
    df.podium = df.podium.map(lambda x: 1 if x == 1 else 0)
    score = 0
    count = 0
    predicted = 0
    for circuit in df[df.season == season]['round'].unique():
        count += 1
        winner = data[(data.season == season) & (data['round'] == circuit) & (data['podium'] == 1)].grid
        try:
            winner = winner.to_numpy()[0]
        except:
            winner = None
        model2.eval()

        conf = 1
        guess = 0
        pred = False
        for grid in range(19):
            test = df[(df.season == season) & (df['round'] == circuit) & (df['grid'] == grid + 1)]
            X_test = test.drop(['driver', 'country', 'podium', 'url'], axis=1)
            try:
                X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
            except:
                continue
            X_test = torch.Tensor(X_test.to_numpy())
            prediction = model2(X_test)
            prob = F.softmax(prediction, dim=1)
            top_p, top_class = prob.topk(1, dim=1)
            if prediction.argmax().item():
                predicted += 1
                print(
                    f'{"CORRECTLY" if grid + 1 == winner else "INCORRECTLY"} predicted. P{grid + 1} should win in round {circuit} (actual winner was P{winner})')
                score += 1 if grid + 1 == winner else 0
                pred = True
                break
            elif top_p[0][0] < conf:
                conf = top_p[0][0]
                guess = grid + 1

        if not pred:
            predicted += 1
            print(
                f'{"CORRECTLY" if guess == winner else "INCORRECTLY"} predicted. P{guess} should win in round {circuit} (actual winner was P{winner})')
            score += 1 if guess == winner else 0

    print(f'{score} out of {predicted} predicted races')

def scorecard(season, model):
    scaler = get_scaler(data)

    df = data.copy()
    df.podium = df.podium.map(lambda x: 1 if x == 1 else 0)
    score = 0
    count = 0
    for circuit in df[df.season == season]['round'].unique():
        count += 1
        test = df[(df.season == season) & (df['round'] == circuit) & (df['grid'] == 1)]

        winner = data[(data.season == season) & (data['round'] == circuit) & (data['podium'] == 1)].grid
        try:
            winner = winner.to_numpy()[0]
        except:
            winner = None

        X_test = test.drop(['driver', 'country', 'podium', 'url'], axis=1)
        y_test = test.podium
        # scaling
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
        X_test = torch.Tensor(X_test.to_numpy())
        with torch.no_grad():
            model.eval()
            prediction = model(X_test)
            prob = F.softmax(prediction, dim=1)
            top_p, top_class = prob.topk(1, dim=1)
            if prediction.argmax().item() == y_test.to_numpy()[0]:
                score += 1
                print(
                    f'CORRECTLY predicted the pole to {"win" if prediction.argmax().item() == 1 else "lose"} with {top_p}% confidence')
            else:
                print(
                    f'INCORRECTLY predicted the pole to {"win" if prediction.argmax().item() == 1 else "lose"} with {top_p}% confidence')

    print(f'{score} out of {count} races')

scorecard(2023,moggle)
for i in range(22):
    print(predict_winner_from_pole(2023, i+1, moggle, data))