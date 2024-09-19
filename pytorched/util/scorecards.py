import torch
import pandas as pd
from scaler import get_scaler
import torch.nn.functional as F


def scorecard(season, model, data):
    scaler = get_scaler(data)

    df = data.copy()
    df.podium = df.podium.map(lambda x: 1 if x == 1 else 0)
    score = 0
    count = 0
    for circuit in df[df.season == season]['round'].unique():
        count += 1
        test = df[(df.season == season) & (df['round'] == circuit) & (df['grid'] == 1)]

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

def scorecard_ts(season, model, data):
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
        model.eval()

        conf = 1
        guess = 0
        pred = False
        for grid in range(20):
            test = df[(df.season == season) & (df['round'] == circuit) & (df['grid'] == grid + 1)]
            X_test = test.drop(['driver', 'country', 'podium', 'url'], axis=1)
            try:
                X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
            except:
                continue
            X_test = torch.Tensor(X_test.to_numpy())
            prediction = model(X_test)
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
                conf = top_p
                guess = grid + 1

        if not pred:
            predicted += 1
            print(
                f'{"CORRECTLY" if guess == winner else "INCORRECTLY"} predicted. P{guess} should win in round {circuit} (actual winner was P{winner})')
            score += 1 if guess == winner else 0

    print(f'{score} out of {predicted} predicted races')

def scorecard_pole(season, data):
    df = data.copy()
    score = 0
    count = 0
    for circuit in df[df.season == season]['round'].unique():
        count += 1
        try:
            winner = data[(data.season == season) & (data['round'] == circuit) & (data['podium'] == 1)].grid.to_numpy()[0]
        except:
            continue
        if winner == 1:
            score += 1
    print(f'{score}')