import pandas as pd
from sklearn.preprocessing import StandardScaler

_standardscaler = None

def get_scaler(df):
    global _standardscaler
    if _standardscaler is not None:
        return _standardscaler
    else:
        _standardscaler = StandardScaler()
        df.podium = df.podium.map(lambda x: 1 if x == 1 else 0)
        df = df[df.season <2021]
        X_train = df.drop(['driver', 'country', 'podium', 'url'], axis=1)
        _standardscaler.fit(X_train)
        return _standardscaler
