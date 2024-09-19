import pandas as pd
import torch
from flask import Flask, jsonify
from model import F1RacePrediction
from util.predictor import predict_winner_from_pole, predict_winner_from_quali, predict_winner_from_quali_list
app = Flask(__name__)
_data = pd.read_csv('data/processed_data.csv')
_model = F1RacePrediction()
_model.load_state_dict(torch.load('racemodel.pth', weights_only=True))

@app.route('/predictquali/<season>/<round>', methods=['GET'])
def predict_quali(season, round):
    return jsonify({'winner': predict_winner_from_quali(season, round, _model, _data)})

@app.route('/predictqualilist/<season>/<round>', methods=['GET'])
def predict_quali_list(season, round):
    ret = predict_winner_from_quali_list(season, round, _model, _data)
    return jsonify({'winners': [ret[0][0], ret[1][0], ret[2][0]]})

@app.route('/predictpole/<season>/<round>', methods=['GET'])
def predict_pole(season, round):
    return jsonify({'win': True if predict_winner_from_pole(season, round, _model, _data) else False})

if __name__ == '__main__':
    app.run()