import pandas as pd
import torch
import json
from flask import Flask, jsonify, request
from model import F1RacePrediction
from util.predictor import predict_winner_from_pole, predict_winner_from_quali, predict_winner_from_quali_list
from util.exceptions import *
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
_data = pd.read_csv('data/processed_data.csv')
_model = F1RacePrediction()
_model.load_state_dict(torch.load('racemodel.pth', weights_only=True))

@app.route('/predictquali/<season>/<round>', methods=['GET'])
def predict_quali(season, round):
    try:
        order = request.headers.get('order')
        if not order:
            order = None
        else:
            order = json.loads(order)
        return jsonify({'winner': predict_winner_from_quali(season, round, _model, _data, order)})
    except NotFoundError as ex:
        return jsonify(exception_response(ex))
    except ExternalServiceError as ex:
        return jsonify(exception_response(ex))
    except:
        return jsonify(exception_response(InternalServerError()))

@app.route('/predictqualilist/<season>/<round>', methods=['GET'])
def predict_quali_list(season, round):
    try:
        order = request.headers.get('order')
        if not order:
            order = None
        else:
            order = json.loads(order)
        ret = predict_winner_from_quali_list(season, round, _model, _data, order)
        return jsonify({'winners': [ret[0][0], ret[1][0], ret[2][0]]})
    except NotFoundError as ex:
        return jsonify(exception_response(ex))
    except ExternalServiceError as ex:
        return jsonify(exception_response(ex))
    except:
        return jsonify(exception_response(InternalServerError()))

@app.route('/predictpole/<season>/<round>', methods=['GET'])
def predict_pole(season, round):
    try:
        order = request.headers.get('order')
        if not order:
            order = None
        else:
            order = json.loads(order)
        return jsonify({'win': True if predict_winner_from_pole(season, round, _model, _data, order) else False})
    except NotFoundError as ex:
        return jsonify(exception_response(ex))
    except ExternalServiceError as ex:
        return jsonify(exception_response(ex))
    except:
        return jsonify(exception_response(InternalServerError()))

@app.route("/")
def home():
    return "FORMULA WON API"

if __name__ == '__main__':
    app.run()