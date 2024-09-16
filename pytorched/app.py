from flask import Flask
app = Flask(__name__)


@app.route('/predictwinner', methods=['GET'])
def predict_winner():
    return 'Hello World!'

if __name__ == '__main__':
    app.run()