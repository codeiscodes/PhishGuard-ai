from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return 'Welcome to the Flask Backend!'

@app.route('/urlChecker')
def checkUrl():
    return
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=7000)