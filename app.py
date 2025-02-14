from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS, cross_origin
import joblib
import pandas as pd
from feature import process_url
import json
import os

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return 'Welcome to the Flask Backend!'

@app.route('/urlChecker', methods=["POST", "GET"])
@cross_origin()
def checkUrl():
    data_bytes = request.get_data()
    data_str = data_bytes.decode('utf-8')
    data_json = json.loads(data_str)
    model = joblib.load("classification_model.pkl")
    print("Model loaded.")
    features = process_url(data_json)
    print("URL Processed")
    print(features)
    feature_names = [
        "UsingIp", "longUrl", "shortUrl", "symbol", "redirecting", "prefixSuffix",
        "SubDomains", "Hppts", "DomainRegLen", "Favicon", "NonStdPort", "HTTPSDomainURL",
        "RequestURL", "AnchorURL", "LinksInScriptTags", "ServerFormHandler", "InfoEmail",
        "AbnormalURL", "WebsiteForwarding", "StatusBarCust", "DisableRightClick",
        "UsingPopupWindow", "IframeRedirection", "AgeofDomain", "DNSRecording",
        "WebsiteTraffic", "PageRank", "GoogleIndex", "LinksPointingToPage", "StatsReport"
    ]
    input_df = pd.DataFrame([features], columns=feature_names)
    prediction = model.predict(input_df)
    return json.dumps({"prediction":str(prediction[0])})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render assigns a port dynamically
    app.run(host="0.0.0.0", port=port)