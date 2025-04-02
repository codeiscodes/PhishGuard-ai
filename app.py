from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS, cross_origin
import joblib
import pandas as pd
from feature import process_url
import json
import os
import csv

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
    [features, response] = process_url(data_json)
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
    print(input_df)
    prediction = model.predict(input_df)
    print(str(prediction[0]))
    print("history", response)
    input_df["label"] = prediction[0]
    input_df["URL"] = data_json
    file_path = "model_prediction.csv"
    data = input_df.values.tolist()
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(input_df.columns)
        writer.writerows(data)
    if isinstance(response, list) and len(response) > 0:
        redirections = str(response[-1])
    else:
        redirections = "-1"

    return json.dumps({"prediction": str(prediction[0]), "redirections": redirections})

@app.route('/updatePrediction', methods=["POST", "GET"])
@cross_origin()
def updatePrediction():
    try:
        df = pd.read_csv("model_prediction.csv")
        if df.empty or "label" not in df.columns:
            return json.dumps({"status": "false", "error": "CSV is empty or missing 'label' column"})
        last_index = df.index[-1]  # Get the last row index
        df.at[last_index, "label"] = "0" if str(df.at[last_index, "label"]) == "1" else "1"
        df.to_csv("model_prediction.csv", index=False)
        return json.dumps({"status": "true"})
    except Exception as e:
        return json.dumps({"status": "false", "error": str(e)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 7000))  # Render assigns a port dynamically
    app.run(host="0.0.0.0", port=port)