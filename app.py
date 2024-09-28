from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pymongo
from dotenv import load_dotenv
import os

app = Flask(__name__)
CORS(app)

# Load environment variables
load_dotenv('.env')

# Load MongoDB connection
mongo_url = os.getenv('MONGO_URL')
client = pymongo.MongoClient(mongo_url)
db = client['test']
collection = db['spam']

# Train the AI model and return the vectorizer and classifier
def ai_machine():
    text_sms = pd.DataFrame(list(collection.find()))
    X = text_sms['Message']
    y = text_sms['Class']
    
    vectorizer = TfidfVectorizer()
    X_transformed = vectorizer.fit_transform(X)
    
    clf = RandomForestClassifier(n_estimators=101, random_state=42)
    clf.fit(X_transformed, y)
    
    return vectorizer, clf

# Load the model and vectorizer once when the app starts
vectorizer, clf = ai_machine()

@app.route("/", methods=['GET'])
def get_home():
    return 'Hello World'

@app.route("/contact", methods=['POST'])
def get_contacted():
    data = request.json
    userquery = str(data.get('message'))
    
    # Transform the user query using the vectorizer
    userquery_transformed = vectorizer.transform([userquery])
    
    # Predict using the trained model
    y_pred = clf.predict(userquery_transformed)
    
    results = 'Good' if y_pred[0] == 'ham' else 'Spam'
    
    return jsonify({"results": results}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
