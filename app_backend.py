# Dependencies
from flask import Flask, request, jsonify
import os
import json

from flask_cors import CORS

import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
import pandas as pd

import pickle

# Your API definition
app_backend = Flask(__name__)
CORS(app_backend)

model = None

features = ["Gender", "Age", "Country", "Education", "Major", "Profession", "Industry", "Experience"]
cat_features = ["Gender", "Country", "Education", "Major", "Profession", "Industry"]

target_clubbing = {5000: 0,  
                15000: 1, 25000: 1, 
                35000: 2, 45000: 2, 
                55000: 3,  65000: 3,  75000: 3,
                85000: 4, 95000: 4, 112500: 4,
                137500: 5,  175000: 5, 225000: 5, 275000: 5, 350000: 5,  450000: 5
                }

classes = {
    0: "0-10,000", 1: "10-30,000", 2: "30-50,000", 3: "50-80,000", 4: "80-120,000", 5: "120,000 and above"
}

dict_cat_value = {
    "Gender": {0: 'Female', 1: 'Male', 2: 'Prefer not to say', 3: 'Prefer to self-describe'},
    "Country": {0: 'Africa', 1: 'Asia', 2: 'China', 3: 'Europe', 4: 'India', 5: 'Middle East', 6: 'North America', 
    7: 'Oceania', 8: 'Other', 9: 'Russia', 10: 'South America', 11: 'USA', 12: 'dna'},
    "Education": {0: 'Bachelor’s degree', 1: 'Doctoral degree', 2: 'I prefer not to answer', 3: 'Master’s degree',
    4: 'No formal education past high school', 5: 'Professional degree', 6: 'Some college/university study without earning a bachelor’s degree'},
    "Major": {0: '?', 1: 'A business discipline (accounting, economics, finance, etc.)', 2: 'Computer science (software engineering, etc.)',
    3: 'Engineering (non-computer focused)', 4: 'Environmental science or geology', 5: 'Fine arts or performing arts', 6: 'Humanities (history, literature, philosophy, etc.)',
    7: 'I never declared a major', 8: 'Information technology, networking, or system administration',
    9: 'Mathematics or statistics', 10: 'Medical or life sciences (biology, chemistry, medicine, etc.)', 11: 'Other',
    12: 'Physics or astronomy', 13: 'Social sciences (anthropology, psychology, sociology, etc.)'},
    "Profession": {0: 'Business Analyst', 1: 'Chief Officer', 2: 'Consultant', 3: 'DBA/Database Engineer', 4: 'Data Analyst',
    5: 'Data Engineer', 6: 'Data Journalist', 7: 'Data Scientist', 8: 'Developer Advocate', 9: 'Manager', 10: 'Marketing Analyst',
    11: 'Other', 12: 'Principal Investigator', 13: 'Product/Project Manager', 14: 'Research Assistant', 15: 'Research Scientist',
    16: 'Salesperson', 17: 'Software Engineer', 18: 'Statistician', 19: 'Student'},
    "Industry": {0: 'Academics/Education', 1: 'Accounting/Finance', 2: 'Broadcasting/Communications', 3: 'Computers/Technology',
    4: 'Energy/Mining', 5: 'Government/Public Service', 6: 'Hospitality/Entertainment/Sports', 7: 'I am a student',
    8: 'Insurance/Risk Assessment', 9: 'Manufacturing/Fabrication', 10: 'Marketing/CRM', 11: 'Medical/Pharmaceutical',
    12: 'Military/Security/Defense', 13: 'Non-profit/Service', 14: 'Online Business/Internet-based Sales', 15: 'Online Service/Internet-based Services',
    16: 'Other', 17: 'Retail/Sales', 18: 'Shipping/Transportation'}}

dict_value_cat = {}

@app_backend.route('/classify', methods=['POST'])
def classify():
    try:
        json_ = request.json
        print(json_)
        data = [[]]
        for feature in features:
            if feature != "Age" and feature != "Experience":
                data[0].append(dict_value_cat[feature][json_[feature]])
            else: data[0].append(json_[feature])

        class_ = model.predict(np.array(data))

        return jsonify({'class': classes[int(class_)]})

    except:

        return jsonify({'trace': traceback.format_exc()})

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345


    with open('classifier.pkl', 'rb') as fi:
        model = pickle.load(fi)
    print ('Model loaded')

    for key, value in dict_cat_value.items():
        dict_value_cat[key] = {}
        for key_, value_ in dict_cat_value[key].items():
            dict_value_cat[key][value_] = key_
    
    print(dict_value_cat)

    app_backend.run(port=port, debug=True)