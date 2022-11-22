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

dict_{0: 'Female',
  1: 'Male',
  2: 'Prefer not to say',
  3: 'Prefer to self-describe'},
 {0: 'Africa',
  1: 'Asia',
  2: 'China',
  3: 'Europe',
  4: 'India',
  5: 'Middle East',
  6: 'North America',
  7: 'Oceania',
  8: 'Other',
  9: 'Russia',
  10: 'South America',
  11: 'USA',
  12: 'dna'},
 {0: 'Bachelor’s degree',
  1: 'Doctoral degree',
  2: 'I prefer not to answer',
  3: 'Master’s degree',
  4: 'No formal education past high school',
  5: 'Professional degree',
  6: 'Some college/university study without earning a bachelor’s degree'},
 {0: '?',
  1: 'A business discipline (accounting, economics, finance, etc.)',
  2: 'Computer science (software engineering, etc.)',
  3: 'Engineering (non-computer focused)',
  4: 'Environmental science or geology',
  5: 'Fine arts or performing arts',
  6: 'Humanities (history, literature, philosophy, etc.)',
  7: 'I never declared a major',
  8: 'Information technology, networking, or system administration',
  9: 'Mathematics or statistics',
  10: 'Medical or life sciences (biology, chemistry, medicine, etc.)',
  11: 'Other',
  12: 'Physics or astronomy',
  13: 'Social sciences (anthropology, psychology, sociology, etc.)'},
 {0: 'Business Analyst',
  1: 'Chief Officer',
  2: 'Consultant',
  3: 'DBA/Database Engineer',
  4: 'Data Analyst',
  5: 'Data Engineer',
  6: 'Data Journalist',
  7: 'Data Scientist',
  8: 'Developer Advocate',
  9: 'Manager',
  10: 'Marketing Analyst',
  11: 'Other',
  12: 'Principal Investigator',
  13: 'Product/Project Manager',
  14: 'Research Assistant',
  15: 'Research Scientist',
  16: 'Salesperson',
  17: 'Software Engineer',
  18: 'Statistician',
  19: 'Student'},
 {0: 'Academics/Education',
  1: 'Accounting/Finance',
  2: 'Broadcasting/Communications',
  3: 'Computers/Technology',
  4: 'Energy/Mining',
  5: 'Government/Public Service',
  6: 'Hospitality/Entertainment/Sports',
  7: 'I am a student',
  8: 'Insurance/Risk Assessment',
  9: 'Manufacturing/Fabrication',
  10: 'Marketing/CRM',
  11: 'Medical/Pharmaceutical',
  12: 'Military/Security/Defense',
  13: 'Non-profit/Service',
  14: 'Online Business/Internet-based Sales',
  15: 'Online Service/Internet-based Services',
  16: 'Other',
  17: 'Retail/Sales',
  18: 'Shipping/Transportation'}


@app.route('/classify', methods=['POST'])
def classify():
    try:
        json_ = request.json
        print(json_)
        text = json_["text"]

        classes, class_colors, class_words = predict(text, model, tokenizer, device)

        return jsonify({'classes': classes, 'class_colors': class_colors, 'class_words': class_words})

    except:

        return jsonify({'trace': traceback.format_exc()})

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345


    with open('classifier.pkl', 'rb') as fi:
        model = pickle.load(fi)

    model = BertForSequenceClassification.from_pretrained(parameters["model"])
    model = model.to(device)
    print ('Model loaded')
    
    ref_token_id = tokenizer.pad_token_id # A token used for generating token reference
    sep_token_id = tokenizer.sep_token_id # A token used as a separator at the end of the text.
    cls_token_id = tokenizer.cls_token_id # A token used for prepending to the word sequence

    app.run(port=port, debug=True)