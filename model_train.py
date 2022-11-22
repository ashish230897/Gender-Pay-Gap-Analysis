import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


def preprocess_data(data):
    data = data.drop(0, axis=0)
    data = data[data['Q9'].notnull()]
    data = data[data['Q9'] != 'I do not wish to disclose my approximate yearly compensation']

    medians = {'0-10,000': 5000, '10-20,000': 15000, '20-30,000': 25000, '30-40,000': 35000, 
           '40-50,000': 45000, '50-60,000': 55000, '60-70,000': 65000, '70-80,000': 75000, 
           '80-90,000': 85000, '90-100,000': 95000, '100-125,000': 112500, 
           '125-150,000': 137500, '150-200,000': 175000, '200-250,000': 225000, 
           '250-300,000': 275000, '300-400,000': 350000, '400-500,000': 450000, 
           '500,000+':500000}
    data['target'] = data['Q9'].apply(lambda x: medians[x])
    data = data[data['target'] < 500000]

    country_regroup = {'Morocco': 'Africa',
                 'Tunisia': 'Africa',
                 'Austria': 'Europe',
                 'Hong Kong (S.A.R.)': 'Asia',
                 'Republic of Korea': 'Asia',
                 'Thailand': 'Asia',
                 'Czech Republic': 'Europe',
                 'Philippines': 'Asia',
                 'Romania': 'Europe',
                 'Kenya': 'Africa',
                 'Finland': 'Europe',
                 'Norway': 'Europe',
                 'Peru': 'South America',
                 'Iran, Islamic Republic of...': 'Middle East',
                 'Bangladesh': 'Asia',
                 'New Zealand': 'Oceania',
                 'Egypt': 'Africa',
                 'Chile': 'South America',
                 'Belarus': 'Europe',
                 'Hungary': 'Europe',
                 'Ireland': 'Europe',
                 'Belgium': 'Europe',
                 'Malaysia': 'Asia',
                 'Denmark': 'Europe',
                 'Greece': 'Europe',
                 'Pakistan': 'Asia',
                 'Viet Nam': 'Asia',
                 'Argentina': 'South America',
                 'Colombia': 'South America',
                 'Indonesia': 'Oceania',
                 'Portugal': 'Europe',
                 'South Africa': 'Africa',
                 'South Korea': 'Asia',
                 'Switzerland': 'Europe',
                 'Sweden': 'Europe',
                 'Israel': 'Middle East',
                 'Nigeria': 'Africa',
                 'Singapore': 'Asia',
                 'I do not wish to disclose my location': 'dna',
                 'Mexico': 'North America',
                 'Ukraine': 'Europe',
                 'Netherlands': 'Europe',
                 'Turkey': 'Asia',
                 'Poland': 'Europe',
                 'Australia': 'Oceania',
                 'Italy': 'Europe',
                 'Spain': 'Europe',
                 'Japan': 'Asia',
                 'France': 'Europe',
                 'Canada': 'North America', 
                 'United Kingdom of Great Britain and Northern Ireland': 'Europe',
                 'Germany': 'Europe',
                 'Brazil': 'South America',
                 'Russia': 'Russia',
                 'Other': 'Other',
                 'China': 'China',
                 'India': 'India',
                 'United States of America': 'USA'}
    data['Q3'] = data['Q3'].apply(lambda x: country_regroup[x])

    features = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"]
    target = ["target"]

    data_model = data[features + target]

    data_model = data_model.fillna('?')

    age_medians = {'30-34': 32, '22-24': 23, '35-39': 37, '18-21': 19.5, '40-44': 42, '25-29': 27, '55-59': 57, '60-69': 64.5, '45-49': 47, '50-54': 52, '70-79': 74.5, '80+': 80}
    exp_medians = {'5-10': 7.5, '0-1': 0.5, '10-15': 12.5, '3-4': 3.5, '1-2': 1.5, '2-3': 2.5, '15-20': 17.5, '4-5': 4.5, '25-30': 27.5, '20-25': 22.5, '30 +': 30, '?': 0}

    data_model['Q2'] = data_model['Q2'].apply(lambda x: age_medians[x])
    data_model['Q8'] = data_model['Q8'].apply(lambda x: exp_medians[x])

    for feature in ["Q1", "Q3", "Q4", "Q5", "Q6", "Q7"]:
        data_model[feature] = data_model[feature].astype('category')

    categorical_columns = data_model.select_dtypes(['category']).columns
    
    mapping_dicts = []
    for col in categorical_columns:
        d = dict(enumerate(data_model[col].cat.categories))
        mapping_dicts.append(d)
    
    data_model[categorical_columns] = data_model[categorical_columns].apply(lambda x: x.cat.codes)
    data_model = data_model.rename(index=str, columns={"Q1": 'Gender', "Q2": 'Age', "Q3": 'Country', 
                                                       "Q4": 'Education', "Q5": 'Major', "Q6": 'Profession', 
                                                       "Q7": 'Industry', "Q8": 'Experience'})

    classes = ['less than 10k', 'between 10k and 30k', 'between 30k and 50k', 'between 50k and 80k', 'between 80k and 125k', 'more than 100k']

    target_clubbing = {5000: 0,  
                  15000: 1, 25000: 1, 
                  35000: 2, 45000: 2, 
                  55000: 3,  65000: 3,  75000: 3,
                  85000: 4, 95000: 4, 112500: 4,
                  137500: 5,  175000: 5, 225000: 5, 275000: 5, 350000: 5,  450000: 5
                 }

    data_model['target'] = data_model['target'].apply(lambda x: target_clubbing[x])

    return data_model, classes, mapping_dicts

def train(x_train, y_train):
    tree = DecisionTreeClassifier(max_depth=2, random_state=1)
    classifier = AdaBoostClassifier(base_estimator=tree, n_estimators=50, random_state=0) 
    classifier.fit(x_train, y_train)

    return classifier


def main():
    data = pd.read_csv("./multipleChoiceResponses.csv")

    data_pre, classes, mapping_dicts = preprocess_data(data)

    data_train, data_test = train_test_split(data_pre, test_size=0.2)
    print("Train data size is {}".format(len(data_train)))
    print("Test data size is {}".format(len(data_test)))

    features = ["Gender", "Age", "Country", "Education", "Major", "Profession", "Industry", "Experience"]

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for row in data_train.iterrows():
        instance = []
        for feature in features:
            instance.append(row[1][feature])
        x_train.append(instance)
        y_train.append(row[1]["target"])

    for row in data_test.iterrows():
        instance = []
        for feature in features:
            instance.append(row[1][feature])
        x_test.append(instance)
        y_test.append(row[1]["target"])
        
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    classifier = train(x_train, y_train)

    y_pred = classifier.predict(x_test)
    print("accuracy is : ", metrics.accuracy_score(y_test, y_pred))

    importances = classifier.feature_importances_
    print("Feature importances are ")
    for i, importance in enumerate(importances):
        print(features[i], importance)

    # save model weights
    import pickle
    with open('./classifier.pkl', 'wb') as fi:
        pickle.dump(classifier, fi)


if __name__ == "__main__":
    main()