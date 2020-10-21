import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from time import time

val_features = pd.read_csv(r'data/val_features.csv')
val_labels = pd.read_csv(r'data/val_labels.csv')

def evaluate_model(name, model, features, labels):
    start = time()
    pred = model.predict(features)
    end = time()
    accuracy = round(accuracy_score(labels, pred), 3)
    precision = round(precision_score(labels, pred), 3)
    recall = round(recall_score(labels, pred), 3)
    print('{} -- \tAccuracy: {} / Precision: {} / Recall: {} / Latency: {}ms'.format(name, accuracy, precision, recall, round((end - start)*1000, 1)))

# Read in models
models = {}
for mdl in ['Original_raw', 'Cleaned_original', 'All', 'Subset_of_selected']:
    models[mdl] = joblib.load('{}_features.pkl'.format(mdl))
    
#Original raw features
features = ['Pclass', 'Sex', 'Age_clean', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']
evaluate_model('Raw Features', models['Original_raw'], val_features[features], val_labels)

#Cleaned Original features
features = ['Pclass', 'Sex', 'Age_clean', 'SibSp', 'Parch', 'Fare_clean', 'Cabin', 'Embarked_clean']
evaluate_model('Raw Features', models['Cleaned_original'], val_features[features], val_labels)

#All features
features = ['Pclass', 'Sex', 'Age_clean', 'SibSp', 'Parch', 'Fare_clean', 'Fare_clean_tr', 'Cabin', 'Cabin_ind', 'Embarked_clean', 'Title', 'Family_cnt']
evaluate_model('Raw Features', models['All'], val_features[features], val_labels)

#Subset of selected features
features = ['Pclass', 'Sex', 'Age_clean', 'Family_cnt', 'Fare_clean_tr', 'Cabin_ind', 'Title']
evaluate_model('Raw Features', models['Subset_of_selected'], val_features[features], val_labels)