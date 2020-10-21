import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
import joblib

all_train_features = pd.read_csv(r'data/train_features.csv')
train_labels = pd.read_csv(r'data/train_labels.csv')
print(all_train_features.head())

#Original raw features
#features = ['Pclass', 'Sex', 'Age_clean', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']
#Cleaned Original features
#features = ['Pclass', 'Sex', 'Age_clean', 'SibSp', 'Parch', 'Fare_clean', 'Cabin', 'Embarked_clean']
#All features
features = ['Pclass', 'Sex', 'Age_clean', 'SibSp', 'Parch', 'Fare_clean', 'Fare_clean_tr', 'Cabin', 'Cabin_ind', 'Embarked_clean', 'Title', 'Family_cnt']
#Subset of selected features
#features = ['Pclass', 'Sex', 'Age_clean', 'Family_cnt', 'Fare_clean_tr', 'Cabin_ind', 'Title']

train_features = all_train_features[features]

matrix = np.triu(train_features.corr())
sns.heatmap(train_features.corr(), annot=True, fmt='.1f', vmin=-1, vmax=1, center= 0, cmap='coolwarm', mask=matrix)

rf = RandomForestClassifier()
parameters = {
    'n_estimators': [2**i for i in range(3, 10)],
    'max_depth': [2, 4, 8, 16, 32, None]
}

cv = GridSearchCV(rf, parameters, cv=5)
cv.fit(train_features, train_labels.values.ravel())

print('BEST PARAMS: {}\n'.format(cv.best_params_))

means = cv.cv_results_['mean_test_score']
stds = cv.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, cv.cv_results_['params']):
    print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))
    
feat_imp = cv.best_estimator_.feature_importances_
indices = np.argsort(feat_imp)
plt.yticks(range(len(indices)), [train_features.columns[i] for i in indices])
plt.barh(range(len(indices)), feat_imp[indices], color='r', align='center')
plt.show()

#joblib.dump(cv.best_estimator_, 'Original_raw_features.pkl')
#joblib.dump(cv.best_estimator_, 'Cleaned_original_features.pkl')
joblib.dump(cv.best_estimator_, 'All_features.pkl')
#joblib.dump(cv.best_estimator_, 'Subset_of_selected_features.pkl')