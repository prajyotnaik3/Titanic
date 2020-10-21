import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.stats
from statsmodels.graphics.gofplots import qqplot
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

dataset = pd.read_csv(r'data\titanic.csv')
#print(dataset.head())

#Filling Null / Missing values
print(dataset.isnull().sum())

dataset['Age_clean'] = dataset['Age'].fillna(dataset['Age'].mean())
#print(dataset.isnull().sum())
dataset['Embarked_clean'] = dataset['Embarked'].fillna('O')
#print(dataset.isnull().sum())

#Remove outliers
print(dataset.describe())

for feat in ['Age_clean', 'SibSp', 'Parch', 'Fare']:
    outliers = []
    data = dataset[feat]
    mean = np.mean(data)
    std =np.std(data)
    
    
    for y in data:
        z_score= (y - mean)/std 
        if np.abs(z_score) > 3:
            outliers.append(y)
    print('\nOutlier caps for {}:'.format(feat))
    print('  --95p: {:.1f} / {} values exceed that'.format(data.quantile(.95), 
          len([i for i in data if i > data.quantile(.95)])))
    print('  --3sd: {:.1f} / {} values exceed that'.format(mean + 3*(std), len(outliers)))
    print('  --99p: {:.1f} / {} values exceed that'.format(data.quantile(.99),
          len([i for i in data if i > data.quantile(.99)])))
    
dataset['Age_clean'].clip(upper=dataset['Age_clean'].quantile(.99), inplace=True)
dataset['Fare_clean'] = dataset['Fare'].clip(upper=dataset['Fare'].quantile(.99))

#Transform skewed features
for feature in ['Age_clean', 'Fare_clean']:
    sns.distplot(dataset[feature], kde=False)
    plt.title('Histogram for {}'.format(feature))
    plt.show()

# Generate QQ plots
for i in [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    data_t = dataset['Fare_clean']**(1/i)
    qqplot(data_t, line='s')
    plt.title("Transformation: 1/{}".format(str(i)))
    
# Box-Cox transformation
for i in [3, 4, 5, 6, 7]:
    data_t = dataset['Fare_clean']**(1/i)
    n, bins, patches = plt.hist(data_t, 50, density=True)
    mu = np.mean(data_t)
    sigma = np.std(data_t)
    plt.plot(bins, scipy.stats.norm.pdf(bins, mu, sigma))
    plt.title("Transformation: 1/{}".format(str(i)))
    plt.show()
    
dataset['Fare_clean_tr'] = dataset['Fare_clean'].apply(lambda x: x**(1/5))

#Creating new features
dataset['Title'] = dataset['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
dataset['Cabin_ind'] = np.where(dataset['Cabin'].isnull(), 0, 1)
dataset['Family_cnt'] = dataset['SibSp'] + dataset['Parch']

#Converting categorical attributes to Numerical features
for feature in ['Sex', 'Cabin', 'Embarked', 'Embarked_clean', 'Title']:
    le = LabelEncoder()
    dataset[feature] = le.fit_transform(dataset[feature].astype(str))
print(dataset.head())

features = dataset.drop(['PassengerId', 'Ticket', 'Name', 'Survived'], axis=1)
labels = pd.DataFrame(dataset['Survived'])

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

X_train.to_csv(r'data/train_features.csv', index=False)
X_val.to_csv(r'data/val_features.csv', index=False)
X_test.to_csv(r'data/test_features.csv', index=False)
y_train.to_csv(r'data/train_labels.csv', index=False)
y_val.to_csv(r'data/val_labels.csv', index=False)
y_test.to_csv(r'data/test_labels.csv', index=False)