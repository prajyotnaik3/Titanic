import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
import seaborn as sns

dataset = pd.read_csv(r'data\titanic.csv')
#dataset.head()
#dataset.columns
#dataset.dtypes
print(dataset.describe())
print(dataset['Survived'].value_counts())

#Droppping categorical attributes
df = dataset.drop(['PassengerId', 'Name', 'Ticket', 'Sex', 'Cabin', 'Embarked'], axis = 1)
print(df.head())
print(df.corr())

for feature in ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']:
    print('\n*** Results for {} ***'.format(feature))
    print(df.groupby('Survived')[feature].describe())
    survived = df[df['Survived']==1][feature]
    not_survived = df[df['Survived']==0][feature]
    tstat, pval = stats.ttest_ind(survived, not_survived, equal_var=False)
    print('t-statistic: {:.1f}, p-value: {:.3}'.format(tstat, pval))
    
print(df.groupby('Pclass')['Fare'].describe())

print(df.groupby(df['Age'].isnull()).mean())

# Plot overlaid histograms for continuous features
for i in ['Age', 'Fare']:
    died = list(df[df['Survived'] == 0][i].dropna())
    survived = list(df[df['Survived'] == 1][i].dropna())
    xmin = min(min(died), min(survived))
    xmax = max(max(died), max(survived))
    width = (xmax - xmin) / 40
    sns.distplot(died, color='r', kde=False, bins=np.arange(xmin, xmax, width))
    sns.distplot(survived, color='g', kde=False, bins=np.arange(xmin, xmax, width))
    plt.legend(['Died', 'Survived'])
    plt.title('Overlaid histogram for {}'.format(i))
    plt.show()
    
# Generate categorical plots for ordinal features
for col in ['Pclass', 'SibSp', 'Parch']:
    sns.catplot(x=col, y='Survived', data=df, kind='point', aspect=2, )
    plt.ylim(0, 1)
    
# Create a new family count feature
df['Family_cnt'] = df['SibSp'] + df['Parch']
sns.catplot(x='Family_cnt', y='Survived', data=df, kind='point', aspect=2, )
plt.ylim(0, 1)

# Check if there are any missing values
print(df.isnull().sum())

#Categorical feature analysis
df2 = dataset.drop(['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare'], axis = 1)
print(df2.head())

print(df2.isnull().sum())

for col in df2.columns:
    print(col, df2[col].nunique())
    
#Check Survival rates
print(df2.groupby('Sex').mean())
print(df2.groupby('Embarked').mean())
print(df2.groupby(df2['Cabin'].isnull()).mean())

print(df2['Ticket'].value_counts())

# Create a title feature by parsing passenger name
df2['Title'] = df2['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
print(df2.head())
print(df2.pivot_table('Survived', index=['Title', 'Sex'], aggfunc=['count', 'mean']))

# Create a title feature by parsing passenger name and create a cabin indicator variable
df2['Title_Raw'] = df2['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
df2['Title'] = df2['Title_Raw'].apply(lambda x: x if x in ['Master', 'Miss', 'Mr', 'Mrs'] else 'Other')
df2['Cabin_ind'] = np.where(df2['Cabin'].isnull(), 0, 1)
print(df2.head())

# Generate categorical plots for features
for col in ['Title', 'Sex', 'Cabin_ind', 'Embarked']:
    sns.catplot(x=col, y='Survived', data=df2, kind='point', aspect=2, )
    plt.ylim(0, 1)

print(df2.pivot_table('Survived', index='Cabin_ind', columns='Embarked', aggfunc='count'))
