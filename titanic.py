import pandas as pd
import sys
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV as gsc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale

sys.path.append('D:\\NIW\\Python modules')
import nickpy

# Data import.
in_data = pd.read_csv(r"C:\Users\Nick\Desktop\Machine Learning\Titanic\train.csv")

in1 = in_data.copy()

# Age cleaning.
in1.loc[in1['Age'].isnull(), 'Age'] = -50

# Sex cleaning.
sex = {'male':1, 'female':2}
in1['Sex'].replace(sex, inplace=True)

#Embarkation cleaning.
embark = {'Q':1, 'C':2, 'S':3}
in1['Embarked'].replace(embark, inplace=True)
in1.loc[in1['Embarked'].isnull(), 'Embarked'] = -99999

# Work on title feature engineering.
titles_1 = in1['Name'].str.split(',')
titles_2 = titles_1.str.get(1)
titles_3 = titles_2.str.split()
titles_4 = titles_3.str.get(0)
unique_titles = titles_4.unique()
#TODO Extract the countess title where the 'the' is.

# print(titles_1[0])

names = in1['Name']
X = in1[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
X = scale(X, copy=True)
Y = in1['Survived']


#nickpy.plot.factor_scatter_matrix(data, labels)

validation_size = 0.20
clf = KNeighborsClassifier(n_neighbors=5)
#cv_results = model_selection.cross_val_score(clf, X, Y, cv=10, scoring='accuracy')
acc_mean, std_mean, conf_mat_mean = nickpy.ml.multi_run_validation(clf, validation_size, 100, X,Y)
conf_mat_pct = nickpy.ml.conf_matrix_pct(conf_mat_mean)

predictions = clf.predict(X)
for name, prediction, label in zip(names, predictions, Y):
    if prediction != label:
        print(name, 'has been classified as ', prediction, 'and should be ', label)