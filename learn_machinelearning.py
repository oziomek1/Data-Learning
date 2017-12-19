# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

"SUPERVISED LEARNINE"

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/column_2C_weka.csv')
print(plt.style.available)

plt.style.use('ggplot')

data.info()
print ('\n\n', data.head())
print('\n\n', data.describe())

color_list = ['red' if i == 'Abnormal' else 'green' for i in data.loc[:, 'class']]
pd.plotting.scatter_matrix(data.loc[:, data.columns != 'class'],
                            c=color_list,
                            figsize=[15,15],
                            diagonal='hist',
                            alpha=0.5, s = 200,
                            marker = '*',
                            edgecolor="black")

degree = data.pelvic_radius.value_counts()
sns.barplot(x=degree.index, y=degree.values)
plt.title('We will see', color="blue", fontsize=15)

sns.countplot(x="class", data=data)
data.loc[:, 'class'].value_counts()
plt.title('KNN', color="blue", fontsize=15)
# plt.show()

#KKN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
x, y = data.loc[:, data.columns != 'class'], data.loc[:, 'class']
knn.fit(x, y)
prediction = knn.predict(x)
print("Prediction: {}".format(prediction))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
knn = KNeighborsClassifier(n_neighbors=3)
x, y = data.loc[:, data.columns != 'class'], data.loc[:, 'class']
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
print("Train accuracy:", knn.score(x_train, y_train))
print("With KNN, where K = 4, accuracy is: ", knn.score(x_test, y_test))

data1 = data[data['class'] == 'Abnormal']
x = np.array(data1.loc[:, 'pelvic_incidence']).reshape(-1, 1)
y = np.array(data1.loc[:,'sacral_slope']).reshape(-1,1)
plt.figure(figsize=[10,10])
plt.scatter(x=x, y=y)
plt.xlabel('pelvic_incidence')
plt.ylabel('sacral_slope')
# plt.show()

from sklearn.linear_model import LinearRegression
regression = LinearRegression()
#predict space
predict_space = np.linspace(min(x), max(x)).reshape(-1, 1)

regression.fit(x, y)
predicted = regression.predict(predict_space)

#print(predicted)
# R^2
print("R^2 score: ", regression.score(x, y))

plt.plot(predict_space, predicted, color='black', linewidth=3)
plt.scatter(x=x, y=y)
plt.xlabel('pelvic_incidence')
plt.ylabel('sacral_slope')
# plt.show()

from sklearn.model_selection import cross_val_score
reg = LinearRegression()
k = 5
cv_result = cross_val_score(reg, x, y, cv=k)
print('CV Scores:', cv_result, '\n', 'CV Scores average:', np.sum(cv_result)/k)

#Ridge
from sklearn.linear_model import Ridge
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=2, test_size=0.3)
ridge = Ridge(alpha=0.2, normalize=True)
ridge.fit(x_train, y_train)
ridge_predict = ridge.predict(x_test)
print('Ridge score', ridge.score(x_test, y_test))

#Lasso
from sklearn.linear_model import Lasso
x = np.array(data1.loc[:,['pelvic_incidence','pelvic_tilt numeric','lumbar_lordosis_angle','pelvic_radius']])
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 3, test_size = 0.3)
lasso = Lasso(alpha = 0.2, normalize = True)
lasso.fit(x_train,y_train)
ridge_predict = lasso.predict(x_test)
print('Lasso score: ',lasso.score(x_test,y_test), '\n Lasso coefficients: ',lasso.coef_)

#confusion matrix with random forest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
x, y = data.loc[:, data.columns != 'class'], data.loc[:, 'class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
rf = RandomForestClassifier(random_state=4)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix \n', cm, '\n Classification report \n', classification_report(y_test, y_pred))
sns.heatmap(cm, annot=True, fmt='d')

# ROC Curve with logistic regression
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
# abnormal = 1 and normal = 0
data['class_binary'] = [1 if i == 'Abnormal' else 0 for i in data.loc[:,'class']]
x,y = data.loc[:,(data.columns != 'class') & (data.columns != 'class_binary')], data.loc[:,'class_binary']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=42)
logreg = LogisticRegression()
logreg.fit(x_train,y_train)
y_pred_prob = logreg.predict_proba(x_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')

# grid search cross validation with 1 hyperparameter
from sklearn.model_selection import GridSearchCV
grid = {'n_neighbors': np.arange(1,60)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, grid, cv=4) # GridSearchCV
knn_cv.fit(x,y)# Fit

print('Tuned K: {}'.format(knn_cv.best_params_), '\n Best score: {}'.format(knn_cv.best_score_))

#grid search CV with 2 hyperparam
param_grid = {'C': np.logspace(-3, 3, 7), 'penalty': ['l1', 'l2']}
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state = 12)
logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg,param_grid,cv=3)
logreg_cv.fit(x_train,y_train)

print("Tuned hyperparameters : {}".format(logreg_cv.best_params_),"\n Best Accuracy: {}".format(logreg_cv.best_score_))



# Load data
data = pd.read_csv('../input/column_2C_weka.csv')
# get_dummies
df = pd.get_dummies(data)
df.drop('class_Normal', axis=1, inplace=True)
df.head()

#SVM, pre-process and pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

steps = [('scalar', StandardScaler()),
         ('SVM', SVC())]
pipeline = Pipeline(steps)
parameters =    {'SVM__C':[1,10,100],
                'SVM__gamma':[0.1, 0.01]}
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state = 1)
cv = GridSearchCV(pipeline,param_grid=parameters,cv=3)
cv.fit(x_train,y_train)

y_pred = cv.predict(x_test)

print("Accuracy: {}".format(cv.score(x_test, y_test)), '\nTuned Model Parameters: {}'.format(cv.best_params_))

