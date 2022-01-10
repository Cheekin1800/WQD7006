import os
import pandas as pd
import numpy as np
import seaborn as sns


## Check and change to desire directory and load the dataset
os.getcwd()
os.chdir("D:\\Data Science\\Python\\projects\\Excel dataset")
df = pd.read_csv("Covid Dataset.csv")

## Increase console displays and have an info & datatypes overview
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
df.info
df.dtypes
df.describe()


## Duplicate dataset for modifying
df_covid = df

## Check if there's any missing data
df_covid.isnull().any()

## Countplot for variable COVID-19
sns.countplot(df['COVID-19'])


## convert all x variable to integer as machine-readable form
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_covid = df_covid.apply(le.fit_transform)
df_covid.dtypes
print(df_covid.head())


## random shuffle the data
from sklearn.utils import shuffle
df_covid = shuffle(df_covid)

## Perform correlational test
cor = df_covid.corr()
cor
sns.heatmap(cor)

## To remove the target from the dataframe
features_ = df_covid.drop(columns = ['COVID-19', 'Wearing Masks','Sanitization from Market'])
target = df_covid['COVID-19']

## Data split to train and test data sets with 30% test size
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features_, target, test_size = 0.3,random_state = 0)



## Decision Tree(DT) Classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
Dt = DecisionTreeClassifier(criterion="entropy",max_depth=4)
DTY = Dt.fit(X_train,y_train)
DTY_pred = DTY.predict(X_test)
A = metrics.accuracy_score(y_test,DTY_pred)*100
print(A)

## DT feature importance
DTimp = pd.DataFrame(data={'Attribute' : X_train.columns,'Importance': Dt.feature_importances_})
DTimp = DTimp.sort_values(by='Importance', ascending=False)
DTimp
sns.barplot(y='Attribute', x='Importance', data=DTimp)

## DT confusion matrix classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, DTY_pred))

## PLOT DT confusion matrix
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(Dt,X_test,y_test)

## Display decision tree output via graphviz
import graphviz
from sklearn import tree
dot_data=tree.export_graphviz(DTY,out_file=None, filled=True)
graph = graphviz.Source(dot_data, format="png")
graph.render("Dt")
graph.view()



## XGB Classifier
from xgboost import XGBClassifier
import xgboost as xgb
XGB = XGBClassifier(max_depth=4,learning_rate=0.1,n_estimators=300,booster="gbtree",reg_lambda=0.5,reg_alpha=0.5)
XGBclf = XGB.fit(X_train, y_train)
y_pred_xgb = XGBclf.predict(X_test)
B = metrics.accuracy_score(y_test,y_pred_xgb)*100
print(B)

## XGB feature importance
importances = pd.DataFrame(data={'Attribute' : X_train.columns,'Importance': XGB.feature_importances_})
importances = importances.sort_values(by='Importance', ascending=False)
importances
sns.barplot(y='Attribute', x='Importance', data=importances)

## XGB confusion matrix classification report
print(classification_report(y_test, y_pred_xgb))

## PLOT XGB confusion matrix
plot_confusion_matrix(XGB,X_test,y_test)

## Plot XGB output
from xgboost import plot_tree
plot_tree(XGB, num_trees=5, rankdir="LR")



## Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators= 40, max_depth=4)
RFY = RF.fit(X_train, y_train)
y_pred_RFY = RFY.predict(X_test)
C = accuracy_score(y_test, y_pred_RFY)*100
print(C)

## Random Forest Feature importance
RFRimp = pd.DataFrame(data={'Attribute' : X_train.columns,'Importance': RFY.feature_importances_})
RFRimp = RFRimp.sort_values(by='Importance', ascending=False)
RFRimp
sns.barplot(y='Attribute', x='Importance', data=RFRimp)

## RF confusion matrix classification report
print(classification_report(y_test, y_pred_RFY))

## PLOT RF confusion matrix
plot_confusion_matrix(RFY,X_test,y_test)

## Display RF output via graphviz
estimator = RFY.estimators_[4]
dot_data=tree.export_graphviz(estimator,out_file=None, filled=True)
graphRF = graphviz.Source(dot_data, format="png")
graphRF.render("RFR")
graphRF.view()

models = pd.DataFrame({'Model':['Decision Tree Classifier', 'XGBoost Classifier', 'Random Forest Classifier'],'Score':[A,B,C]})
models.sort_values(by='Score', ascending=False, ignore_index=True)



