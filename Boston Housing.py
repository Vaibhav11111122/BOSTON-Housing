import pandas as pd
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df = pd.read_csv('/housing.csv',header=None, delimiter=r"\s+", names=column_names)
df.head()
print(df)
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
bos = load_boston
lrg = LinearRegression()
le =  LabelEncoder()
x = df.drop('MEDV',axis=1)
x = df.drop('CHAS',axis=1)
x = df.drop('LSTAT',axis=1)
print(x)
#x = df.drop('RM',axis=1)
g = pd.cut(df['MEDV'],3,labels=['0','1','2'])
le.fit(g)
y = le.transform(g)
correlation_matrix = df.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True)
**Linear Regression**
X_train, X_test, Y_train, Y_test = train_test_split(x,y,random_state=5,test_size=0.2)
train = lrg.fit(X_train, Y_train)
print(train)
y_pred = lrg.predict(X_test)
print(y_pred)
print(mean_squared_error(Y_test,y_pred))
train = lrg.fit(X_train, Y_train)
print(train)
y_pred = lrg.predict(X_test)
print(y_pred)
print(mean_squared_error(Y_test,y_pred))







