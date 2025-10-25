# IMPLEMENTATION-OF-DECISION-TREE-CLASSIFIERS-MODEL-FOR-PREDICTING-EMPLOYEE-CHURN

## AIM:

To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## EQUIOMENTS REQUIRED :

1. Hardware – PCs

2. Anaconda – Python 3.7 Installation / Jupyter notebook

## ALGORITM :
1. Import the required libraries.

2. Upload and read the dataset.

3. Check for any null values using the isnull() function.

4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.
 
## PROGRAM :

```

/*

IMPLEMENTATION-OF-DECISION-TREE-CLASSIFIERS-MODEL-FOR-PREDICTING-EMPLOYEE-CHURN

DEVELOPED BY : ANTONY ABISHEK K

REGISTER NUMBER : 212223240009 

*/

```

```

import pandas as pd
data = pd.read_csv("Employee (1).csv")
data.head()

data.info()
data.isnull().sum()
data['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data['salary'] = le.fit_transform(data['salary'])
data.head()

x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()

y=data['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state =100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```
## OUTPUT :

## DATA HEAD :

<img width="2230" height="414" alt="image" src="https://github.com/user-attachments/assets/3b56fd3a-951a-42d7-a938-fab14f5af97b" />

## DATASET INFO :

<img width="2230" height="590" alt="image" src="https://github.com/user-attachments/assets/9a213b5d-a64a-498e-8c1c-283d3aaa09ec" />

## NULL DATASET :

<img width="2230" height="406" alt="image" src="https://github.com/user-attachments/assets/6cf9445e-e53a-48f4-8d8e-404b4b8d536b" />

## VALUES COUNT IN LEFT COLUMN :

<img width="2230" height="178" alt="image" src="https://github.com/user-attachments/assets/2bb7592e-302f-43a1-bb44-1685bd042b42" />

## DATASET TRANSFORMED HEAD :

<img width="2230" height="410" alt="image" src="https://github.com/user-attachments/assets/a9e2a716-12ef-45f5-9622-117c0545cb29" />

## X.HEAD :

<img width="2230" height="354" alt="image" src="https://github.com/user-attachments/assets/50bcf9b8-4d44-4b05-93b8-1dc12fa417dd" />


## ACCURACY :

<img width="2230" height="86" alt="image" src="https://github.com/user-attachments/assets/651d6610-e248-4b03-83ee-1e409bb9c1a7" />

## DATA PREDICTION :

<img width="2230" height="188" alt="image" src="https://github.com/user-attachments/assets/c26b3332-e62a-4c6f-b5ee-fc373a83f143" />

## RESULT :

Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
