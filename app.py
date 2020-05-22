# app.py
from flask import Flask           # import flask
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import jsonify
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import requests
import json
app = Flask(__name__)             # create an app instance

@app.route("/")                   # at the end point /
def hello():
    #-------------
    # dataset = pd.read_csv('Salary_Data.csv')
    # X = dataset.iloc[:, :-1].values
    # y = dataset.iloc[:, 1].values

    # #----------------
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

    # #--------------
    # from sklearn.linear_model import LinearRegression
    # regressor=LinearRegression()
    # regressor.fit(X_train,y_train)

    # #-------------
    # filename = 'finalized_model'
    # pickle.dump(regressor, open(filename, 'wb'))
    # loaded_model = pickle.load(open(filename, 'rb'))

    # #---------------
    # new_input=np.array([12])
    # new_input1=new_input.reshape(1,-1)
    # print(regressor.predict(new_input1))
    # predictedoutput= regressor.predict(new_input1)
    # #--------------
    # result = loaded_model.score(X_test, y_test)
    # print("Accuracy ",result)
    # #----------
    # # plt.scatter(X_test,y_test,color='red')
    # # plt.plot(X_train,regressor.predict(X_train),color='blue')
    # # plt.title('Salary vs Experience(Test Set)')
    # # plt.xlabel('Years of Experience')
    # # plt.ylabel('Salary')
    # # plt.show()
    # -*- coding: utf-8 -*-


        dataset = pd.read_csv('data2.csv')

        '''
        dataset['sex']=dataset['sex'].replace(['female','male','other'],[0,1,2])
        dataset['smoking']=dataset['smoking'].replace(['never','quit','yes'],[0,1,2])
        dataset['working']=dataset['working'].replace(['home','never','stopped','travel critical','travel non critical'],[1,0,2,3,4])
        dataset['income']=dataset['income'].replace(['blank','gov','high','med','low'],[0,4,3,2,1])'''

        X=dataset.iloc[:,0:16].values
        y=dataset.iloc[:,16].values


        from sklearn.preprocessing import OneHotEncoder
        from sklearn.compose import ColumnTransformer
        ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [0,4,5,9])],remainder='passthrough')
        X= np.array(ct.fit_transform(X), dtype=np.float)

        X=X[:,[0,1,3,4,5,6,8,9,11,12,13,15,16,17,18,19,20,21,22,23,24,25,26,27]]


        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,random_state=42)


        from sklearn.preprocessing import StandardScaler
        sc_X = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.transform(X_test)

        from sklearn.linear_model import LinearRegression
        regressor=LinearRegression()
        regressor.fit(X_train,y_train)

        y_pred = regressor.predict(X_test)
        print("Accuracy for training set",regressor.score(X_train,y_train)*100)
        print("Accuracy for test set",regressor.score(X_test,y_test)*100)

        '''
        gender=int(input("Enter 0/1/2 for female/male/other:"))
        age=int(input("Enter age:"))
        height=int(input("Enter height:"))
        weight=int(input("Enter weight:"))
        income=int(input("Enter 0/1/2/3/4 if- no income/gov/high/med/low income:"))
        smoking=int(input("Enter 0/1/2 for  smoking- never/quit/yes:"))
        alcohol=int(input("Enter the number of times you've consumed alcohol in the last 14 days, if not then 0:"))
        contacts=int(input("Enter total contacts:"))
        totalpeople=int(input("Enter total people in your house:"))
        working=int(input("Enter 0/1/2/3/4 for working options- home/never/stopped/travel critical/travel non critical:"))
        masks=int(input("Enter total no. of masks you have:"))
        symptoms=int(input("Enter 0 if you have no covid19 symptoms else 1:"))
        contactsinfected=int(input("Enter 0 if you have not been in contact with an infected person else 1:"))
        asthma=int(input("Enter 0 if you dont have asthma else 1:"))
        lung=int(input("Enter 0 if you have dont any other lung disease else 1:"))
        healthworker=int(input("Enter 0 if you're not a health worker else 1:"))'''

        new_input=np.array([0,1,0,0,0,1,0,0,0,0,1,0,37,166,62,9,2,3,4,0,0,0,0,0])
        new_input1=new_input.reshape(1,-1)
        new_output = regressor.predict(new_input1)
        print("\nRisk of death:", new_output/100,"%")
        freqs = {
        'predictedoutput': new_output[0]/100
        # 'accuracy': result,
        }
        return  jsonify(freqs)  
        # username = request.args.get('username')
        # password = request.args.get('password')       # which returns "hello world"
if __name__ == "__main__":        # on running python app.py
    app.run()                     # run the flask app