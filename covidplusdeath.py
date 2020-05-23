# -*- coding: utf-8 -*-
"""
Created on Sat May 23 11:16:54 2020

@author: Devang  Mehrotra
"""

import numpy as np
import pandas as pd

dataset = pd.read_csv('data1.csv')

dataset['sex']=dataset['sex'].replace(['female','male','other'],[0,1,2])
dataset['smoking']=dataset['smoking'].replace(['never','quit','yes'],[0,1,2])
dataset['working']=dataset['working'].replace(['home','never','stopped','travel critical','travel non critical'],[1,0,2,3,4])
dataset['income']=dataset['income'].replace(['blank','gov','high','med','low'],[0,4,3,2,1])
dataset['blood_type']=dataset['blood_type'].replace(['unknown','abn','abp','an','ap','bn','bp','on','op'],[0,1,2,3,4,5,6,7,8])
dataset['insurance']=dataset['insurance'].replace(['no','yes'],[0,1])

X1=dataset.iloc[:,[0,1,2,3,4,5,6,9,10,12,18,19,20,21,26,32]].values
y1=dataset.iloc[:,33].values
X2=dataset.iloc[:,0:33].values
y2=dataset.iloc[:,34].values

from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size = 0.25,random_state=42)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size = 0.25,random_state=42)

from sklearn.linear_model import LinearRegression
regressor1=LinearRegression()
regressor2=LinearRegression()
regressor1.fit(X_train1,y_train1)
regressor2.fit(X_train2,y_train2)

y_pred1 = regressor1.predict(X_test1)
y_pred2 = regressor2.predict(X_test2)
print("Accuracy for covid risk training set",regressor1.score(X_train1,y_train1)*100,"%")
print("Accuracy for covid risk test set",regressor1.score(X_test1,y_test1)*100,"%")
print("Accuracy for death risk training set",regressor2.score(X_train2,y_train2)*100,"%")
print("Accuracy for death risk test set",regressor2.score(X_test2,y_test2)*100,"%")

#put url here in the arrays
#covid risk
new_input_covid=np.array([[gender,age,height,weight,income,smoking,alcohol,contacts,housecount,working,masks,symptoms,contactsinfected,asthma,lung,health_worker]])
new_input_covid1=new_input_covid.reshape(1,-1)
new_output_covid= regressor1.predict(new_input_covid1)
print("\nRisk of getting covid-19", abs(new_output_covid),"%")

#death risk
new_input_death=np.array([[[gender,age,height,weight,income,smoking,alcohol,blood_type,insurance,contacts_count,housecount,public_transport,working,self_social_distancing,self_washing_hands,house_social_distancing,house_washing_hands,sanitizer,masks,symptoms,contactsinfected,asthma,kidney_disease,liver_disease,compromised_immune,heart_disease,lung,diabetes,hiv_positive,hypertension,other_chronic,nursing_home,health_worker]]])
new_input_death1=new_input_death.reshape(1,-1)
new_output_death= regressor2.predict(new_input_death1)
print("\nRisk of dying from covid-19", abs(new_output_death),"%")

'''
inputs for-

gender=Enter 0/1/2 for female/male/other
age=Enter age
height=Enter height in cms
weight=Enter weight in kgs
income=Enter 0/1/2/3/4 if- no income/low/med/high/gov income
smoking=Enter 0/1/2 for smoking- never/quit/yes
alcohol=Enter the number of times you've consumed alcohol in the last 14 days, if not then 0
blood_type= Enter 0/1/2/3/5/6/7/8 for your blood type- unknown/abn/abp/an/ap/bn/bp/on/op
insurance=Enter 0/1 if you- dont have/have insurance
contacts=Enter total contacts
housecount=Enter total people in your house
public_transport=Enter total number of times you used public transport in the last week
working=Enter 0/1/2/3/4 for working options- never/home/stopped/travel critical/travel non critical
self_social_distancing= Enter 0/1/2 if you- avoid/sometimes/regularly follow social distancing 
self_washing_hands= Enter 0/1/2 if you- dont/sometimes/regularly wash your hands
house_social_distancing= Enter 0/1/2 if your family - doesnt/sometimes/regulary follow soaicl distancing 
house_washing_hands=Enter 0/1/2 if your family- dont/sometimes/regularly wash hands
sanitizer= Enter total sanitizers available to you
masks=Enter total no. of masks you have
symptoms=Enter 0 if you have no covid19 symptoms else 1
contactsinfected= Enter 0 if you have not been in contact with an infected person else 1
asthma=Enter 0 if you dont have asthma else 1
kidney_disease= Enter 0 if you dont have any kindey disease else 1
liver_disease=Enter 0 if you dont have any liver disease else 1
compromised_immune=Enter 0 if you dont have a compromised immune system else 1
heart_disease=Enter 0 if you dont have any heart disease else 1
lung=Enter 0 if you have dont have any other lung disease else 1
diabetes=Enter 0 if you have dont have diabetes else 1
hiv_positive=Enter 0 if you are not HIV positive else 1
hypertension=Enter 0 if you have dont have hypertension else 1
other_chronic=Enter 0 if you have dont have any other chronic disease else 1
nursing_home=Enter 0 if you dont work in a nursing home else 1
health_worker=Enter 0 if you're not a health worker else 1'''

# gen=0&age=40&hgt=132&wgt=40&inc=4&smk=1&alc=12&bt=8&ins=1&concnt=20&hcnt=4&ptrnspt=5&wrkng=3&ssocdis=1&swashnds=1&hsocdis=1&hwashnds=1&san=3&masks=2&sym=1&coninf=0&asthma=0&kidney=0&liver=0&cmpimm=0&heart=1&lung=0&diabetes=0&hiv=0&hyprt=1&otherchr=0&nrsng=0&hwork=0