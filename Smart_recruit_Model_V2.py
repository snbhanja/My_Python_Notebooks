###   Step 1 : Import required libraries and read test and train data set. Append both.

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc

train=pd.read_csv('G:/Udacity-ML/Smart Recruits/Train_pjb2QcD.csv')
test=pd.read_csv('G:/Udacity-ML/Smart Recruits/Test_wyCirpO.csv')


train['Type']='Train' #Create a flag for Train and Test Data set
test['Type']='Test'

fullData = pd.concat([train,test],axis=0) #Combined both Train and Test Data set


###Step 3: View the column names / summary of the dataset
fullData.columns # This will show all the column names
fullData.head(10) # Show first 10 records of dataframe
fullData.describe() #You can look at summary of numerical fields by using describe() function

###Step 4: Identify the a) ID variables b)  Target variables c) Categorical Variables
### d) Numerical Variables e) Other Variables

ID_col = ['ID']
target_col =  ['Business_Sourced']
cat_cols = ['Office_PIN', 'Application_Receipt_Date', 'Applicant_City_PIN',
       'Applicant_Gender', 'Applicant_BirthDate', 'Applicant_Marital_Status',
       'Applicant_Occupation', 'Applicant_Qualification', 'Manager_DOJ',
       'Manager_Joining_Designation', 'Manager_Current_Designation',
       'Manager_Grade', 'Manager_Status', 'Manager_Gender', 'Manager_DoB',
       'Manager_Num_Application', 'Manager_Num_Coded', 
       'Manager_Num_Products', 'Manager_Business2']
       
num_cols= list(set(list(fullData.columns))-set(cat_cols)-set(ID_col)-set(target_col))

other_col=['Type'] #Test and Train Data set identifier

###Step 5 : Identify the variables with missing values and create a flag for those

fullData.isnull().any()#Will return the feature with True or False,True means have missing value else False

num_cat_cols = num_cols+cat_cols # Combined numerical and Categorical variables
       
       #Create a new variable for each variable having missing value with VariableName_NA 
# and flag missing value with 1 and other with 0


for var in num_cat_cols:
    if train[var].isnull().any()==True:
        train[var+'_NA']=train[var].isnull()*1 


###Step 6 : Impute Missing values

#Impute numerical missing values with mean
fullData[num_cols] = fullData[num_cols].fillna(fullData[num_cols].mean(),inplace=True)

#Impute categorical missing values with -9999
fullData[cat_cols] = fullData[cat_cols].fillna(value = -9999)

### Step 7 : Create a label encoders for categorical variables and split the data set to train & test, 
### further split the train data set to Train and Validate

#create label encoders for categorical features
for var in cat_cols:
 number = LabelEncoder()
 fullData[var] = number.fit_transform(fullData[var].astype('str'))

#Target variable is also a categorical so convert it
fullData["Business_Sourced"] = number.fit_transform(fullData["Business_Sourced"].astype('str'))

train=fullData[fullData['Type']=='Train']
test=fullData[fullData['Type']=='Test']

train['is_train'] = np.random.uniform(0, 1, len(train)) <= .75
Train, Validate = train[train['is_train']==True], train[train['is_train']==False]

### Step 8 : Pass the imputed and dummy (missing values flags) variables into the modelling process.
### I am using random forest to predict the class

features=list(set(list(fullData.columns))-set(ID_col)-set(target_col)-set(other_col))

x_train = Train[list(features)].values
y_train = Train["Business_Sourced"].values
x_validate = Validate[list(features)].values
y_validate = Validate["Business_Sourced"].values
x_test=test[list(features)].values

random.seed(100)
rf = RandomForestClassifier(n_estimators=1000)
rf.fit(x_train, y_train)

### Step 9 : Check performance and make predictions

status = rf.predict_proba(x_validate)
fpr, tpr, _ = roc_curve(y_validate, status[:,1])
roc_auc = auc(fpr, tpr)
print (roc_auc)

final_status = rf.predict_proba(x_test)
test["Business_Sourced"]=final_status[:,1]
test.to_csv('G:/Udacity-ML/Smart Recruits/sample_submission.csv',columns=['ID','Business_Sourced'])

