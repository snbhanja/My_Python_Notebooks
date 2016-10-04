
# coding: utf-8

# In[1]:

get_ipython().magic('pylab inline')


# ## Import Libraries

# In[2]:

import pandas as pd
import numpy as np
import matplotlib as plt


# ## Read the dataset

# In[3]:

train = pd.read_csv("G:/Udacity-ML/loan-prediction-iii_V2/train.csv") #Reading the dataset in a dataframe using Pandas


# In[4]:

test = pd.read_csv("G:/Udacity-ML/loan-prediction-iii_V2/test.csv")


# ## Data Exploration

# In[5]:

train.head(10)


# In[6]:

train.describe()


# describe() function shows details about numeric columns. But for categorical columns we can get frequency distribution

# In[7]:

train['Gender'].value_counts()


# In[8]:

train['Married'].value_counts()


# In[9]:

train['Dependents'].value_counts()


# In[10]:

train['Education'].value_counts()


# In[11]:

train['Self_Employed'].value_counts()


# In[12]:

train['Credit_History'].value_counts()


# In[13]:

train['Property_Area'].value_counts()


# # Distribution analysis

# Lets start by plotting the histogram of ApplicantIncome using the following commands

# In[15]:

train['ApplicantIncome'].hist(bins=50)


# Next we will see box plot of ApplicantIncome column

# In[16]:

train.boxplot(column='ApplicantIncome')


# This confirms the presence of a lot of outliers/extreme values.

# Let us segregate them by Education:

# In[18]:

train.boxplot(column='ApplicantIncome', by = 'Education')


# We can see that there is no substantial different between the mean income of graduate and non-graduates but there are many outliers.
# 
# 

# ---------------------------------

# Let’s look at the histogram and boxplot of LoanAmount using the following command:

# In[20]:

train['LoanAmount'].hist(bins=50)


# In[21]:

train.boxplot(column='LoanAmount')


# Plot a box plot for variable LoanAmount by variable Gender of training data set

# In[23]:

train.boxplot(column='LoanAmount', by = 'Gender')


# ---------------------------

# --  ###  Categorical variable analysis

# In[25]:

pd.crosstab( train ['Gender'], train ["Loan_Status"], margins=True)


# we can also look at proportions can be more intuitive in making some quick insights

# In[26]:

def percentageConvert(ser):
  return ser/float(ser[-1])

pd.crosstab(train ["Gender"], train ["Loan_Status"], margins=True).apply(percentageConvert, axis=1)


#  --- Two-way comparison: Credit History and Loan Status

# In[27]:

pd.crosstab(train ["Credit_History"], train ["Loan_Status"], margins=True).apply(percentageConvert, axis=1)


# ## Handling missing values

# Check missing values in the dataset

# In[28]:

train.isnull().sum()


# -------

# build a supervised learning model to predict loan amount on the basis of other variables and then use age along with other variables to predict survival.

# A key hypothesis is that the whether a person is educated or self-employed can combine to give a good estimate of loan amount.

# First, let’s look at the boxplot to see if a trend exists:

# In[31]:

train.boxplot(column='LoanAmount', by = ['Education','Self_Employed'])


#  Thus we see some variations in the median of loan amount for each group and this can be used to impute the values. 
# But first, we have to ensure that each of Self_Employed and Education variables should not have a missing values.

# For 'Self_Employed' the occurance of 'No' is more than 'Yes'. So, ipute missing values with No

# In[32]:

train['Self_Employed'].fillna('No',inplace=True)


# Now, we will create a Pivot table, which provides us median values for all the groups of unique values of Self_Employed 
#  and Education features. Next, we define a function, which returns the values of these cells and apply it to fill the missing 
#  values of loan amount:

# In[33]:

table = train.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median)


# Define function to return value of this pivot_table

# In[34]:

def fage(x):
 return table.loc[x['Self_Employed'],x['Education']]


# Replace missing values

# In[35]:

train['LoanAmount'].fillna(train[train['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)


# --------

#  Lets Impute other variables

# Since 489/614= 79.6% are male, so lets imute missing values with 'male' 

# In[39]:

train['Gender'].fillna('Male',inplace=True)


# In[40]:

train['Married'].fillna('Yes',inplace=True)


# In[41]:

train['Dependents'].fillna('0',inplace=True)


# In[42]:

train['Loan_Amount_Term'].fillna('360',inplace=True)


# In[43]:

train['Credit_History'].fillna('1.0',inplace=True)


# In[44]:

train.isnull().sum()


# Now all the columns having NaN imputed.

# --------

# ## Treat Outliers or Extreme values

# Columns ApplicantIncome and CoapplicantIncome have outliers or extreme values.

# Lets check the box plots

# In[46]:

train['ApplicantIncome'].hist(bins = 50)


# In[55]:

train['CoapplicantIncome'].hist(bins = 50)


# In[48]:

train['LoanAmount'].hist(bins = 50)


# As we see all these 3 columns have outliers ,  let’s try a log transformation to nullify their effect:

# Add both ApplicantIncome and CoapplicantIncome to TotalIncome

# In[50]:

train['TotalIncome'] = train['ApplicantIncome'] + train['CoapplicantIncome']


# Perform log transformation of TotalIncome and LoanAmount to make it closer to normal

# In[51]:

train['TotalIncome_log']= np.log(train['TotalIncome'])


# In[52]:

train['TotalIncome_log'] = np.log(train['LoanAmount'])


# Check the distribution of LoanAmount_log and TotalIncome_log

# In[54]:

train['LoanAmount_log'].hist(bins=20)


# In[56]:

train['TotalIncome_log'].hist(bins=20)


# Also, I encourage you to think about possible additional information which can be derived from the data.
#   For example, creating a column for LoanAmount/TotalIncome might make sense as it gives an idea of how well the applicant 
#  is suited to pay back his loan.   

# In[57]:

train['LaTi'] = train['LoanAmount']/train['TotalIncome']


# ---------------------

# ## Imputing missing values and outlier normalization for test data set

# In[71]:

test.head(10)


# In[72]:

test.describe()


# In[73]:

test.isnull().sum()


# So, we have to impute values for categorical columns Gender, Dependents, Self_Employed, Credit_History and numeric term LoanAmount             

# In[75]:

test['Gender'].value_counts()


# In[80]:

test['Gender'].fillna('Male',inplace=True)


# In[78]:

test['Dependents'].value_counts()


# In[81]:

test['Dependents'].fillna('0',inplace=True)


# In[82]:

test['Self_Employed'].value_counts()


# In[83]:

test['Self_Employed'].fillna('No',inplace=True)


# In[84]:

test['Credit_History'].value_counts()


# In[86]:

test['Credit_History'].fillna('1.0',inplace=True)


# In[91]:

test['Loan_Amount_Term'].value_counts()


# In[92]:

test['Loan_Amount_Term'].fillna('360.0',inplace=True)


# Now we will impute missing values of LoanAmount

# In[87]:

test.boxplot(column='LoanAmount', by = ['Education','Self_Employed'])


# In[88]:

table = test.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median)


# In[89]:

test['LoanAmount'].fillna(test[test['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)


# In[93]:

test.isnull().sum()


# -------------------

#    
#     
#     

# ## Building a Predictive Model in Python

# Library "Scikit Learn" only works with numeric array. Hence, we need to label all the character variables into a numeric array. For example Variable "Gender" has two labels "Male" and "Female". Hence, we will transform the labels to number as 1 for "Male" and 0 for "Female".
# 
# "Scikit Learn" library has a module called "LabelEncoder" which helps to label character labels into numbers so first import module "LabelEncoder".

# In[59]:

from sklearn.preprocessing import LabelEncoder


# In[60]:

var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']


# In[61]:

number = LabelEncoder()


# In[62]:

for i in var_mod:
    train[i] = number.fit_transform(train[i])


# Now check the data type of categorical variables

# In[63]:

train.dtypes


# -------

# ----   LabelEncode columns of test dataset   ----------->

# In[96]:

var_mod2 = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area']


# In[98]:

for i in var_mod2:
    test[i] = number.fit_transform(test[i])


# In[99]:

test.dtypes


# -----------------------

# Now both train and test datset are ready for Model preparation.

# We are going to model our trainset with Random Forest Calssifier

# -----We dont know which varaibles are important so we will take all features to train with Random Forest-----

# Import module for Random Forest

# In[116]:

from sklearn.cross_validation import KFold   #For K-fold cross validation
import sklearn.ensemble #For randomForest
from sklearn import metrics

predictors=['Credit_History','Dependents', 'Education', 'Gender', 'LoanAmount_log',
            'Loan_Amount_Term', 'Married', 'Property_Area', 'Self_Employed', 'TotalIncome_log']

# Converting predictors and outcome to numpy array
x_train = train[predictors].values
y_train = train['Loan_Status'].values

# Model Building
model = sklearn.ensemble.RandomForestClassifier(n_estimators=100)
model.fit(x_train, y_train)


# In[117]:

featimp = pd.Series(model.feature_importances_, index=predictors).sort_values(ascending=False)

print (featimp)


# In[127]:

#Generic function for making a classification model and accessing performance:


# In[129]:

def classification_model(model, data, predictors, outcome):
  #Fit the model:
  model.fit(data[predictors],data[outcome])
  
  #Make predictions on training set:
  predictions = model.predict(data[predictors])
  
  #Print accuracy
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  print ("Accuracy : %s " % "{0:.3%}".format(accuracy))

  #Perform k-fold cross-validation with 5 folds
  kf = KFold(data.shape[0], n_folds=5)
  error = []
  for train, test in kf:
    # Filter training data

    train_predictors = (data[predictors].iloc[train,:])
    
    # The target we're using to train the algorithm.
    train_target = data[outcome].iloc[train]
    
    # Training the algorithm using the predictors and target.
    model.fit(train_predictors, train_target)
    
    #Record error from each cross-validation run
    error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
 
  print ("Cross-Validation Score : %s " % "{0:.3%}".format(np.mean(error)))

  #Fit the model again so that it can be refered outside the function:
  model.fit(data[predictors],data[outcome]) 


# In[131]:

from sklearn.ensemble import RandomForestClassifier


# In[139]:

model = RandomForestClassifier(n_estimators=25, min_samples_split=25, max_depth=7, max_features=1)
predictor_var = ['TotalIncome_log','LoanAmount_log','Credit_History','Dependents','Property_Area']
classification_model(model, train,predictor_var,outcome_var)


# In[141]:

predictor_var


# In[143]:

outcome_var


# lets predict in test set . Fingers crossed :p

# In[145]:

test.head(10)


# In[147]:

test = test.drop('Loan_Status', 1)


# In[148]:

test.head(10)


# In[150]:

test.dtypes


# In[152]:

# Converting predictors and outcome to numpy array
test_predictors = ['Credit_History','Dependents','Property_Area']
x_test = test[test_predictors].values


# In[153]:


#Predict Output
predicted= model.predict(x_test)


# ------------We got a problem

# In[155]:

test['TotalIncome'] = test['ApplicantIncome'] + test['CoapplicantIncome']


# In[156]:

test['TotalIncome_log']= np.log(test['TotalIncome'])


# In[157]:

test['LoanAmount_log']= np.log(test['LoanAmount'])


# In[158]:

# Converting predictors and outcome to numpy array
x_test = test[predictors].values




# In[159]:

#Predict Output
predicted= model.predict(x_test)


# In[162]:

#Output file to make submission
test.to_csv("G:/Udacity-ML/loan-prediction-iii_V2/Submission.csv",columns=['Loan_ID','Loan_Status'])


# In[163]:


#Reverse encoding for predicted outcome
predicted = number.inverse_transform(predicted)


# In[164]:

#Store it to test dataset
test['Loan_Status']=predicted


#Output file to make submission
test.to_csv("G:/Udacity-ML/loan-prediction-iii_V2/Submission.csv",columns=['Loan_ID','Loan_Status'])


# In[165]:

test.head(10)


# In[166]:

#Coding LoanStatus as Y=1, N=0:
print 'Before Coding:'
print pd.value_counts(test["Loan_Status"])
test["Loan_Status"] = coding(test["Loan_Status"], {0:'N',1:'Y'})
print '\nAfter Coding:'
print pd.value_counts(test["Loan_Status"])


# In[167]:

#Coding LoanStatus as 1=Y, 0=N:
print ('Before Coding:')
print pd.value_counts(test["Loan_Status"])
test["Loan_Status"] = coding(test["Loan_Status"], {0:'N',1:'Y'})
print ('\nAfter Coding:')
print pd.value_counts(test["Loan_Status"])


# In[168]:

test["Loan_Status"] = coding(test["Loan_Status"], {0:'N',1:'Y'})


# In[169]:

def coding(col, codeDict):
  colCoded = pd.Series(col, copy=True)
  for key, value in codeDict.items():
    colCoded.replace(key, value, inplace=True)
  return colCoded


# In[170]:

test["Loan_Status"] = coding(test["Loan_Status"], {0:'N',1:'Y'})


# In[171]:

test.head(10)


# In[172]:

#Output file to make submission
test.to_csv("G:/Udacity-ML/loan-prediction-iii_V2/Submission.csv",columns=['Loan_ID','Loan_Status'])


# In[ ]:



