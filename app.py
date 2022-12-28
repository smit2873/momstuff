
import streamlit as st 
import pandas 


######### Example 1
st.title("Are you a LinkedIn user? ")
educ = st.selectbox("Education level", 
              options = ["","less than highschool","High school incomplete","High School Graduate","Some college, no degree","Two year associate degree","Four year college or University","Some postgraduate or professional school"])

if educ == "less than highschool":
     educ = 1
elif educ == "High school incomplete":
     educ = 2
elif educ == "High School Graduate":
     educ = 3
elif educ == "Some college, no degree":
     educ = 4
elif educ == "Two year associate degree":
     educ = 5
elif educ == "Four year college or University":
     educ = 6
elif educ == "Some postgraduate or professional school":
     educ = 7
else: 
     educ = 8

#income
####################################
inc = st.selectbox("Income", 
              options = ["","Less than $10,000","10 to under $20,000","20 to under $30,000", "30 to under $40,000","40 to under $50,000","50 to under $75,000","75 to under $100,000","100 to under $150,000","$150,000 or more?"])


if inc == "Less than $10,000":
     inc = 1
elif inc == "10 to under $20,000":
     inc = 2
elif inc == "20 to under $30,000":
     inc = 3
elif inc == "30 to under $40,000":
     inc = 4
elif inc == "40 to under $50,000":
     inc = 5
elif inc == "50 to under $75,000":
     inc = 6
elif inc == "75 to under $100,000":
     inc = 7
elif inc == "100 to under $150,000":
     inc = 8
else: 
     inc = 9
#########################################

#Married?
mar = st.selectbox("Married", 
              options = ["","Yes", "No"])

if mar == "Yes":
     mar = 1
else: 
     mar = 0

#########################################
#Age?

age = st.slider(label="Enter age", 
           min_value=1,
           max_value=98,
           value=7)

 #########################################
#Degree
deg = st.selectbox("Degree", 
              options = ["","Yes", "No"])

if deg == "Yes":
     deg = 1
else: 
     deg = 0
#########################################
#Parent
par = st.selectbox("Parent", 
              options = ["","Yes", "No"])

if par== "Yes":
     par = 1
else: 
     par = 0
#########################################
#Females?
fem = st.selectbox("Female", 
              options = ["","Yes", "No"])

if fem == "Yes":
     fem = 1
else: 
     fem = 0


#######################Python Code###################################################
#!/usr/bin/env python
# coding: utf-8

# <h4>1)Read in the data, call the dataframe "s"  and check the dimensions of the dataframe</H4>

# In[200]:


import pandas 

s=pandas.read_csv("social_media_usage.csv")

# <h4>2)Define a function called clean_sm that takes one input, x, and uses `np.where` to check whether x is equal to 1. If it is, make the value of x = 1, otherwise make it 0. Return x. Create a toy dataframe with three rows and two columns and test your function to make sure it works as expected</h4>

# In[201]:


import numpy as np
def clean_sm(x):
    x=np.where(x == 1, 1, 0)
    return (x)
toy= np.array([[1, 2],[1,0],[1, 8]])
clean_sm(toy)



# <h4>3)Create a new dataframe called "ss". The new dataframe should contain a target column called sm_li which should be a binary variable ( that takes the value of 1 if it is 1 and 0 otherwise (use clean_sm to create this) which indicates whether or not the individual uses LinkedIn, and the following features: income (ordered numeric from 1 to 9, above 9 considered missing), education (ordered numeric from 1 to 8, above 8 considered missing), parent (binary), married (binary), female (binary), and age (numeric, above 98 considered missing). Drop any missing values. Perform exploratory analysis to examine how the features are related to the target.</h4>

# In[353]:


# Compile data into 1 table
ss = s[['web1h','income','educ2','par','marital','gender','age']].copy()
ss.columns = ['sm_li','income', 'education', 'parent', 'married',
                'female', 'age']
#Does Required Cleaning
ss['sm_li']=clean_sm(ss['sm_li'])
ss['income'] = np.where(ss['income'] <9,ss['income'],np.nan)
ss['education'] = np.where(ss['education'] < 8, ss['education'],np.nan)
ss['parent'] = np.where(ss['parent']== 1, 1, 0)
ss['married'] = np.where(ss['married']== 1, 1, 0)
ss['female'] = np.where(ss['female']== 1, 1, 0)
ss['age'] = np.where(ss['age']<98 , ss['age'], np.nan)
ss=ss.dropna() #Drop Na Values



#Exmploratory Graphs
(ss.groupby(['sm_li'])['age'].mean()) #The agerage age of linkedin users are 44
(ss.groupby(['income'])['sm_li'].sum()) # the 7th and 8th income range have the highests linkedin usage
print(ss.groupby(['sm_li'])['parent'].sum()) #144 non parents vs 64 parents
print(ss.groupby(['sm_li'])['married'].sum()) #261 non married vs 99 married use linkedin
print(ss.groupby(['sm_li'])['female'].sum()) #353 men while 134 women use linkeding
print(ss.groupby(['education'])['sm_li'].sum() )#The 6th education level has the highest linkeden usage




# <h4> 4)Create a target vector (y) and feature set (X)</h4>
# 

# In[354]:


X = ss[["income", "education", "parent", "married","female", "age"]]
y = ss["sm_li"]


# <h4> 5 Split the data into training and test sets. Hold out 20% of the data for testing. Explain what each new object contains and how it is used in machine learning</h4>

# In[355]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix



X_train, X_test, y_train, y_test= train_test_split(X, y, stratify = y, test_size = 0.2, random_state = 189)

# <h4> 6) Instantiate a logistic regression model and set class_weight to balanced. Fit the model with the training data.</h4>

# In[356]:


# Initialize algorithm 
lr = LogisticRegression(class_weight = 'balanced')
# Fit algorithm to training data
lr.fit(X_train, y_train)




# <h4> 7)Evaluate the model using the testing data. What is the model accuracy for the model? Use the model to make predictions and then generate a confusion matrix from the model. Interpret the confusion matrix and explain what each number means.</h4>




# Make predictions using the model and the testing data
y_pred = lr.predict(X_test)

# Get other metrics with classification_report
#print(classification_report(y_test, y_pred))



# confusion Matrix
#confusion_matrix(y_test, y_pred)






# <h4>8)Create the confusion matrix as a dataframe and add informative column names and index names that indicate what each quadrant represents</h4>

# In[358]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#conf=pandas.DataFrame(confusion_matrix(y_test, y_pred),index=["Act neg","Act pos"],
#columns=["Neg predict", "Pos Predict"])
#conf





# <H4> 9)Aside from accuracy, there are three other metrics used to evaluate model performance: precision, recall, and F1 score. Use the results in the confusion matrix to calculate each of these metrics by hand. Discuss each metric and give an actual example of when it might be the preferred metric of evaluation. After calculating the metrics by hand, create a classification_report using sklearn and check to ensure your metrics match those of the classification_report.</H4>

# In[359]:


print(classification_report(y_test, y_pred))


# <h4>Use the model to make predictions. For instance, what is the probability that a high income (e.g. income=8), with a high level of education (e.g. 7), non-parent who is married female and 42 years old uses LinkedIn? How does the probability change if another person is 82 years old, but otherwise the same?</h4<

# In[395]:


# New data for predictions
newdata = pandas.DataFrame({
    "income": [inc],
    "education": [educ],
    "parent": [par],
    "married":[mar],
    "female":[fem],
    "age":[age]
    
})




# Use model to make predictions
#lr.predict(newdata)
with st.form("key1"):
    # ask for input
    submit= st.form_submit_button("Click to find LinkedIn Prediction ")
newdata["prediction"]=lr.predict(newdata.iloc[:, :6].values)
ynew = lr.predict_proba(newdata.iloc[:, :6].values)
if submit:
	ans=lr.predict(newdata.iloc[:, :6].values)
	if ans==[1]:
		st.write('The probability that you will use Linkedin is:',round(ynew[0,1],2))
		st.title("You are a Linkedin user")
	else:
		st.write('The probability that you will use Linkedin is:',round(ynew[0,1],2))
		st.title("You are not a Linkedin user")
	
    



