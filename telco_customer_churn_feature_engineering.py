###########################################
# Telco Customer Churn Feature Engineering
###########################################


# Problem: You are tasked with developing a machine learning model to predict which customers will leave the company (customer churn).
# Before building the model, you are expected to perform the necessary data analysis and feature engineering steps.

# The Telco customer churn data contains information about a fictional telecommunications company providing home phone and internet services to 7,043 customers in California during the third quarter.
# It includes data on which customers have left, stayed, or signed up for services.

# 21 variables and 7,043 observations

# CustomerId: Customer ID
# Gender: Gender
# SeniorCitizen: Whether the customer is a senior citizen (1, 0)
# Partner: Whether the customer has a partner (Yes, No) – indicates whether the customer is married
# Dependents: Whether the customer has dependents (Yes, No) – includes children, parents, grandparents
# tenure: Number of months the customer has stayed with the company
# PhoneService: Whether the customer has phone service (Yes, No)
# MultipleLines: Whether the customer has multiple lines (Yes, No, No phone service)
# InternetService: The customer's internet service provider (DSL, Fiber optic, No)
# OnlineSecurity: Whether the customer has online security (Yes, No, No internet service)
# OnlineBackup: Whether the customer has online backup (Yes, No, No internet service)
# DeviceProtection: Whether the customer has device protection (Yes, No, No internet service)
# TechSupport: Whether the customer has tech support (Yes, No, No internet service)
# StreamingTV: Whether the customer streams TV (Yes, No, No internet service) – Indicates if the customer uses internet service to stream TV shows from a third-party provider
# StreamingMovies: Whether the customer streams movies (Yes, No, No internet service) – Indicates if the customer uses internet service to stream movies from a third-party provider
# Contract: The customer's contract term (Month-to-month, One year, Two years)
# PaperlessBilling: Whether the customer has paperless billing (Yes, No)
# PaymentMethod: The customer's payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
# MonthlyCharges: The amount charged to the customer monthly
# TotalCharges: The total amount charged to the customer
# Churn: Whether the customer has churned (Yes, No) – Indicates if the customer left the service in the last month or quarter

# Each row represents a unique customer. The variables contain information about customer service subscriptions, account details, and demographic data:
# Customer service information: Phone service, multiple lines, internet service, online security, online backup, device protection, tech support, and streaming TV and movies.
# Customer account information: How long they’ve been a customer, contract type, payment method, paperless billing, monthly charges, and total charges.
# Customer demographics: Gender, senior status, partner status, and whether they have dependents.



# Required Library and Functions
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from datetime import date
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,LabelEncoder,StandardScaler,RobustScaler
import warnings
warnings.simplefilter(action='ignore')

pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)
pd.set_option("display.float_format",lambda x:"%.3f" % x)
pd.set_option("display.width",500)

df=pd.read_csv('datasets/Telco-Customer-Churn.csv')

df.head()
df.shape
df.info()

# Totalcharges must be a numeric variable
df["TotalCharges"]=pd.to_numeric(df["TotalCharges"],errors="coerce")

df["Churn"]=df["Churn"].apply(lambda x: 1 if x=="Yes" else 0)


####################################################################
# EDA - Exploratory Data Analysis
####################################################################

##########################################
# General Picture
##########################################

def check_df(dataframe,head=5):
    print('########################### Shape ###########################')
    print(dataframe.shape)
    print('########################### Types ###########################')
    print(dataframe.dtypes)
    print('########################### Head ###########################')
    print(dataframe.head(head))
    print('########################### Tail ###########################')
    print(dataframe.tail(head))
    print('########################### NA ###########################')
    print(dataframe.isnull().sum())
    print('########################### Quantiles ###########################')
    numeric_columns = dataframe.select_dtypes(include=['float64', 'int64'])
    if not numeric_columns.empty:
        print(numeric_columns.quantile([0, 0.05, 0.5, 0.95, 0.99, 1]).T)


check_df(df)

###########################################
# Define Numeric and Categorical Variables
###########################################

def grab_col_names(dataframe, cat_th=10,car_th=20):
    """
    Define the names of categorical, numerical, and categorical but cardinal variables in the dataset.
    Note: Categorical variables also include those that appear numeric but are actually categorical

    Parameters
    ----------
    dataframe: dataframe
                dataframe from which variable names are to be extracted
    cat_th: int, optional
                Threshold value for the number of classes for variables that are numerical but actually categorical
    car_th: int,optional
                Threshold value for the number of classes for variables that are categorical but cardinal

    Returns
    -------
        cat_cols: list
                List of categorical variables
        num_cols: list
                List of numerical variables
        cat_but_car: list
                List of cardinal variables that appear categorical


    Examples
    --------
        import seaborn as sns
        df=sns.load_dataset("iris")
        print(grab_col_names(df))

    Notes
    ------
        cat_cols + num_cols + cat_but_car = Total number of variables
        num_but_cat is within cat_cols'
    """

    #cat_cols, cat_but_car
    cat_cols=[col for col in dataframe.columns if dataframe[col].dtypes == 'O']
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique()< cat_th and dataframe[col].dtypes != 'O']
    cat_but_car=[col for col in dataframe.columns if dataframe[col].nunique()>car_th and dataframe[col].dtypes =='O']
    cat_cols=cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    #num_cols
    num_cols=[col for col in dataframe.columns if dataframe[col].dtypes != 'O']
    num_cols=[col for col in num_cols if col not in num_but_cat]

    print(f'Observations: {dataframe.shape[0]}')
    print(f'Variables: {dataframe.shape[1]}')
    print(f'cat_cols {len(cat_cols)}')
    print(f'num_cols {len(num_cols)}')
    print(f'cat_but_car {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols,num_cols,cat_but_car

cat_cols,num_cols,cat_but_car=grab_col_names(df)


###########################################
# Analysis of Categorical Variables
###########################################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        '`Ratio':100 * dataframe[col_name].value_counts()/len(dataframe)}))
    print('**********************************************')
    if plot:
        sns.countplot(x=dataframe[col_name],data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df,col)


###########################################
# Analysis of Numerical Variables
###########################################

def num_summary(dataframe,numerical_col,plot=False):
    quantiles=[0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95,0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df,col)


###########################################
# Target Analysis of Numerical Variables
###########################################

def target_summary_with_num(dataframe,target,numerical_col):
    print(dataframe.groupby(target).agg({numerical_col:'mean'}),end='\n\n\n')

for col in num_cols:
    target_summary_with_num(df,"Churn",col)


###########################################
# Target Analysis of Categorical Variables
###########################################

def target_summary_with_cat(dataframe,target,categorical_col):
    print(categorical_col)
    print(pd.DataFrame({
        "TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
        "COUNT":dataframe[categorical_col].value_counts(),
        "RATIO":100*dataframe[categorical_col].value_counts()/len(dataframe)
    }), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df,"Churn",col)


###########################################
# Correlation
###########################################

df[num_cols].corr()

numeric_cols = df.select_dtypes(include=[np.number])

# Correlation Matrix
f,ax=plt.subplots(figsize=[18,13])
sns.heatmap(numeric_cols.corr(),annot=True, fmt='.2f',ax=ax,cmap='magma')
ax.set_title('Correlation Matrix',fontsize=20)
plt.show()

#TotalCharges monthly charges appear to be highly correlated with tenure
numeric_cols.corrwith(df["Churn"]).sort_values(ascending=False)

###############################################################
# Feature Engineering
###############################################################

###########################################
# Missing Value Analysis
###########################################

df.isnull().sum()

def missing_values_table(dataframe,na_name=False):
    na_columns=[col for col in dataframe.columns if dataframe[col].isnull().sum()>0]
    n_miss=dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio=(dataframe[na_columns].isnull().sum()/dataframe.shape[0]*100).sort_values(ascending=False)
    missing_df=pd.concat([n_miss,np.round(ratio,2)],axis=1,keys=['n_miss','ratio'])
    print(missing_df,end='\n')
    if na_name:
        return na_columns

na_columns=missing_values_table(df,na_name=True)

# Totalcharge can be filled with the amount to be paid monthly.
df["TotalCharges"].fillna(df["TotalCharges"].median(),inplace=True)

df.isnull().sum()

###########################################
# Base Model Setup
###########################################

dff=df.copy()
cat_cols=[ col for col in cat_cols if col not in ["Churn"]]

def one_hot_encoder(df,categorical_cols,drop_first=True):
    df=pd.get_dummies(df,columns=categorical_cols,drop_first=drop_first)
    dummy_columns = df.filter(regex='|'.join(categorical_cols)).columns
    df[dummy_columns] = df[dummy_columns].astype(int)
    return df

dff=one_hot_encoder(dff,cat_cols,drop_first=True)


y=dff['Churn']
X=dff.drop(['Churn',"customerID"],axis=1)
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.30,random_state=17)

catboost_model=CatBoostClassifier(verbose=False,random_state=12345).fit(X_train,y_train)
y_pred=catboost_model.predict(X_test)

print(f'Accuracy: {round(accuracy_score(y_pred,y_test),2)}')
print(f'Recall: {round(recall_score(y_pred,y_test),3)}')
print(f'Precision: {round(precision_score(y_pred,y_test),2)}')
print(f'F1: {round(f1_score(y_pred,y_test),2)}')
print(f'Auc: {round(roc_auc_score(y_pred,y_test),2)}' )

#Base Model
# Accuracy: 0.78
# Recall: 0.633
# Precision: 0.49
# F1: 0.55
# Auc: 0.73

###########################################
# Outlier Analysis
###########################################

def outlier_threshold(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe,col_name):
    low_limit, up_limit =outlier_threshold(dataframe,col_name)
    if dataframe[(dataframe[col_name]>up_limit) | (dataframe[col_name]<low_limit)].any(axis=None):
        return True
    else:
        return False


def replace_with_threshold(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_threshold(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

###########################################
# Outlier Analysis and Suppression Process
###########################################
for col in num_cols:
    print(col,check_outlier(df,col))
    if check_outlier(df,col):
        replace_with_threshold(df,col)

###########################################
# Feature Extraction
###########################################

# Creating annual categorical variable from tenure variable
df.loc[(df['tenure']>=0) & (df['tenure']<=12),'NEW_TENURE_YEAR']='0-1 Year'
df.loc[(df['tenure']>12) & (df['tenure']<=24),'NEW_TENURE_YEAR']='1-2 Year'
df.loc[(df['tenure']>24) & (df['tenure']<=36),'NEW_TENURE_YEAR']='2-3 Year'
df.loc[(df['tenure']>36) & (df['tenure']<=48),'NEW_TENURE_YEAR']='3-4 Year'
df.loc[(df['tenure']>48) & (df['tenure']<=60),'NEW_TENURE_YEAR']='4-5 Year'
df.loc[(df['tenure']>60) & (df['tenure']<=72),'NEW_TENURE_YEAR']='5-6 Year'

# Indicate customers with a 1 or 2 year contract as Engaged
df["NEW_Engaged"] = df["Contract"].apply(lambda x: 1 if x in ["One year","Two year"] else 0)

# Persons who do not receive any support, backup or protection
df['NEW_noProt']=df.apply(lambda x:1 if (x['OnlineBackup'] != 'Yes') or (x['DeviceProtection']!='Yes') or (x['TechSupport']!='Yes') else 0, axis=1)

# Customers who have a monthly contract and are young
df['NEW_Young_Not_Engaged']=df.apply(lambda x:1 if (x['NEW_Engaged']==0) and (x['SeniorCitizen']==0) else 0,axis=1)

# Total number of services received by the person
df['NEW_TotalServices']=(df[['PhoneService','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']]=='Yes').sum(axis=1)

# People who receive any streaming service
df['NEW_FLAG_ANY_STREAMING']=df.apply(lambda x: 1 if (x['StreamingTV']=='Yes') or (x['StreamingMovies']=='Yes') else 0,axis=1)

# Does a person make automatic payments?
df['NEW_FLAG_AutoPayment']=df['PaymentMethod'].apply(lambda x:1 if x in ['Bank transfer (automatic)','Credit card (automatic)'] else 0)

# Average monthly payment
df['NEW_AVG_Charges']=df['TotalCharges']/(df['tenure']+1)

# Current Price increase compared to average price
df['NEW_Increase']=df['NEW_AVG_Charges']/df['MonthlyCharges']

# Fee per service
df['NEW_AVG_Service_Fee']=df['MonthlyCharges']/(df['NEW_TotalServices']+1)

df.head()
df.shape

###########################################
# Encoding
###########################################

# The process of separating variables according to their types
cat_cols,num_cols,cat_but_car=grab_col_names(df)

# Label Encoding
def label_encoder(dataframe,binary_col):
    labelencoder=LabelEncoder()
    dataframe[binary_col]=labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols=[col for col in df.columns if df[col].dtypes == 'O' and df[col].nunique()==2]

for col in binary_cols:
    df=label_encoder(df,col)

# One-Hot Encoding Process
# Updating cat_cols list
cat_cols=[col for col in cat_cols if col not in binary_cols and col not in ['Churn','NEW_TotalServices']]


df=one_hot_encoder(df,cat_cols,drop_first=True)

df.shape
df.head()

y=df['Churn']
X=df.drop(['Churn',"customerID"],axis=1)
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.30,random_state=17)

catboost_model=CatBoostClassifier(verbose=False,random_state=12345).fit(X_train,y_train)
y_pred=catboost_model.predict(X_test)

print(f'Accuracy: {round(accuracy_score(y_pred,y_test),2)}')
print(f'Recall: {round(recall_score(y_pred,y_test),3)}')
print(f'Precision: {round(precision_score(y_pred,y_test),2)}')
print(f'F1: {round(f1_score(y_pred,y_test),2)}')
print(f'Auc: {round(roc_auc_score(y_pred,y_test),2)}' )

# Accuracy: 0.79
# Recall: 0.65
# Precision: 0.5
# F1: 0.56
# Auc: 0.74


#Base Model
# Accuracy: 0.78
# Recall: 0.633
# Precision: 0.49
# F1: 0.55
# Auc: 0.73


###########################################
# Feature Importance
###########################################

def plot_feature_importance(importance,names,model_type):
    # Create arrays from feature importance and feature names
    feature_importance=np.array(importance)
    feature_names=np.array(names)

    #Create a Dataframe using Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df=pd.DataFrame(data)

    # Sort the DataFrame in order decreasing deature importances
    fi_df.sort_values(by=['feature_importance'],ascending=False,inplace=True)

    #Define size of bar plot
    plt.figure(figsize=(15,10))

    # Plot Seaborn bar chart
    sns.barplot(x=fi_df['feature_importance'],y=fi_df['feature_names'])

    # Add chart labels
    plt.title(model_type+' FEATURE IMPOTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    plt.show()

plot_feature_importance(catboost_model.get_feature_importance(),X.columns,'CATBOOST')

