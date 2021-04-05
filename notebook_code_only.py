import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from feature_engine import imputation as mdi
import altair as alt
import os


# sklearns pipeline
from sklearn.pipeline import Pipeline

# for feature engineering
from sklearn.preprocessing import StandardScaler
from feature_engine import imputation as mdi
from feature_engine import discretisation as dsc
from feature_engine import encoding as ce
from feature_engine.discretisation import EqualFrequencyDiscretiser
from feature_engine import selection as sel

# for glm model
import h2o

# performance measure
from sklearn.metrics import roc_auc_score


os.getcwd()

data = pd.read_csv('data/marketing_campaign.csv', sep=';')
data.head()


pd.set_option('display.max_columns', None)

X_train, X_test, y_train, y_test = train_test_split(data.drop(
    ['ID', 'Response'], axis=1),
    data['Response'],
    stratify = data['Response'], 
    test_size=0.2,
    random_state=0)



# check cardinality and remove vars with single value
X_train.nunique()

# I should have set this in the pipeline


X_train.drop(columns = ['Z_CostContact', 'Z_Revenue'], inplace=True)
X_test.drop(columns = ['Z_CostContact', 'Z_Revenue'], inplace=True)

# cardinality of all variables
# dep var
# other vars
# missing data
# make lists of variable types
year_vars = ['Year_Birth']
dt_vars = ['Dt_Customer']
# tenure with business in months

X_train.isna().mean()
# so none of the data is missing but small % for incode
X_train[X_train['Income'].isna()==True]
# so only need to impute income

# now I want to model as survival model using recency
recency_info = X_train['Recency'].value_counts().reset_index().rename(columns={'Recency':'Cnt', 'index':'Recency'})
alt.Chart(recency_info.tail(10)).mark_bar().encode(
    y='Recency:O',
    x='Cnt:Q'
)


# numeric variables we wish to treat as discrete -> discrete means a numeric var with countable nnumber of events e.g. poisson/binomial etc
discrete = [
    var for var in X_train.columns if X_train[var].dtype != 'O'
        and len(X_train[var].unique()) < 20 
        and var not in year_vars + dt_vars
        and len(X_train[var].unique())>2
        and var != 'Recency'

]

categorical = [
    var for var in X_train.columns if X_train[var].dtype == 'O' 
    and var not in year_vars + dt_vars
    and len(X_train[var].unique())>2
]


numerical = [
    var for var in X_train.columns if X_train[var].dtype != 'O'
    if      var not in discrete 
        and var not in ['ID', 'SalePrice', 'Recency']
        and var not in year_vars + dt_vars
        and len(X_train[var].unique())>2

    
]


ohe_vars = [var for var in  X_train.columns if len(X_train[var].unique()) == 2]



print('There are {} continuous variables'.format(len(numerical)))
print('There are {} discrete variables'.format(len(discrete)))
print('There are {} categorical variables'.format(len(categorical)))
print('There are {} ohe_vars variables'.format(len(ohe_vars)))



min_dt_train = pd.to_datetime(X_train['Dt_Customer']).min()
max_dt_train = pd.to_datetime(X_train['Dt_Customer']).max()


def add_age__add_tenure(df):
    # capture difference between year variable and
    # year the house was sold
    # want to then put age into brackets as we can see outliers
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])
    df['cust_age'] = max_dt_train.year - df['Year_Birth']
    df['cust_tenure'] = (df['Dt_Customer'].dt.year - min_dt_train.year) * 12
    df['cust_tenure'] = df['cust_tenure'] + (df['Dt_Customer'].dt.month-min_dt_train.month)

    return df

X_train = add_age__add_tenure(X_train)
X_test = add_age__add_tenure(X_test)

# drop YrSold
X_train.drop(year_vars + dt_vars, axis=1, inplace=True)
X_test.drop(year_vars + dt_vars, axis=1, inplace=True)


# discretise vars
discretise_vars = ['cust_age', 'cust_tenure']




X_train[discrete] = X_train[discrete].astype('O')
X_test[discrete] = X_test[discrete].astype('O')


# get missing column info
X_train[X_train.columns[(X_train.isnull().mean()>0)==True]].isnull().mean()


X_train[categorical].nunique()
X_train[discrete].nunique()


data_prep_pipe = Pipeline([

    # missing data imputation - section 4
    ('missing_ind',
     mdi.AddMissingIndicator(
         variables=['Income'])),

    ('imputer_num',
     mdi.MeanMedianImputer(
         imputation_method='mean',
         variables=['Income'])),

    # rare label encoding
    ('rare_label_enc',
     ce.RareLabelEncoder(tol=0.01, 
                         n_categories=2, 
                         variables = categorical + discrete)),
    
    

    # discretise age and tenure
    ('disc_age_tenure',
    dsc.EqualFrequencyDiscretiser(q=10, 
                                  variables=discretise_vars,
                                  return_object=True)),
    # simple ordinal encoder
    ('ordinal_enc',
     ce.OrdinalEncoder(encoding_method='ordered',
                       variables=categorical + discrete + discretise_vars))

])


data_prep_pipe.fit(X_train, y_train)

X_train_2 = data_prep_pipe.transform(X_train)
X_test_2 = data_prep_pipe.transform(X_train)


X_mod, X_val, y_mod, y_val = train_test_split(X_train_2,
    y_train,
    stratify=y_train,
    test_size=0.3,
    random_state=0)

y_mod.value_counts()/len(y_mod)
y_val.value_counts()/len(y_val)


# now split train_2 into mod and val -> this will be for hyperparameter
# tuning of glm and random forest


def surv_analysis_df(df_x, df_y):
    df = df_x.copy()
    df['CUST_ID'] = range(len(df))
    df['TARGET_TEMP'] = df_y
    df=df.reset_index(drop=True)
    df['time'] = df['Recency'].apply(lambda x: np.arange(x+1))
    df = df.apply(pd.Series.explode).reset_index(drop=True)
    # now set the y to zero and set to 1 in last record case for the cust case
    df['TARGET'] = np.where(df['time'] == df['Recency'], df['TARGET_TEMP'], 0)
    df.drop(columns='TARGET_TEMP', inplace=True)

    X, y = df.drop(columns=['TARGET']), df['TARGET']
    return X, y


X_mod_surv, y_mod_surv = surv_analysis_df(X_mod, y_mod)
X_val_surv, y_val_surv = surv_analysis_df(X_val, y_val)


# init h2o
h2o.init()

# now get glm estimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator

# finally save above as parquet which you'll read in to h2o
X_mod_surv['TARGET'] = y_mod_surv
X_val_surv['TARGET'] = y_val_surv

X_mod_surv.to_csv('data/mod.csv')
X_val_surv.to_csv('data/val.csv')


h2o_mod = h2o.import_file(path = 'data/mod.csv', destination_frame = 'h2o_mod')
h2o_val = h2o.import_file(path = 'data/val.csv', destination_frame = 'h2o_val')

# convert dummy var as factor
h2o_mod['TARGET'] = h2o_mod['TARGET'].asfactor()
h2o_val['TARGET'] = h2o_val['TARGET'].asfactor()


exc_vars = ['TARGET', 'CUST_ID', 'Recency']
x_vars = set(X_mod_surv.columns).difference(set(exc_vars))
x_vars = list(x_vars)
x_vars.sort()
x_vars



glm_model = H2OGeneralizedLinearEstimator(
                                   family = 'Binomial',
                                   model_id = 'glm_model',
                                   alpha = 1, # lasso regression, set alpha = 0 for ridge
                                   #lambda_ = 0,
                                   lambda_search = True,
                                   standardize = True,
                                   intercept = True)

glm_model.train(x = x_vars, 
                y = 'TARGET',
                training_frame = h2o_mod,
                validation_frame = h2o_val)

# excellent now let's get the performance of the model
# but also I can submit my prediction as well

def get_pred(h2o_model, h2o_df):
    df = h2o_model.predict(h2o_df)['p1'].as_data_frame()
    vars_to_keep = ['TARGET', 'CUST_ID', 'Recency', 'time']
    df[vars_to_keep] = h2o_df[vars_to_keep].as_data_frame()
    return df

h2o_mod_pef = get_pred(glm_model, h2o_mod)    
h2o_val_pef = get_pred(glm_model, h2o_val)    

roc_auc_score(h2o_mod_pef['TARGET'], h2o_mod_pef['p1'])
mask = h2o_mod_pef['time'] == h2o_mod_pef['Recency']
roc_auc_score(h2o_mod_pef[mask]['TARGET'], h2o_mod_pef[mask]['p1'])


roc_auc_score(h2o_val_pef['TARGET'], h2o_val_pef['p1'])
mask = h2o_val_pef['time'] == h2o_val_pef['Recency']
roc_auc_score(h2o_val_pef[mask]['TARGET'], h2o_val_pef[mask]['p1'])

# so this is the model that we've trained let's now submit a test case
# there is no test case to submit!

# now I want to use shitty sklearn to do this? or not bothering!
# also need to create a version of sklearn that does this with using a simpler
# mod and val view of the data rather than k-fold validation! 


