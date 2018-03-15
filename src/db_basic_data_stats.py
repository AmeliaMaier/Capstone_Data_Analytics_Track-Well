import psycopg2
import pandas as pd
import os
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoLarsCV
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from pandas.plotting import scatter_matrix
from sklearn.metrics import confusion_matrix
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')


psql_user = os.environ.get('PSQL_USER')
psql_password = os.environ.get('PSQL_PASSWORD')

#tables = ["preset", "preset_array", "protocol", "protocol_array", "reminder", "scale_option", "statistic", "map_protocol_day_entry", "user_table", "entry", "comment"]

def logistic_regression(df):
    logreg = LogisticRegression()
    columns = ['dup_protocol_started','caffeine_yn', 'married_yn', 'usual_activity_len', 'alcohol_yn', 'bio_sex_answered', 'bio_sex', 'menstruation_yn', 'blood_type_answered', 'usual_diet_len', 'usual_medications_len', 'usual_conditions_len']
    #X=df[['dup_protocol_started','caffeine_yn', 'married_yn', 'usual_activity_len', 'alcohol_yn', 'bio_sex_answered', 'bio_sex', 'menstruation_yn', 'blood_type_answered', 'usual_diet_len', 'usual_medications_len', 'usual_conditions_len']]
    #X=df[['dup_protocol_started','caffeine_yn', 'married_yn', 'usual_activity_len', 'bio_sex_answered', 'menstruation_yn', 'blood_type_answered', 'usual_diet_len', 'usual_medications_len']]
    #X=stand(df[['dup_protocol_started','caffeine_yn', 'married_yn', 'usual_activity_len', 'bio_sex_answered', 'menstruation_yn', 'blood_type_answered', 'usual_diet_len', 'usual_medications_len']])
    X=stand(df[['dup_protocol_started','caffeine_yn', 'married_yn', 'usual_activity_len', 'alcohol_yn', 'bio_sex_answered', 'bio_sex', 'menstruation_yn', 'blood_type_answered', 'usual_diet_len', 'usual_medications_len', 'usual_conditions_len']])

    y=df['user_active_yn']
    logit_model=sm.Logit(y,X)
    result=logit_model.fit()
    print(result.summary())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    print("With train test fix")
    print(f'Mean accuracy of logistic regression classifier on test set: {logreg.score(X_test, y_test)}')
    print(confusion_matrix(y_test, y_pred))
    print("tn, fp\nfn, tp")
    print('\n\n')

    coef_list = pd.DataFrame(data = {'features':columns, 'estimatedCoefficients':logreg.coef_.ravel()})
    coef_list['sort'] = coef_list.estimatedCoefficients.abs()
    print(coef_list.sort_values(by='sort', ascending=False))
    print(f'Estimated intercept: {logreg.intercept_}')
    print_roc_curve(y_test, logreg, X_test)

    print("With KFold cross validation")
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    modelCV = LogisticRegression()
    scoring = 'accuracy'
    results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
    print(f"10-fold cross validation average accuracy: {results.mean()}")
    scoring = 'recall'
    results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
    print(f"10-fold cross validation average recall: {results.mean()}")
    scoring = 'precision'
    results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
    print(f"10-fold cross validation average precision: {results.mean()}")
    print(classification_report(y_test, y_pred))

    print_roc_curve(y_test, logreg, X_test)

def print_roc_curve(y_test, logreg, X_test):
    logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()

def linear_regression(df):
    columns = ['dup_protocol_started', 'usual_activity_len', 'usual_medications_len','caffeine_yn','bio_sex', 'smoke_yn', 'pregnant_yn']
    #X = df[['dup_protocol_started','dup_protocol_finished', 'usual_activity_len', 'alcohol_yn', 'menstruation_yn', 'usual_medications_len', 'usual_conditions_len','caffeine_yn','bio_sex', 'smoke_yn', 'pregnant_yn']]
    #X = stand(df[['dup_protocol_started','dup_protocol_finished', 'usual_activity_len', 'alcohol_yn', 'menstruation_yn', 'usual_medications_len', 'usual_conditions_len','caffeine_yn','bio_sex', 'smoke_yn', 'pregnant_yn']])
    X = stand(df[['dup_protocol_started', 'usual_activity_len', 'usual_medications_len', 'caffeine_yn','bio_sex', 'smoke_yn', 'pregnant_yn']])
    y=df['user_activity_score']
    result = sm.OLS( y, X ).fit()
    print(result.summary())
    lm = LinearRegression()
    model = lm.fit(X,y)
    predictions = lm.predict(X)
    print(f'R^2: {lm.score(X,y)}')
    coef_list = pd.DataFrame(data = {'features':columns, 'estimatedCoefficients':lm.coef_})
    coef_list['sort'] = coef_list.estimatedCoefficients.abs()
    print(coef_list.sort_values(by='sort', ascending=False))
    print(f'Estimated intercept: {lm.intercept_}')
    plt.scatter(y, predictions)
    plt.ylabel("predictions")
    plt.xlabel('actuals')
    plt.title('linear regression results\nuser activity score')
    plt.xlim(0,50)
    plt.ylim(0,50)
    plt.show()
    '''   version 1
                            OLS Regression Results
    ===============================================================================
    Dep. Variable:     user_activity_score   R-squared:                       0.535
    Model:                             OLS   Adj. R-squared:                  0.530
    Method:                  Least Squares   F-statistic:                     122.2
    Date:                 Wed, 14 Mar 2018   Prob (F-statistic):          1.56e-185
    Time:                         16:59:53   Log-Likelihood:                -3390.7
    No. Observations:                 1181   AIC:                             6803.
    Df Residuals:                     1170   BIC:                             6859.
    Df Model:                           11
    Covariance Type:             nonrobust
    =========================================================================================
                                coef    std err          t      P>|t|      [0.025      0.975]
    -----------------------------------------------------------------------------------------
    dup_protocol_started      3.9421      0.403      9.790      0.000       3.152       4.732
    dup_protocol_finished    -0.8984      0.316     -2.841      0.005      -1.519      -0.278
    usual_activity_len        0.0142      0.004      3.998      0.000       0.007       0.021
    alcohol_yn                0.2119      0.308      0.689      0.491      -0.392       0.816
    menstruation_yn           1.2501      0.416      3.006      0.003       0.434       2.066
    usual_medications_len     0.0084      0.006      1.457      0.145      -0.003       0.020
    usual_conditions_len     -0.0019      0.004     -0.482      0.630      -0.010       0.006
    caffeine_yn               0.4721      0.313      1.508      0.132      -0.142       1.086
    bio_sex                   0.4168      0.312      1.335      0.182      -0.196       1.030
    smoke_yn                  1.1498      0.526      2.184      0.029       0.117       2.183
    pregnant_yn               1.7779      1.136      1.564      0.118      -0.452       4.008
    ==============================================================================
    Omnibus:                      676.774   Durbin-Watson:                   1.845
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             8886.702
    Skew:                           2.383   Prob(JB):                         0.00
    Kurtosis:                      15.565   Cond. No.                         499.
    ==============================================================================

    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    R^2: 0.07011509931309434

        estimatedCoefficients               features      sort
    0                2.226435   dup_protocol_started  2.226435
    10               1.518028            pregnant_yn  1.518028
    9                1.176404               smoke_yn  1.176404
    1               -0.884182  dup_protocol_finished  0.884182
    4                0.467655        menstruation_yn  0.467655
    7                0.442619            caffeine_yn  0.442619
    8               -0.404983                bio_sex  0.404983
    3                0.254901             alcohol_yn  0.254901
    2                0.014552     usual_activity_len  0.014552
    5                0.007029  usual_medications_len  0.007029
    6               -0.002571   usual_conditions_len  0.002571
    Estimated intercept: 2.411472879553404
    '''
    '''version 2
         OLS Regression Results
    ===============================================================================
    Dep. Variable:     user_activity_score   R-squared:                       0.035
    Model:                             OLS   Adj. R-squared:                  0.025
    Method:                  Least Squares   F-statistic:                     3.806
    Date:                 Wed, 14 Mar 2018   Prob (F-statistic):           2.15e-05
    Time:                         17:06:15   Log-Likelihood:                -3821.6
    No. Observations:                 1181   AIC:                             7665.
    Df Residuals:                     1170   BIC:                             7721.
    Df Model:                           11
    Covariance Type:             nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    x1             0.5627      0.207      2.715      0.007       0.156       0.969
    x2            -0.3930      0.202     -1.941      0.053      -0.790       0.004
    x3             0.5698      0.200      2.846      0.005       0.177       0.963
    x4             0.1240      0.216      0.575      0.566      -0.299       0.548
    x5             0.1726      0.242      0.714      0.475      -0.301       0.647
    x6             0.1684      0.199      0.847      0.397      -0.222       0.558
    x7            -0.0926      0.206     -0.450      0.653      -0.496       0.311
    x8             0.2193      0.223      0.981      0.327      -0.219       0.658
    x9            -0.1876      0.245     -0.766      0.444      -0.668       0.293
    x10            0.3007      0.194      1.552      0.121      -0.080       0.681
    x11            0.1808      0.195      0.926      0.355      -0.202       0.564
    ==============================================================================
    Omnibus:                      709.761   Durbin-Watson:                   0.878
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):            10065.968
    Skew:                           2.515   Prob(JB):                         0.00
    Kurtosis:                      16.389   Cond. No.                         2.62
    ==============================================================================

    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    R^2: 0.07011509931309434
        estimatedCoefficients               features      sort
    2                0.569819     usual_activity_len  0.569819
    0                0.562729   dup_protocol_started  0.562729
    1               -0.392978  dup_protocol_finished  0.392978
    9                0.300709               smoke_yn  0.300709
    7                0.219345            caffeine_yn  0.219345
    8               -0.187567                bio_sex  0.187567
    10               0.180813            pregnant_yn  0.180813
    4                0.172554        menstruation_yn  0.172554
    5                0.168376  usual_medications_len  0.168376
    3                0.124050             alcohol_yn  0.124050
    6               -0.092570   usual_conditions_len  0.092570
    Estimated intercept: 4.4603242419024856

    '''
    '''
                OLS Regression Results
    ===============================================================================
    Dep. Variable:     user_activity_score   R-squared:                       0.030
    Model:                             OLS   Adj. R-squared:                  0.025
    Method:                  Least Squares   F-statistic:                     5.256
    Date:                 Wed, 14 Mar 2018   Prob (F-statistic):           6.35e-06
    Time:                         17:09:35   Log-Likelihood:                -3824.2
    No. Observations:                 1181   AIC:                             7662.
    Df Residuals:                     1174   BIC:                             7698.
    Df Model:                            7
    Covariance Type:             nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    x1             0.3870      0.186      2.083      0.038       0.022       0.752
    x2             0.5524      0.192      2.875      0.004       0.175       0.929
    x3             0.1433      0.190      0.754      0.451      -0.230       0.516
    x4             0.3039      0.192      1.586      0.113      -0.072       0.680
    x5            -0.3171      0.187     -1.700      0.089      -0.683       0.049
    x6             0.3289      0.192      1.713      0.087      -0.048       0.705
    x7             0.1684      0.195      0.864      0.388      -0.214       0.551
    ==============================================================================
    Omnibus:                      717.623   Durbin-Watson:                   0.873
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):            10479.890
    Skew:                           2.542   Prob(JB):                         0.00
    Kurtosis:                      16.679   Cond. No.                         1.63
    ==============================================================================

    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    R^2: 0.06167387899509347
       estimatedCoefficients               features      sort
    1               0.552443     usual_activity_len  0.552443
    0               0.386994   dup_protocol_started  0.386994
    5               0.328866               smoke_yn  0.328866
    4              -0.317130                bio_sex  0.317130
    3               0.303859            caffeine_yn  0.303859
    6               0.168369            pregnant_yn  0.168369
    2               0.143305  usual_medications_len  0.143305
    Estimated intercept: 4.4603242419024856
    '''

def stand(df):
    scaler = StandardScaler()
    scaler.fit(df)
    return scaler.transform(df)

def query_to_dataframe(query_string):
    engine = create_engine(f'postgresql+psycopg2://{psql_user}:{psql_password}@localhost:5432/matt',echo=False)
    table_df=pd.read_sql_query(query_string,con=engine)
    return table_df

def signup_rate_hist_monthly():
    #limit by date because the user table was pulled down a few days after all the others and has some extra users in it (people who signed up between 3/3/18 and the second pull
    user_signup_hist_query = """
    SELECT _id AS user_id, created_date AS signup_datetime
        FROM user_table
        WHERE  created_date <= '03/03/2018';
    """
    user_signup_df = query_to_dataframe(user_signup_hist_query)
    user_signup_df['signup_date'] = user_signup_df['signup_datetime'].dt.date
    #user_signup_df['user_id'] = user_signup_df.user_id.str.replace('x', '.').astype(float)
    user_signup_df['signup_date'] =  pd.to_datetime(user_signup_df['signup_date'])
    print(user_signup_df.info())
    print(user_signup_df.head())
    #user_signup_df.hist()
    user_signup_df[['signup_date','user_id']].groupby([user_signup_df["signup_date"].dt.year, user_signup_df["signup_date"].dt.month]).count().plot(kind="bar")
    plt.show()

def create_main_df():
    table_dataframes = []
    tables = ["preset", "preset_array", "protocol", "protocol_array",  "scale_option",  "map_protocol_day_entry", "user_table", "entry"]
    for table in tables:
        all_rows_per_table_query = f'SELECT * FROM {table};'
        table_dataframes.append(query_to_dataframe(all_rows_per_table_query))

    table_dataframes[0].dropna(axis=1,how='all', inplace=True)
    for column in table_dataframes[0].columns:
        if column == '_id':
            table_dataframes[0].rename(index=str, columns={column: "preset_id"}, inplace=True)
        else:
            table_dataframes[0].rename(index=str, columns={column: f"preset_{column}"}, inplace=True)
    table_dataframes[1].dropna(axis=1,how='all', inplace=True)
    for column in table_dataframes[1].columns:
        if column == '_id':
            table_dataframes[1].rename(index=str, columns={column: "preset_array_id"}, inplace=True)
        else:
            table_dataframes[1].rename(index=str, columns={column: f"preset_array_{column}"}, inplace=True)
    table_dataframes[2].dropna(axis=1,how='all', inplace=True)
    for column in table_dataframes[2].columns:
        if column == '_id':
            table_dataframes[2].rename(index=str, columns={column: "protocol_id"}, inplace=True)
        else:
            table_dataframes[2].rename(index=str, columns={column: f"protocol_{column}"}, inplace=True)
    table_dataframes[3].dropna(axis=1,how='all', inplace=True)
    for column in table_dataframes[3].columns:
        if column == '_id':
            table_dataframes[3].rename(index=str, columns={column: "protocol_array_id"}, inplace=True)
        else:
            table_dataframes[3].rename(index=str, columns={column: f"protocol_array_{column}"}, inplace=True)
    # table_dataframes[4].rename(index=str, columns={"_id": "scale_option_id"}, inplace=True)
    # table_dataframes[4].dropna(axis=1,how='all', inplace=True)
    table_dataframes[5].dropna(axis=1,how='all', inplace=True)
    for column in table_dataframes[5].columns:
        if column == '_id':
            table_dataframes[5].rename(index=str, columns={column: "map_protocol_day_entry_id"}, inplace=True)
        if column == 'entry_id' or column == 'protocol_id' or column == 'protocol_array_id':
            continue
        else:
            table_dataframes[5].rename(index=str, columns={column: f"map_protocol_day_entry_{column}"}, inplace=True)
    table_dataframes[6].dropna(axis=1,how='all', inplace=True)
    for column in table_dataframes[6].columns:
        if column == '_id':
            table_dataframes[6].rename(index=str, columns={column: "user_id"}, inplace=True)
        else:
            table_dataframes[6].rename(index=str, columns={column: f"user_{column}"}, inplace=True)
    table_dataframes[7].dropna(axis=1,how='all', inplace=True)
    for column in table_dataframes[7].columns:
        if column == 'chosen_user':
            table_dataframes[7].rename(index=str, columns={column: "user_id"}, inplace=True)
        elif column == '_id':
            table_dataframes[7].rename(index=str, columns={column: "entry_id"}, inplace=True)
        elif column == 'preset_array':
            table_dataframes[7].rename(index=str, columns={column: "preset_array_id"}, inplace=True)
        else:
            table_dataframes[7].rename(index=str, columns={column: f"entry_{column}"}, inplace=True)

def create_smaller_main_df():
    table_dataframes = []
    tables = ["user_table", "entry"]
    for table in tables:
        all_rows_per_table_query = f'SELECT * FROM {table};'
        table_dataframes.append(query_to_dataframe(all_rows_per_table_query))

    table_dataframes[0].dropna(axis=1,how='all', inplace=True)
    for column in table_dataframes[0].columns:
        if column == '_id':
            table_dataframes[0].rename(index=str, columns={column: "user_id"}, inplace=True)
        else:
            table_dataframes[0].rename(index=str, columns={column: f"user_{column}"}, inplace=True)
    table_dataframes[1].dropna(axis=1,how='all', inplace=True)
    for column in table_dataframes[1].columns:
        if column == 'chosen_user':
            table_dataframes[1].rename(index=str, columns={column: "user_id"}, inplace=True)
        elif column == '_id':
            table_dataframes[1].rename(index=str, columns={column: "entry_id"}, inplace=True)
        elif column == 'preset_array':
            table_dataframes[1].rename(index=str, columns={column: "preset_array_id"}, inplace=True)
        else:
            table_dataframes[1].rename(index=str, columns={column: f"entry_{column}"}, inplace=True)

    main_df = table_dataframes[0].merge(table_dataframes[1],how='left',on = 'user_id')
    #print(main_df.info())
    return main_df

def df_to_csv(df, path):
    df.to_csv(path, ",")
def csv_to_df(path):
    return pd.read_csv(path, ",")

def clean_data_types(main_df):
    #main_df['user_id'] = main_df.user_id.str.replace('x', '.').astype(float)
    main_df["entry_chosen_datetime"] = main_df['entry_chosen_datetime'].dt.date
    main_df["entry_chosen_datetime"] = pd.to_datetime(main_df['entry_chosen_datetime'])
    # main_df['user_bio_sex'].replace(('Male','Female'), (1,0), inplace=True)
    # main_df['entry_bio_sex'].replace(('Male','Female'), (1,0), inplace=True)
    # main_df['user_pregnant_yn'].fillna(-1, inplace=True)
    # main_df['entry_pregnant_yn'].fillna(-1, inplace=True)
    return main_df

def get_counts(main_df):
    counts_df = pd.DataFrame(data=main_df['user_id'].unique(), columns=["user_id"], index=main_df['user_id'].unique())
    for column in main_df.columns:
        if column in ["user_id", 'days_inactive', 'days_active', 'estimated_created_date']:
            continue
        else:
            counts_df[f'{column}_cnt'] = main_df.groupby("user_id")[column].nunique()
    counts_df = counts_df.reset_index().drop("index", axis=1)
    counts_df['data_points'] = counts_df.drop(["user_id"],axis=1).sum(axis=1)
    return counts_df[["user_id","data_points","entry_chosen_datetime_cnt","entry_id_cnt"]]

def get_days_active(main_df):
    active_days = main_df.groupby('user_id').max()['entry_created_date'] - main_df.groupby('user_id').min()['estimated_created_date']
    active_days = active_days.fillna(1)
    active_days = active_days.dt.ceil('1D')
    active_days = active_days.dt.days.astype(int)
    active_days = active_days.reset_index()
    active_days = active_days.rename(columns={'index': 'user_id', 0: 'days_active'})
    #defining 1 as lowest num of active days possible for math reasons
    main_df = main_df.merge(active_days, how='left', on='user_id')
    return main_df

def get_days_inactive(main_df):
    inactive_days = pd.to_datetime('03/03/2018') - np.maximum(main_df.groupby('user_id').max()['entry_created_date'],main_df.groupby('user_id').max()['estimated_created_date'])
    #inactive_days = active_days.fillna(0)
    inactive_days = inactive_days.dt.days.astype(int)
    inactive_days = inactive_days.reset_index()
    inactive_days = inactive_days.rename(columns={'index': 'user_id', 'entry_created_date': 'days_inactive'})
    main_df = main_df.merge(inactive_days, how='left', on='user_id')
    return main_df

def get_estimated_created_date(main_df):
    estimated_created_date_df = np.minimum(main_df.groupby('user_id').min()['user_created_date'],main_df.groupby('user_id').min()['entry_created_date'])
    estimated_created_date_df = estimated_created_date_df.reset_index()
    estimated_created_date_df = estimated_created_date_df.rename(columns={'index': 'user_id', 'user_created_date': 'estimated_created_date'})
    main_df = main_df.merge(estimated_created_date_df, how='left', on='user_id')
    return main_df

def corr_heatmap_with_values(df):
    corr = df.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(17,15))
    cmap = sns.color_palette('coolwarm')
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, square=True, linewidths=.5,
                yticklabels=True, annot=True, fmt='.2f', cbar_kws={'shrink':.5})
    plt.title('Correlation Matrix', fontsize=20)
    plt.xticks(rotation=90, fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()
    plt.show()

def dec_vs_other_months(hypothesis_df):
    #hypothesis_df['estimated_created_date'] = hypothesis_df['estimated_created_date'].astype('datetime64[ns]')
    hypothesis_df_Dec = hypothesis_df.loc[hypothesis_df['estimated_created_date'].dt.month == 12]
    hypothesis_df_not_Dec = hypothesis_df.loc[hypothesis_df['estimated_created_date'].dt.month < 12]
    dec_active = hypothesis_df_not_Dec.loc[hypothesis_df_not_Dec['user_active_yn']==1]
    other_active = hypothesis_df_not_Dec.loc[hypothesis_df_not_Dec['user_active_yn']==1]
    # plt.hist(hypothesis_df_Dec['user_activity_score'], color='cyan', label='Dec Active Scores')
    # plt.hist(hypothesis_df_not_Dec['user_activity_score'],alpha=0.7, color='pink', label='Year Active Scores')
    # plt.ylabel('number of users')
    # plt.xlabel('count per variable')
    # plt.legend()
    # plt.show()
    #
    # plt.hist(hypothesis_df_Dec['user_active_yn'], color='cyan', label='Dec Active YN')
    # plt.hist(hypothesis_df_not_Dec['user_active_yn'],alpha=0.7, color='pink', label='Year Active YN')
    # plt.ylabel('number of users')
    # plt.xlabel('count per variable')
    # plt.legend()
    # plt.show()

    bootstrap_ci(dec_active['user_activity_score'], 'Dec User Activity Score')
    bootstrap_ci(other_active['user_activity_score'], 'Not Dec User Activity Score')
    bootstrap_ci(hypothesis_df_Dec['user_active_yn'], 'Dec User Active YN')
    bootstrap_ci(hypothesis_df_not_Dec['user_active_yn'], 'Not Dec User Active YN')

    #dec sample for H on user_active_score
    print("Running ttest on user_active score for Dec vs all other months")
    print(f"Dec Average: {dec_active['user_activity_score'].mean()}")
    print(f'Other Average: {other_active["user_activity_score"].mean()}')
    hypothesis_df_Dec_sample = dec_active.sample(len(other_active.index))
    user_active_score_ttest = stats.ttest_ind(hypothesis_df_Dec_sample['user_activity_score'], other_active['user_activity_score'])
    print(user_active_score_ttest)
    #Ttest_indResult(statistic=-5.9016620625301828, pvalue=4.0248341180765505e-09)

    #dec sample for H on user activity yn
    print("Running ttest on user active yn for Dec vs all other months")
    print(f"Dec Average: {hypothesis_df_Dec['user_active_yn'].mean()}")
    print(f'Other Average: {hypothesis_df_not_Dec["user_active_yn"].mean()}')
    hypothesis_df_Dec_sample = hypothesis_df_Dec.sample(len(hypothesis_df_not_Dec.index))
    user_active_ttest = stats.ttest_ind(hypothesis_df_Dec_sample['user_active_yn'], hypothesis_df_not_Dec['user_active_yn'])
    print(user_active_ttest)
    #Ttest_indResult(statistic=-12.497492110151295, pvalue=6.2818586376725302e-35)


    '''
    Running ttest on user_active score for Dec vs all other months
    Dec Average: 0.8813206096033261
    Other Average: 2.769237275134839
    Ttest_indResult(statistic=-14.63079024707694, pvalue=8.6358543621434763e-47)

    Running ttest on user active yn for Dec vs all other months
    Dec Average: 0.30687830687830686
    Other Average: 0.5045742434904996
    Ttest_indResult(statistic=-10.82884489661955, pvalue=8.3293441015901507e-27)

    '''
def lasso_fit_and_print(target_name, model, predictor, target, predvar_columns):
    pred_train, pred_test, tar_train, tar_test = train_test_split(predictor, target,test_size=.3)
    model=LassoLarsCV(cv=10, precompute=False).fit(pred_train,tar_train)
    m_log_alphas = -np.log10(model.alphas_)
    ax = plt.gca()
    plt.plot(m_log_alphas, model.coef_path_.T)
    plt.axvline(-np.log10(model.alpha_), color='k',label='alpha CV')
    plt.ylabel('Regression Coefficients')
    plt.legend()
    plt.xlabel('-log(alpha)')
    plt.title(f'Regression Coefficients Progression for Lasso Paths\n{target_name}')
    plt.show()
    # print variable names and regression coefficients
    var_imp = pd.DataFrame(data = {'predictors':list(predvar_columns.values),'coefficients':model.coef_})
    var_imp['sort'] = var_imp.coefficients.abs()
    print(f"{target_name} Coefficients")
    print(var_imp.sort_values(by='sort', ascending=False))

def lasso_attempt(data_clean):
    #creating predictors and targets
    predvar_all = data_clean.drop(['user_active_yn','user_activity_score', 'user_activity_cnt', 'days_active', 'days_inactive'], axis=1)
    target1 = data_clean['user_active_yn']
    active_users = data_clean.loc[data_clean['user_active_yn']==1]
    target2 = active_users['user_activity_score']
    target3 = active_users['user_activity_cnt']
    target4 = active_users['days_active']
    target5 = active_users['days_inactive']
    predvar_active = active_users.drop(['user_active_yn','user_activity_score', 'user_activity_cnt', 'days_active', 'days_inactive'], axis=1)
    scaler = StandardScaler()
    scaler.fit(predvar_all)
    predictors_all=scaler.transform(predvar_all)
    scaler.fit(predvar_active)
    predictors_active = scaler.transform(predvar_active)
    lasso_fit_and_print("User Active YN", predictors_all, target1, predvar_all.columns)
    lasso_fit_and_print("User Activity Score", predictors_active, target2, predvar_active.columns)
    lasso_fit_and_print("User Activity Count", predictors_active, target3, predvar_active.columns)
    lasso_fit_and_print("Days Active", predictors_active, target4, predvar_active.columns)
    lasso_fit_and_print("Days Inactive", predictors_active, target5, predvar_active.columns)

def correlation_maps_lasso(user_profile):
    corr_heatmap_with_values(user_profile[['user_active_yn','dup_protocol_started','caffeine_yn','dup_protocol_finished', 'married_yn', 'usual_activity_len', 'alcohol_yn', 'bio_sex_answered', 'bio_sex', 'menstruation_yn', 'blood_type_answered', 'usual_diet_len', 'usual_medications_len', 'usual_conditions_len']])

    corr_heatmap_with_values(user_profile[['user_activity_score', 'dup_protocol_started','dup_protocol_finished', 'usual_activity_len', 'alcohol_yn', 'menstruation_yn', 'usual_medications_len', 'usual_conditions_len','caffeine_yn','bio_sex', 'smoke_yn', 'pregnant_yn']])

    corr_heatmap_with_values(user_profile[['user_activity_cnt', 'usual_activity_len','pregnant_answered', 'caffeine_yn', 'usual_diet_len', 'smoke_answered', 'pregnant_yn', 'dup_protocol_started','usual_medications_len','usual_conditions_len', 'alcohol_yn', 'bio_sex']])

    corr_heatmap_with_values(user_profile[['days_active', 'pregnant_answered','caffeine_yn', 'alcohol_answered', 'married_answered', 'menstruation_answered', 'usual_diet_len', 'caffeine_answered','usual_activity_len','blood_type_answered', 'married_yn', 'dup_protocol_started',]])

    corr_heatmap_with_values(user_profile[['days_inactive', 'height_likelihood','blood_type_answered', 'caffeine_yn', 'dup_protocol_started', 'pregnant_yn', 'smoke_yn', 'dup_protocol_active','smoke_answered','menstruation_yn', 'dup_protocol_finished', 'bio_sex_answered']])

def add_months(user_profile_df):
    user_profile_df['month_created'] = user_profile_df['created_date'].dt.month.astype(int)
    user_profile_df = pd.get_dummies(user_profile_df, columns=['month_created'])
    for num in range(1,13):
        if f'month_created_{num}' in user_profile_df.columns:
            continue
        else:
            user_profile_df[f'month_created_{num}'] = [0]*len(user_profile_df.index)
    return user_profile_df

def bootstrap(sample_array, resample=10000):
    '''Implement a bootstrap function to randomly draw with replacement from a given sample. The function should take a sample as a numpy ndarray and the number of resamples as an integer (default: 10000). The function should return a list of numpy ndarray objects, each ndarray is one bootstrap sample.'''
    samples = []
    sample_array=sample_array.ravel()
    for num in range(resample):
        samples.append(np.random.choice(sample_array, len(sample_array)))
    return samples

def bootstrap_ci(sample,name, stat_function=np.mean, iterations=1000, ci=95):
    '''Implement a bootstrap_ci function to calculate the confidence interval of any sample statistic (in this case the mean). The function should take a sample, a function that computes the sample statistic, the number of resamples (default: 10000), and the confidence interval (default :95%). The function should return the lower and upper bounds of the confidence interval and the bootstrap distribution of the test statistic.'''
    sample_lst = bootstrap(sample, iterations)
    results = []
    for sample_set in sample_lst:
        results.append(stat_function(sample_set))
    results.sort()
    results = np.array(results)

    '''this does the same as the np.percentile under it'''
    #results_cut = results[int((len(results)*((1-(ci/100))/2))):int(len(results)-(len(results)*((1-(ci/100))/2)))]
    plot_histogram_with_normal(results, name)
    return (np.percentile(results, q=[(100-ci)/2,100-(100-ci)/2]), np.mean(results))

def plot_histogram_with_normal(data, name):
    '''simplest form of a histogram'''
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    x_range = np.linspace(stats.norm.ppf(0.0001, data.mean(), data.std()),stats.norm.ppf(0.9999, data.mean(), data.std()), 100)
    ax.hist(data, normed=True)
    normal = stats.norm.pdf(x_range, data.mean(), data.std())
    normal_line =ax.plot(x_range, normal, label='normal pmf', color='r')
    ax.set_title(f"\nmean:{data.mean()} std:{data.std()}\n{stats.kstest(data, 'norm',args=[data.mean(), data.std()])}")
    ax.set_xlabel(name)
    plt.show()

def create_user_profile():
    main_df = clean_data_types(create_smaller_main_df())
    main_df= get_estimated_created_date(main_df)
    main_df= get_days_active(main_df)
    main_df= get_days_inactive(main_df)
    counts_df = get_counts(main_df)
    missing_counts = main_df[['user_id', 'estimated_created_date', 'days_active', 'days_inactive']]
    missing_counts = missing_counts.groupby(['user_id'])['estimated_created_date', 'days_active', 'days_inactive'].min()
    missing_counts = missing_counts.reset_index()
    counts_df = counts_df.merge(missing_counts, how='left', on='user_id')

    counts_df['user_activity_cnt'] = counts_df['entry_chosen_datetime_cnt']+counts_df['entry_id_cnt']
    counts_df['user_activity_score'] = counts_df['user_activity_cnt'].astype(float)/counts_df['days_active'].astype(float)

    user_profile_df = query_to_dataframe('SELECT * FROM user_table;')
    user_profile_df.rename(index=str, columns={'_id': "user_id"}, inplace=True)

    user_profile_df = user_profile_df.merge(counts_df[['user_id','user_activity_score', 'user_activity_cnt', 'days_active', 'days_inactive', 'estimated_created_date']],how='left',on = 'user_id')
    user_profile_df['user_active_yn'] =  np.where(user_profile_df['user_activity_score']==0,0,1)
    user_profile_df.dropna(axis=1,how='all', inplace=True)
    user_profile_df['bio_sex'].replace(('Male','Female'), (1,0), inplace=True)
    user_profile_df['dup_protocol_started'] = user_profile_df['dup_protocol_started'].fillna(0)
    user_profile_df['dup_protocol_started'] =  np.where(user_profile_df['dup_protocol_started']==0,0,1)
    user_profile_df['dup_protocol_active'] = user_profile_df['dup_protocol_active'].fillna(0)
    user_profile_df['dup_protocol_active'] =  np.where(user_profile_df['dup_protocol_active']==0,0,1)
    user_profile_df['dup_protocol_finished'] = user_profile_df['dup_protocol_finished'].fillna(0)
    user_profile_df['dup_protocol_finished'] =  np.where(user_profile_df['dup_protocol_finished']==0,0,1)
    user_profile_df.drop('modified_date', axis=1, inplace=True)
    user_profile_df['usual_conditions_len'] = user_profile_df['usual_conditions'].str.len()
    user_profile_df['usual_conditions_len'] =user_profile_df['usual_conditions_len'].fillna(0)
    user_profile_df.drop('usual_conditions', axis=1, inplace=True)
    user_profile_df.drop('anon_code', axis=1, inplace=True)
    user_profile_df['usual_medications_len'] = user_profile_df['usual_medications'].str.len()
    user_profile_df['usual_medications_len'] =user_profile_df['usual_medications_len'].fillna(0)
    user_profile_df.drop('usual_medications', axis=1, inplace=True)
    user_profile_df['usual_diet_len'] = user_profile_df['usual_diet'].str.len()
    user_profile_df['usual_diet_len'] =user_profile_df['usual_diet_len'].fillna(0)
    user_profile_df.drop('usual_diet', axis=1, inplace=True)
    user_profile_df['usual_activity_len'] = user_profile_df['usual_activity'].str.len()
    user_profile_df['usual_activity_len'] =user_profile_df['usual_activity_len'].fillna(0)
    user_profile_df.drop('usual_activity', axis=1, inplace=True)
    user_profile_df.drop('protocol_array_list', axis=1, inplace=True)
    user_profile_df['smoke_answered'] = user_profile_df['smoke_yn']
    user_profile_df['smoke_answered'] = user_profile_df['smoke_answered'].fillna(-1)
    user_profile_df['smoke_answered'] = np.where(user_profile_df['smoke_answered']==-1, 0, 1)
    user_profile_df['smoke_yn'] = user_profile_df['smoke_yn'].fillna(0)

    user_profile_df['alcohol_answered'] = user_profile_df['alcohol_yn']
    user_profile_df['alcohol_answered'] = user_profile_df['alcohol_answered'].fillna(-1)
    user_profile_df['alcohol_answered'] = np.where(user_profile_df['alcohol_answered']==-1, 0, 1)
    user_profile_df['alcohol_yn'] = user_profile_df['alcohol_yn'].fillna(0)

    user_profile_df['married_answered'] = user_profile_df['married_yn']
    user_profile_df['married_answered'] = user_profile_df['married_answered'].fillna(-1)
    user_profile_df['married_answered'] = np.where(user_profile_df['married_answered']==-1, 0, 1)
    user_profile_df['married_yn'] = user_profile_df['married_yn'].fillna(0)
    user_profile_df['caffeine_answered'] = user_profile_df['caffeine_yn']
    user_profile_df['caffeine_answered'] = user_profile_df['caffeine_answered'].fillna(-1)
    user_profile_df['caffeine_answered'] = np.where(user_profile_df['caffeine_answered']==-1, 0, 1)
    user_profile_df['caffeine_yn'] = user_profile_df['caffeine_yn'].fillna(0)
    user_profile_df['pregnant_answered'] = user_profile_df['pregnant_yn']
    user_profile_df['pregnant_answered'] = user_profile_df['pregnant_answered'].fillna(-1)
    user_profile_df['pregnant_answered'] = np.where(user_profile_df['pregnant_answered']==-1, 0, 1)
    user_profile_df['pregnant_yn'] = user_profile_df['pregnant_yn'].fillna(0)
    user_profile_df['blood_type_answered'] = user_profile_df['blood_type']
    user_profile_df['blood_type_answered'] = user_profile_df['blood_type_answered'].fillna(-1)
    user_profile_df['blood_type_answered'] = np.where(user_profile_df['blood_type_answered']==-1, 0, 1)
    user_profile_df.drop('blood_type', axis=1, inplace=True)

    user_profile_df['bio_sex_answered'] = user_profile_df['bio_sex']
    user_profile_df['bio_sex_answered'] = user_profile_df['bio_sex_answered'].fillna(-1)
    user_profile_df['bio_sex_answered'] = np.where(user_profile_df['bio_sex_answered']==-1, 0, 1)
    user_profile_df['bio_sex'] = user_profile_df['bio_sex'].fillna(0)

    user_profile_df['menstruation_answered'] = user_profile_df['menstruation_yn']
    user_profile_df['menstruation_answered'] = user_profile_df['menstruation_answered'].fillna(-1)
    user_profile_df['menstruation_answered'] = np.where(user_profile_df['menstruation_answered']==-1, 0, 1)
    user_profile_df['menstruation_yn'] = user_profile_df['menstruation_yn'].fillna(0)

    '''user_profile_df=add_months(user_profile_df)'''
    user_profile_df.drop('created_date', axis=1, inplace=True)

    #so as to not have to pull sensus data, useing http://www.usablestats.com/lessons/normal for average heights and distributions
    #adult male heights are on average 70 inches  (5'10) with a standard deviation of 4 inches. Adult women are on average a bit shorter and less variable in height with a mean height of 65  inches (5'5) and standard deviation of 3.5 inches
    # male_average_height = 177.8
    # male_sd_height = 10.16
    # female_average_height = 165.1
    # female_sd_height = 8.89
    #male_norm = stats.norm(male_average_height,male_sd_height)
    #female_norm = stats.norm(female_average_height,female_sd_height)
    adult_avg_height = 177+165/2
    adult_sd_height = 20
    adult_norm = stats.norm(adult_avg_height, adult_sd_height)
    user_profile_df['height_cm'] = user_profile_df['height_cm'].fillna(0)
    user_profile_df['height_likelihood'] = user_profile_df['height_cm'].apply(lambda x: adult_norm.pdf(x))
    user_profile_df.drop('height_cm', axis=1, inplace=True)
    return user_profile_df


#user_profile_df = create_user_profile()
'''
hypothesis_df = user_profile_df[['user_id','user_active_yn','user_activity_score', 'user_activity_cnt', 'days_active', 'days_inactive', 'estimated_created_date']]
dec_vs_other_months(hypothesis_df)
'''
'''
lasso_attempt(user_profile_df.drop(['user_id','estimated_created_date'],axis=1))
'''
#correlation_maps_lasso(user_profile_df)
#df_to_csv(user_profile_df, 'user_profile.csv')
user_profile_df = csv_to_df('user_profile.csv')
user_profile_df['estimated_created_date'] = user_profile_df['estimated_created_date'].astype('datetime64[ns]')

logistic_regression(user_profile_df)
#linear_regression(user_profile_df.loc[user_profile_df['user_active_yn']==1])

#dec_vs_other_months(user_profile_df[['user_id','user_active_yn','user_activity_score', 'user_activity_cnt', 'days_active', 'days_inactive', 'estimated_created_date']])


# plt.scatter(counts_df['entry_chosen_datetime_cnt'], counts_df['entry_id_cnt'])
# plt.xlabel("entry_chosen_datetime_cnt")
# plt.ylabel("entry_id_cnt")
# plt.show()
# plt.scatter(user_profile_df['user_activity_score'], user_profile_df['days_inactive'].replace((0), (1)))
# plt.xlabel("user active score")
# plt.ylabel("inactive days")
# plt.show()

# counts_df['days_active'].replace((0), (1),inplace=True)
# counts_without_zero = counts_df.loc[counts_df['entry_chosen_datetime_cnt']>0]
# counts_without_zero = counts_without_zero.loc[counts_without_zero['entry_id_cnt']>0]
# plt.scatter(np.log(counts_without_zero['entry_chosen_datetime_cnt']/counts_without_zero['days_active']), np.log(counts_without_zero['entry_id_cnt']/counts_without_zero['days_active']))
# plt.xlabel("log unique days with entry by total days active")
# plt.ylabel("log average data points per total day active")
# plt.title('Without Zeros')
# plt.show()
