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



#
# def create_main_df():
#     table_dataframes = []
#     tables = ["preset", "preset_array", "protocol", "protocol_array",  "scale_option",  "map_protocol_day_entry", "user_table", "entry"]
#     for table in tables:
#         all_rows_per_table_query = f'SELECT * FROM {table};'
#         table_dataframes.append(query_to_dataframe(all_rows_per_table_query))
#
#     table_dataframes[0].dropna(axis=1,how='all', inplace=True)
#     for column in table_dataframes[0].columns:
#         if column == '_id':
#             table_dataframes[0].rename(index=str, columns={column: "preset_id"}, inplace=True)
#         else:
#             table_dataframes[0].rename(index=str, columns={column: f"preset_{column}"}, inplace=True)
#     table_dataframes[1].dropna(axis=1,how='all', inplace=True)
#     for column in table_dataframes[1].columns:
#         if column == '_id':
#             table_dataframes[1].rename(index=str, columns={column: "preset_array_id"}, inplace=True)
#         else:
#             table_dataframes[1].rename(index=str, columns={column: f"preset_array_{column}"}, inplace=True)
#     table_dataframes[2].dropna(axis=1,how='all', inplace=True)
#     for column in table_dataframes[2].columns:
#         if column == '_id':
#             table_dataframes[2].rename(index=str, columns={column: "protocol_id"}, inplace=True)
#         else:
#             table_dataframes[2].rename(index=str, columns={column: f"protocol_{column}"}, inplace=True)
#     table_dataframes[3].dropna(axis=1,how='all', inplace=True)
#     for column in table_dataframes[3].columns:
#         if column == '_id':
#             table_dataframes[3].rename(index=str, columns={column: "protocol_array_id"}, inplace=True)
#         else:
#             table_dataframes[3].rename(index=str, columns={column: f"protocol_array_{column}"}, inplace=True)
#     # table_dataframes[4].rename(index=str, columns={"_id": "scale_option_id"}, inplace=True)
#     # table_dataframes[4].dropna(axis=1,how='all', inplace=True)
#     table_dataframes[5].dropna(axis=1,how='all', inplace=True)
#     for column in table_dataframes[5].columns:
#         if column == '_id':
#             table_dataframes[5].rename(index=str, columns={column: "map_protocol_day_entry_id"}, inplace=True)
#         if column == 'entry_id' or column == 'protocol_id' or column == 'protocol_array_id':
#             continue
#         else:
#             table_dataframes[5].rename(index=str, columns={column: f"map_protocol_day_entry_{column}"}, inplace=True)
#     table_dataframes[6].dropna(axis=1,how='all', inplace=True)
#     for column in table_dataframes[6].columns:
#         if column == '_id':
#             table_dataframes[6].rename(index=str, columns={column: "user_id"}, inplace=True)
#         else:
#             table_dataframes[6].rename(index=str, columns={column: f"user_{column}"}, inplace=True)
#     table_dataframes[7].dropna(axis=1,how='all', inplace=True)
#     for column in table_dataframes[7].columns:
#         if column == 'chosen_user':
#             table_dataframes[7].rename(index=str, columns={column: "user_id"}, inplace=True)
#         elif column == '_id':
#             table_dataframes[7].rename(index=str, columns={column: "entry_id"}, inplace=True)
#         elif column == 'preset_array':
#             table_dataframes[7].rename(index=str, columns={column: "preset_array_id"}, inplace=True)
#         else:
#             table_dataframes[7].rename(index=str, columns={column: f"entry_{column}"}, inplace=True)
#


def csv_to_df(path):
    return pd.read_csv(path, ",")




def corr_heatmap_with_values(df, title):
    corr = df.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(10,10))
    cmap = sns.color_palette('coolwarm')
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, square=True, linewidths=.5,
                yticklabels=True, annot=True, fmt='.2f', cbar_kws={'shrink':.5})
    plt.title(title, fontsize=20)
    plt.xticks(rotation=60, fontsize=11, horizontalalignment='right')
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()
    plt.show()



def correlation_maps_lasso(user_profile):
    corr_heatmap_with_values(user_profile[['user_active_yn','dup_protocol_started','caffeine_yn','dup_protocol_finished', 'married_yn', 'usual_activity_len', 'alcohol_yn', 'bio_sex_answered', 'bio_sex', 'menstruation_yn', 'blood_type_answered', 'usual_diet_len', 'usual_medications_len', 'usual_conditions_len']], "User Active YN\nLasso Based Correlation Heat Map")

    corr_heatmap_with_values(user_profile[['user_activity_score', 'dup_protocol_started','dup_protocol_finished', 'usual_activity_len', 'alcohol_yn', 'menstruation_yn', 'usual_medications_len', 'usual_conditions_len','caffeine_yn','bio_sex', 'smoke_yn', 'pregnant_yn']], 'User Activity Score\nLasso Based Correlation Heat Map')

    corr_heatmap_with_values(user_profile[['user_activity_cnt', 'usual_activity_len','pregnant_answered', 'caffeine_yn', 'usual_diet_len', 'smoke_answered', 'pregnant_yn', 'dup_protocol_started','usual_medications_len','usual_conditions_len', 'alcohol_yn', 'bio_sex']], 'User Activity Count\nLasso Based Correlation Heat Map')

    corr_heatmap_with_values(user_profile[['days_active', 'pregnant_answered','caffeine_yn', 'alcohol_answered', 'married_answered', 'menstruation_answered', 'usual_diet_len', 'caffeine_answered','usual_activity_len','blood_type_answered', 'married_yn', 'dup_protocol_started',]], 'Days Active\nLasso Based Correlation Heat Map')

    corr_heatmap_with_values(user_profile[['days_inactive', 'height_likelihood','blood_type_answered', 'caffeine_yn', 'dup_protocol_started', 'pregnant_yn', 'smoke_yn', 'dup_protocol_active','smoke_answered','menstruation_yn', 'dup_protocol_finished', 'bio_sex_answered']], 'Days Inactive\nLasso Based Correlation Heat Map')
#
# def add_months(user_profile_df):
#     user_profile_df['month_created'] = user_profile_df['created_date'].dt.month.astype(int)
#     user_profile_df = pd.get_dummies(user_profile_df, columns=['month_created'])
#     for num in range(1,13):
#         if f'month_created_{num}' in user_profile_df.columns:
#             continue
#         else:
#             user_profile_df[f'month_created_{num}'] = [0]*len(user_profile_df.index)
#     return user_profile_df




# print(standard_confusion_matrix([1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0, 0]))




#user_profile_df = create_user_profile()
'''
hypothesis_df = user_profile_df[['user_id','user_active_yn','user_activity_score', 'user_activity_cnt', 'days_active', 'days_inactive', 'estimated_created_date']]
dec_vs_other_months(hypothesis_df)
'''
'''
lasso_attempt(user_profile_df.drop(['user_id','estimated_created_date'],axis=1))
'''
#df_to_csv(user_profile_df, 'user_profile.csv')
user_profile_df = csv_to_df('user_profile.csv')
user_profile_df['estimated_created_date'] = user_profile_df['estimated_created_date'].astype('datetime64[ns]')

#corr_heatmap_with_values(user_profile_df, "User Profile Features\nCorrelation Heat Map")
#correlation_maps_lasso(user_profile_df)

# month_year_active = user_profile_df[['user_id', 'estimated_created_date', 'days_active', 'days_inactive', 'user_activity_score']]
# month_year_active['Profile Creation Month'] = month_year_active['estimated_created_date'].dt.month.astype(int) + (month_year_active['estimated_created_date'].dt.year.astype(int) - 2017)*12
# plt.hist(month_year_active['Profile Creation Month'])
# plt.ylabel('Total Number of Users')
# plt.xlabel('Profile Creation Month')
# plt.title('User Count Per Signup Month')
# plt.show()

#dec_vs_other_months(user_profile_df[['user_id','user_active_yn','user_activity_score', 'user_activity_cnt', 'days_active', 'days_inactive', 'estimated_created_date']])

#lasso_attempt(user_profile_df.drop(['user_id','estimated_created_date'],axis=1))


# plt.hist(user_profile_df['user_active_yn'] )
# plt.xlabel("Users Active YN")
# plt.show()


logistic_regression(user_profile_df)
logistic_balanced(user_profile_df.loc[user_profile_df['estimated_created_date'].dt.month < 12])
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
