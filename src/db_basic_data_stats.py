import psycopg2
import pandas as pd
import os
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LassoLarsCV
from sklearn.preprocessing import StandardScaler
from pandas.plotting import scatter_matrix


psql_user = os.environ.get('PSQL_USER')
psql_password = os.environ.get('PSQL_PASSWORD')

#tables = ["preset", "preset_array", "protocol", "protocol_array", "reminder", "scale_option", "statistic", "map_protocol_day_entry", "user_table", "entry", "comment"]


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
    df.to_csv(path, "|")
def csv_to_df(path):
    return pd.read_csv(path, "|")

def clean_data_types(main_df):
#    main_df['user_id'] = main_df.user_id.str.replace('x', '.').astype(float)
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
        if column == 'user_id':
            continue
        counts_df[f'{column}_cnt'] = main_df.groupby("user_id")[column].nunique()
    counts_df = counts_df.reset_index().drop("index", axis=1)
    counts_df['data_points'] = counts_df.drop('user_id',axis=1).sum(axis=1)
    return counts_df[["user_id","data_points","entry_chosen_datetime_cnt","entry_id_cnt"]]

def get_days_active(main_df):
    active_days = main_df.groupby('user_id').max()['entry_created_date'] - np.minimum(main_df.groupby('user_id').min()['user_created_date'],main_df.groupby('user_id').min()['user_created_date'])
    active_days = active_days.fillna(1)
    active_days = active_days.dt.ceil('1D')
    active_days = active_days.dt.days.astype(int)
    active_days = active_days.reset_index()
    active_days = active_days.rename(columns={'index': 'user_id', 0: 'days_active'})
    #defining 1 as lowest num of active days possible for math reasons
    return active_days

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

def dec_vs_other_months(df):

    hypothesis_df_Dec = hypothesis_df.loc[hypothesis_df['created_date'].dt.month == 12]
    hypothesis_df_not_Dec = hypothesis_df.loc[hypothesis_df['created_date'].dt.month < 12]
    hypothesis_df_Dec_sample = hypothesis_df_Dec.sample(1421)
    user_active_score_ttest = stats.ttest_ind(hypothesis_df_Dec_sample['user_activity_score'], hypothesis_df_not_Dec['user_activity_score'])
    print(user_active_score_ttest)
    #Ttest_indResult(statistic=-5.9016620625301828, pvalue=4.0248341180765505e-09)
    user_active_ttest = stats.ttest_ind(hypothesis_df_Dec_sample['user_active_yn'], hypothesis_df_not_Dec['user_active_yn'])
    print(user_active_ttest)
    #Ttest_indResult(statistic=-12.497492110151295, pvalue=6.2818586376725302e-35)

def lasso_attempt(data_clean):
    predvar = data_clean.drop(['user_active_yn','user_activity_score'], axis=1)
    target1 = data_clean['user_active_yn']
    predvar2 = data_clean.loc[data_clean['user_active_yn']==1]
    target2 = predvar2['user_activity_score']
    predvar2 = predvar2.drop(['user_active_yn','user_activity_score'], axis=1)
    scaler = StandardScaler()
    scaler.fit(predvar)
    predictors=scaler.transform(predvar)
    scaler.fit(predvar2)
    predictors2 = scaler.transform(predvar2)
    #target1
    pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, target1,test_size=.3)
    model=LassoLarsCV(cv=10, precompute=False).fit(pred_train,tar_train)
    m_log_alphas = -np.log10(model.alphas_)
    ax = plt.gca()
    plt.plot(m_log_alphas, model.coef_path_.T)
    plt.axvline(-np.log10(model.alpha_), color='k',label='alpha CV')
    plt.ylabel('Regression Coefficients')
    plt.legend()
    plt.xlabel('-log(alpha)')
    plt.title('Regression Coefficients Progression for Lasso Paths\nUser Active YN')
    plt.show()
    # print variable names and regression coefficients
    var_imp = pd.DataFrame(data = {'predictors':list(predvar.columns.values),'coefficients':model.coef_})
    var_imp['sort'] = var_imp.coefficients.abs()
    print("User Active YN Coefficients")
    print(var_imp.sort_values(by='sort', ascending=False))

    #target2
    pred_train, pred_test, tar_train, tar_test = train_test_split(predictors2, target2,
    test_size=.3)
    model=LassoLarsCV(cv=10, precompute=False).fit(pred_train,tar_train)
    m_log_alphas = -np.log10(model.alphas_)
    ax = plt.gca()
    plt.plot(m_log_alphas, model.coef_path_.T)
    plt.axvline(-np.log10(model.alpha_), color='k',label='alpha CV')
    plt.ylabel('Regression Coefficients')
    plt.legend()
    plt.xlabel('-log(alpha)')
    plt.title('Regression Coefficients Progression for Lasso Paths\nUser Activity Score')
    plt.show()
    # print variable names and regression coefficients
    var_imp = pd.DataFrame(data = {'predictors':list(predvar.columns.values),'coefficients':model.coef_})
    var_imp['sort'] = var_imp.coefficients.abs()
    print("User Activity Score Coefficients")
    print(var_imp.sort_values(by='sort', ascending=False))
    '''    User Active YN Coefficients
        coefficients             predictors      sort
    3       0.197839   dup_protocol_started  0.197839
    8       0.049685            caffeine_yn  0.049685
    5       0.046396  dup_protocol_finished  0.046396
    13      0.027296     usual_activity_len  0.027296
    7       0.027225             married_yn  0.027225
    9       0.019618        menstruation_yn  0.019618
    20      0.016957       bio_sex_answered  0.016957
    11      0.014370  usual_medications_len  0.014370
    12      0.013858         usual_diet_len  0.013858
    19      0.009921    blood_type_answered  0.009921
    6       0.007458             alcohol_yn  0.007458
    18     -0.003364      pregnant_answered  0.003364
    14     -0.002166         smoke_answered  0.002166
    16      0.000000       married_answered  0.000000
    21      0.000000  menstruation_answered  0.000000
    17      0.000000      caffeine_answered  0.000000
    0       0.000000            pregnant_yn  0.000000
    15      0.000000       alcohol_answered  0.000000
    1       0.000000                bio_sex  0.000000
    10      0.000000   usual_conditions_len  0.000000
    4       0.000000               smoke_yn  0.000000
    2       0.000000    dup_protocol_active  0.000000
    22      0.000000      height_likelihood  0.000000
    User Activity Score Coefficients
        coefficients             predictors      sort
    13      0.492935     usual_activity_len  0.492935
    3       0.356365   dup_protocol_started  0.356365
    4       0.256695               smoke_yn  0.256695
    8       0.198926            caffeine_yn  0.198926
    9       0.174969        menstruation_yn  0.174969
    5      -0.166565  dup_protocol_finished  0.166565
    11      0.077961  usual_medications_len  0.077961
    1      -0.041830                bio_sex  0.041830
    6       0.027837             alcohol_yn  0.027837
    16      0.000000       married_answered  0.000000
    21      0.000000  menstruation_answered  0.000000
    20      0.000000       bio_sex_answered  0.000000
    19      0.000000    blood_type_answered  0.000000
    18      0.000000      pregnant_answered  0.000000
    17      0.000000      caffeine_answered  0.000000
    0       0.000000            pregnant_yn  0.000000
    15      0.000000       alcohol_answered  0.000000
    14      0.000000         smoke_answered  0.000000
    12      0.000000         usual_diet_len  0.000000
    10      0.000000   usual_conditions_len  0.000000
    7       0.000000             married_yn  0.000000
    2       0.000000    dup_protocol_active  0.000000
    22      0.000000      height_likelihood  0.000000'''


def add_months(user_profile_df):
    user_profile_df['month_created'] = user_profile_df['created_date'].dt.month.astype(int)
    user_profile_df = pd.get_dummies(user_profile_df, columns=['month_created'])
    for num in range(1,13):
        if f'month_created_{num}' in user_profile_df.columns:
            continue
        else:
            user_profile_df[f'month_created_{num}'] = [0]*len(user_profile_df.index)
    return user_profile_df

#df_to_csv(create_main_df(), "main_df.csv")
#main_df = csv_to_df("main_df.csv")
#main_df = create_smaller_main_df()
#print(main_df.info())
main_df = clean_data_types(create_smaller_main_df())
counts_df = get_counts(main_df)
counts_df = counts_df.merge(get_days_active(main_df), how='left',on = 'user_id')
counts_df['user_activity_cnt'] = counts_df['entry_chosen_datetime_cnt']+counts_df['entry_id_cnt']
counts_df['user_activity_score'] = counts_df['user_activity_cnt'].astype(float)/counts_df['days_active'].astype(float)

# counts_without_zero = counts_df.loc[counts_df['entry_chosen_datetime_cnt']>0]
# counts_without_zero = counts_without_zero.loc[counts_without_zero['entry_id_cnt']>0]
# counts_without_zero = counts_without_zero.loc[counts_without_zero['days_active'].dt.days>0]

user_profile_df = query_to_dataframe('SELECT * FROM user_table;')
user_profile_df.rename(index=str, columns={'_id': "user_id"}, inplace=True)
#user_profile_df['user_id'] = user_profile_df.user_id.str.replace('x', '.').astype(float)


user_profile_df = user_profile_df.merge(counts_df[['user_id','user_activity_score']],how='left',on = 'user_id')
user_profile_df['user_active_yn'] =  np.where(user_profile_df['user_activity_score']==0,0,1)

hypothesis_df = user_profile_df[['user_id','user_active_yn','user_activity_score','created_date']]
dec_vs_other_months(hypothesis_df)



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

corr_heatmap_with_values(user_profile_df)

lasso_attempt(user_profile_df.drop('user_id',axis=1))


sns.set(style="ticks")
for column in user_profile_df.columns:
    if column not in ['user_activity_score','user_active_yn']:
        sns.pairplot(user_profile_df[['user_active_yn', column, 'user_activity_score']], hue="user_active_yn")
        plt.show()

# scatter_matrix(user_profile_df.drop(['dup_protocol_started','caffeine_yn', 'usual_diet_len', 'alcohol_yn', 'height_likelihood', 'menstruation_yn','married_yn', 'usual_activity_len', 'pregnant_answered', 'bio_sex_answered', 'blood_type_answered', 'usual_conditions_len'],axis=1), alpha=0.2, figsize=(6, 6), diagonal='kde')
# n = len(user_profile_df.drop(['dup_protocol_started','caffeine_yn', 'usual_diet_len', 'alcohol_yn', 'height_likelihood', 'menstruation_yn','married_yn', 'usual_activity_len', 'pregnant_answered', 'bio_sex_answered', 'blood_type_answered', 'usual_conditions_len'],axis=1).columns)
# for x in range(n):
#     for y in range(n):
#         # to get the axis of subplots
#         ax = axs[x, y]
#         # to make x axis name vertical
#         ax.xaxis.label.set_rotation(90)
#         # to make y axis name horizontal
#         ax.yaxis.label.set_rotation(0)
#         # to make sure y axis names are outside the plot area
#         ax.yaxis.labelpad = 50
# plt.show()
# scatter_matrix(user_profile_df.drop(['married_answered','menstruation_answered', 'caffeine_answered', 'pregnant_yn', 'alcohol_answered', 'smoke_answered','bio_sex', 'dup_protocol_finished', 'smoke_yn', 'dup_protocol_active', 'usual_medications_len'],axis=1), alpha=0.2, figsize=(6, 6), diagonal='kde')
# n = len(user_profile_df.drop(['married_answered','menstruation_answered', 'caffeine_answered', 'pregnant_yn', 'alcohol_answered', 'smoke_answered','bio_sex', 'dup_protocol_finished', 'smoke_yn', 'dup_protocol_active', 'usual_medications_len'],axis=1).columns)
# for x in range(n):
#     for y in range(n):
#         # to get the axis of subplots
#         ax = axs[x, y]
#         # to make x axis name vertical
#         ax.xaxis.label.set_rotation(90)
#         # to make y axis name horizontal
#         ax.yaxis.label.set_rotation(0)
#         # to make sure y axis names are outside the plot area
#         ax.yaxis.labelpad = 50
# plt.show()


'''with months removed
User Active YN Coefficients
    coefficients             predictors      sort
3       0.097547   dup_protocol_started  0.097547
8       0.029384            caffeine_yn  0.029384
12      0.024942         usual_diet_len  0.024942
6       0.013645             alcohol_yn  0.013645
22      0.013267      height_likelihood  0.013267
9       0.011293        menstruation_yn  0.011293
7       0.010622             married_yn  0.010622
13      0.009749     usual_activity_len  0.009749
18     -0.006906      pregnant_answered  0.006906
20      0.004162       bio_sex_answered  0.004162
19     -0.003268    blood_type_answered  0.003268
10      0.001989   usual_conditions_len  0.001989
16      0.000000       married_answered  0.000000
21      0.000000  menstruation_answered  0.000000
17      0.000000      caffeine_answered  0.000000
0       0.000000            pregnant_yn  0.000000
15      0.000000       alcohol_answered  0.000000
14      0.000000         smoke_answered  0.000000
1       0.000000                bio_sex  0.000000
5       0.000000  dup_protocol_finished  0.000000
4       0.000000               smoke_yn  0.000000
2       0.000000    dup_protocol_active  0.000000
11      0.000000  usual_medications_len  0.000000    '''

# plt.hist(counts_df['entry_chosen_datetime_cnt'], color='cyan', label='unique_days_with_entries')
# plt.hist(counts_df['entry_id_cnt'],alpha=0.7, color='pink', label='data_points_per_day')
# plt.ylabel('number of users')
# plt.xlabel('count per variable')
# plt.legend()
# plt.show()
# plt.scatter(counts_df['entry_chosen_datetime_cnt'], counts_df['entry_id_cnt'])
# plt.xlabel("entry_chosen_datetime_cnt")
# plt.ylabel("entry_id_cnt")
# plt.show()
# plt.scatter(counts_df['entry_chosen_datetime_cnt']/counts_df['days_active'].replace((0), (1)), counts_df['entry_id_cnt']/counts_df['days_active'].replace((0), (1)))
# plt.xlabel("unique days with entry by total days active")
# plt.ylabel("average data points per total day active")
# plt.show()

# counts_df['days_active'].replace((0), (1),inplace=True)
# counts_without_zero = counts_df.loc[counts_df['entry_chosen_datetime_cnt']>0]
# counts_without_zero = counts_without_zero.loc[counts_without_zero['entry_id_cnt']>0]
# plt.scatter(np.log(counts_without_zero['entry_chosen_datetime_cnt']/counts_without_zero['days_active']), np.log(counts_without_zero['entry_id_cnt']/counts_without_zero['days_active']))
# plt.xlabel("log unique days with entry by total days active")
# plt.ylabel("log average data points per total day active")
# plt.title('Without Zeros')
# plt.show()
