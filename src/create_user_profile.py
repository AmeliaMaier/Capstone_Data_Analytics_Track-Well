'''
file creates the csv for the user_profile_df. Should only be run once for each date range

'''
import pandas as pd
from sklearn.pipeline import make_pipeline
import trackwell_pipelines as pipe

psql_user = os.environ.get('PSQL_USER')
psql_password = os.environ.get('PSQL_PASSWORD')
psql_db_name = os.environ.get('PSQL_TRACKWELL')


def create_user_profile():
    main_df = merge_user_entry_tables()

    '''currently at sudo-code level, changing full feature engineering section into pipelines'''
    pipeline{
        FeatureUnion{'MergeOnUserID'
            #pipeline {DropEmptyColumns(everything but 'user_created_date')},
            pipeline {GroupByUserIDMin(), ToDateDropTime(just 'entry_chosen_datetime' & 'user_created_date'), CreateEstimatedUserCreatedDate()}
            }
        pipline {GroupByUserIDMax(), CreateMaxDaysActive(), CreateDaysSinceActive()

        }
            }

    #main_df = entry_dt_update_type(main_df)
    #main_df = get_estimated_created_date(main_df)
    #main_df = get_days_active(main_df)
    #main_df = get_days_inactive(main_df)
    counts_df = get_counts(main_df)
    user_profile = clean_user_profile(counts_df)
    df_to_csv(user_profile, 'user_profile.csv')


def merge_user_entry_tables():
    '''
    Input: none
    Return: main_df with the user_table and entry table merged together
    '''
    #read in the tables as seperate dataframes
    table_dataframes = []
    tables = ["user_table", "entry"]
    for table in tables:
        all_rows_per_table_query = f'SELECT * FROM {table};'
        table_dataframes.append(query_to_dataframe(all_rows_per_table_query))
    #drop columns with all na
    #table_dataframes[0] = drop_empty_columns(table_dataframes[0])
    #table_dataframes[1] = drop_empty_columns(table_dataframes[1])
    #change column names for user_table
    for column in table_dataframes[0].columns:
        if column == '_id':
            table_dataframes[0].rename(index=str, columns={column: "user_id"}, inplace=True)
        else:
            table_dataframes[0].rename(index=str, columns={column: f"user_{column}"}, inplace=True)
    #change column names for entry table
    for column in table_dataframes[1].columns:
        if column == 'chosen_user':
            table_dataframes[1].rename(index=str, columns={column: "user_id"}, inplace=True)
        elif column == '_id':
            table_dataframes[1].rename(index=str, columns={column: "entry_id"}, inplace=True)
        elif column == 'preset_array':
            table_dataframes[1].rename(index=str, columns={column: "preset_array_id"}, inplace=True)
        else:
            table_dataframes[1].rename(index=str, columns={column: f"entry_{column}"}, inplace=True)
    #return the two dataframes merged together on user_id
    return table_dataframes[0].merge(table_dataframes[1],how='left',on = 'user_id')

# def drop_empty_columns(df):
#     '''
#     Input: a dataframe to clean
#     Return: dataframe without columns that are entirly null
#     '''
#     return df.dropna(axis=1,how='all')

# def entry_dt_update_type(main_df):
#     main_df["entry_chosen_datetime"] = main_df['entry_chosen_datetime'].dt.date
#     main_df["entry_chosen_datetime"] = pd.to_datetime(main_df['entry_chosen_datetime'])
#     return main_df

# def get_estimated_created_date(main_df):
#     estimated_created_date_df = np.minimum(main_df.groupby('user_id').min()['user_created_date'],main_df.groupby('user_id').min()['entry_created_date'])
#     estimated_created_date_df = estimated_created_date_df.reset_index()
#     estimated_created_date_df = estimated_created_date_df.rename(columns={'index': 'user_id', 'user_created_date': 'estimated_created_date'})
#     main_df = main_df.merge(estimated_created_date_df, how='left', on='user_id')
#     return main_df

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

# def get_days_active(main_df):
#     active_days = main_df.groupby('user_id').max()['entry_created_date'] - main_df.groupby('user_id').min()['estimated_created_date']
#     active_days = active_days.fillna(1)
#     active_days = active_days.dt.ceil('1D')
#     active_days = active_days.dt.days.astype(int)
#     active_days = active_days.reset_index()
#     active_days = active_days.rename(columns={'index': 'user_id', 0: 'days_active'})
#     #defining 1 as lowest num of active days possible for math reasons
#     main_df = main_df.merge(active_days, how='left', on='user_id')
#     return main_df

# def get_days_inactive(main_df):
#     inactive_days = pd.to_datetime('03/03/2018') - np.maximum(main_df.groupby('user_id').max()['entry_created_date'],main_df.groupby('user_id').max()['estimated_created_date'])
#     #inactive_days = active_days.fillna(0)
#     inactive_days = inactive_days.dt.days.astype(int)
#     inactive_days = inactive_days.reset_index()
#     inactive_days = inactive_days.rename(columns={'index': 'user_id', 'entry_created_date': 'days_inactive'})
#     main_df = main_df.merge(inactive_days, how='left', on='user_id')
#     return main_df



def clean_user_profile(counts_df):
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

def df_to_csv(df, path):
    df.to_csv(path, ",")

if __name__ = "__main__":
    create_user_profile()
