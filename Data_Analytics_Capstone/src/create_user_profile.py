'''
file creates the csv for the user_profile_df. Should only be run once for each date range

'''
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from trackwell_pipelines import *
import read_from_db as ReadFromDB
from sklearn.externals import joblib

psql_user = os.environ.get('PSQL_USER')
psql_password = os.environ.get('PSQL_PASSWORD')
psql_db_name = os.environ.get('PSQL_TRACKWELL')


def create_user_profile():
    user_df = read_single_table('user_table')
    entry_df = read_single_table('entry_df')
    user_df_pipeline = Pipeline([('CreateUserEntryDF', CreateUserEntryDF())])

    x_pipeline = Pipeline([
        ('DropEmptyColumns', DropEmptyColumns()),
        ('MaleFemaleInts', StringsTo1_0()),
        ('AnsweredOrNot', AnsweredOrNot()),
        ('TextLength', OpenTextLength()),
        ('FillNA', NAto0()),
        ('HeightLikelihood', CreateHeightLikelihood()),
        ('DropColumns', DropListedColumns())])
    MaleFemaleInts_column_names=['bio_sex']
    MaleFemaleInts_string_tuple=('Male','Female')
    AnsweredOrNot_column_names=['dup_protocol_started', 'smoke_yn','alcohol_yn','married_yn','caffeine_yn', 'pregnant_yn','blood_type', 'bio_sex', 'menstruation_yn']
    TextLength_column_names=['usual_conditions', 'usual_medications', 'usual_diet', 'usual_activity']
    FillNA_column_names=['smoke_yn','alcohol_yn','married_yn','caffeine_yn', 'pregnant_yn', 'bio_sex', 'menstruation_yn'],
    HeightLikelihood_column_name='height_cm',
    DropColumns_column_names=['dup_protocol_active', 'dup_protocol_finished', 'modified_date', 'usual_conditions', 'anon_code', 'usual_medications', 'usual_diet', 'usual_activity', 'protocol_array_list', 'dup_protocol_started','blood_type', 'created_date', 'height_cm']
    x_pipeline = x_pipeline.set_params(MaleFemaleInts_column_names=MaleFemaleInts_column_names, MaleFemaleInts_string_tuple=MaleFemaleInts_string_tuple, AnsweredOrNot_column_names=AnsweredOrNot_column_names, TextLength_column_names=TextLength_column_names, FillNA_column_names=FillNA_column_names, HeightLikelihood_column_name=HeightLikelihood_column_name, DropColumns_column_names=DropColumns_column_names)


    '''currently at sudo-code level, changing full feature engineering section into pipelines'''
    y_pipeline_eucd = Pipeline([
        ('DropEmptyColumns', DropEmptyColumns()),
        ('GroupMin', GroupByUserIDMin()),
        ('DateTime', ToDateDropTime()),
        ('CreateEUCD', CreateEstimatedUserCreatedDate())])
    y_pipeline_eucd = y_pipeline_eucd.set_params( DateTime_column_names=['entry_chosen_datetime', 'user_created_date'])

    y_pipeline_max = Pipeline([
        ('DropEmptyColumns', DropEmptyColumns()),
        ('GroupMax', GroupByUserIDMax())])

    y_union = FeatureUnion([
        ('EUCD', y_pipeline_eucd),
        ('Max', y_pipeline_max)])

    y_pipeline_dates = Pipeline([
        ('Union', y_union),
        ('CreateMDA', CreateMaxDaysActive()),
        ('CreateDSA', CreateDaysSinceActive()),
        ('SelectColumns', SelectColumns())])
    y_pipeline_dates = y_pipeline_dates.set_params( CreateMDA_column_names=['entry_created_date', 'estimated_created_date'], CreateDSA_column_names=['entry_created_date', 'estimated_created_date'], CreateDSA_data_pull_date=pd.to_datetime('03/03/2018'), SelecctColumns_column_names=['max_active_days', 'days_since_active'])


    y_pipeline_counts = Pipeline([


    ])

    user_df, entry_df, user_entry_df = user_df_pipeline.fit().transform(user_df, entry_df)
    x = x_pipeline.fit_transform(user_df)
    y_dates = y_pipeline_dates.fit_transform(user_entry_df)

    joblib.dump(user_df_pipeline, 'pickles/pipeline1_user_df.pkl')
    joblib.dump(x_pipeline, 'pickles/pipeline2_x.pkl')
    joblib.dump(y_pipeline_dates, 'pickles/pipeline3_y_dates.pkl')
    joblib.dump(y_pipeline_counts, 'pickles/pipeline4_y_counts.pkl')



    counts_df = get_counts(main_df)
    user_profile = clean_user_profile(counts_df)
    df_to_csv(user_profile, 'user_profile.csv')

def read_single_table(table):
    db_conn = ReadFromDB(f'{psql_user}:{psql_password}@{psql_db_name}')
    all_rows_per_table_query = f'SELECT * FROM {table};'
    return db_conn.query_for_df(all_rows_per_table_query)


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




def clean_user_profile(counts_df):
    missing_counts = main_df[['user_id', 'estimated_created_date', 'days_active', 'days_inactive']]
    missing_counts = missing_counts.groupby(['user_id'])['estimated_created_date', 'days_active', 'days_inactive'].min()
    missing_counts = missing_counts.reset_index()
    counts_df = counts_df.merge(missing_counts, how='left', on='user_id')

    counts_df['user_activity_cnt'] = counts_df['entry_chosen_datetime_cnt']+counts_df['entry_id_cnt']
    counts_df['user_activity_score'] = counts_df['user_activity_cnt'].astype(float)/counts_df['days_active'].astype(float)


    user_profile_df = user_profile_df.merge(counts_df[['user_id','user_activity_score', 'user_activity_cnt', 'days_active', 'days_inactive', 'estimated_created_date']],how='left',on = 'user_id')
    user_profile_df['user_active_yn'] =  np.where(user_profile_df['user_activity_score']==0,0,1)

    return user_profile_df

def df_to_csv(df, path):
    df.to_csv(path, ",")

if __name__ = "__main__":
    create_user_profile()
