import psycopg2
import pandas as pd
import os
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import numpy as np

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
    user_signup_df['user_id'] = user_signup_df.user_id.str.replace('x', '.').astype(float)
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

    for df in table_dataframes:
        print(df.info())
    main_df = ((((table_dataframes[6].merge(table_dataframes[7],how='left',on = 'user_id')).merge(table_dataframes[5], how='left', on='entry_id')).merge(table_dataframes[2],how='left', on='protocol_id')).merge(table_dataframes[1], how='left', on='preset_array_id')).merge(table_dataframes[3],how='left', on='protocol_array_id')
    print(main_df.info())
    #print(main_df.describe())
    #print(main_df.head())
    return main_df

def df_to_csv(df, path):
    df.to_csv(path, "|")
def csv_to_df(path):
    return pd.read_csv(path, "|")

#df_to_csv(create_main_df(), "main_df.csv")
#main_df = csv_to_df("main_df.csv")
main_df = create_main_df()
main_df['user_id'] = main_df.user_id.str.replace('x', '.').astype(float)
main_df["entry_chosen_datetime"] = main_df['entry_chosen_datetime'].dt.date
main_df["entry_chosen_datetime"] = pd.to_datetime(main_df['entry_chosen_datetime'])
main_df['user_bio_sex'].replace(('Male','Female'), (1,0), inplace=True)
main_df['entry_bio_sex'].replace(('Male','Female'), (1,0), inplace=True)
main_df['user_pregnant_yn'].fillna(-1, inplace=True)
main_df['entry_pregnant_yn'].fillna(-1, inplace=True)

# user_date_count = main_df.groupby("user_id", as_index=False ).entry_chosen_datetime.nunique()
# user_entry_count = main_df.groupby("user_id", as_index=False ).entry_id.nunique()
counts_df = pd.DataFrame(data=main_df['user_id'].unique(), columns=["user_id"], index=main_df['user_id'].unique())
for column in main_df.columns:
    if column == 'user_id':
        continue
    counts_df[f'{column}_cnt'] = main_df.groupby("user_id")[column].nunique()
counts_df = counts_df.reset_index().drop("index", axis=1)

counts_df.hist('entry_chosen_datetime_cnt')
counts_df.hist('entry_id_cnt')
plt.show()
plt.scatter(counts_df['entry_chosen_datetime_cnt'], counts_df['entry_id_cnt'])
plt.xlabel("entry_chosen_datetime_cnt")
plt.ylabel("entry_id_cnt")
plt.show()

counts_df['data_points'] = counts_df.drop('user_id').sum(axis=0)
