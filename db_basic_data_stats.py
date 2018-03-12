import psycopg2
import pandas as pd
import os
from sqlalchemy import create_engine
import matplotlib.pyplot as plt

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

table_dataframes = []
tables = ["preset", "preset_array", "protocol", "protocol_array",  "scale_option",  "map_protocol_day_entry", "user_table", "entry"]
for table in tables:
    all_rows_per_table_query = f'SELECT * FROM {table};'
    table_dataframes.append(query_to_dataframe(all_rows_per_table_query))
# for df in table_dataframes:
#     print(df.info())
table_dataframes[0].rename(index=str, columns={"_id": "preset_id"}, inplace=True)
table_dataframes[0].dropna(axis=1,how='all', inplace=True)
table_dataframes[1].rename(index=str, columns={"_id": "preset_array_id"}, inplace=True)
table_dataframes[1].dropna(axis=1,how='all', inplace=True)
table_dataframes[2].rename(index=str, columns={"_id": "protocol_id"}, inplace=True)
table_dataframes[2].dropna(axis=1,how='all', inplace=True)
table_dataframes[3].rename(index=str, columns={"_id": "protocol_array_id"}, inplace=True)
table_dataframes[3].dropna(axis=1,how='all', inplace=True)
table_dataframes[4].rename(index=str, columns={"_id": "scale_option_id"}, inplace=True)
table_dataframes[4].dropna(axis=1,how='all', inplace=True)
table_dataframes[5].rename(index=str, columns={"_id": "map_protocol_day_entry_id"}, inplace=True)
table_dataframes[5].dropna(axis=1,how='all', inplace=True)
table_dataframes[6].rename(index=str, columns={"_id": "user_id"}, inplace=True)
table_dataframes[6].dropna(axis=1,how='all', inplace=True)
table_dataframes[7].rename(index=str, columns={"chosen_user": "user_id", "_id": "entry_id", "preset_array":"preset_array_id"}, inplace=True)
table_dataframes[7].dropna(axis=1,how='all', inplace=True)



main_df = ((((table_dataframes[6].merge(table_dataframes[7],how='outer',on = 'user_id')).merge(table_dataframes[5], how='outer', on='entry_id')).merge(table_dataframes[2],how='outer', on='protocol_id')).merge(table_dataframes[1], how='outer', on='preset_array_id')).merge(table_dataframes[3],how='outer', on='protocol_array_id')

print(main_df.info())
print(main_df.describe())
print(main_df.head())
