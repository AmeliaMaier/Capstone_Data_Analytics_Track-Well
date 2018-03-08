import psycopg2
import pandas as pd
import os
from sqlalchemy import create_engine

psql_user = os.environ.get('PSQL_USER')
psql_password = os.environ.get('PSQL_PASSWORD')


#tables = ["preset", "preset_array", "protocol", "protocol_array", "reminder", "scale_option", "statistic", "map_protocol_day_entry", "user", "entry", "comment"]
tables = ['preset_array']

engine = create_engine(f'postgresql+psycopg2://{psql_user}:{psql_password}@localhost:5432/matt',echo=False)

for table in tables:
    table_df=pd.read_sql_query(f'SELECT * FROM {table};',con=engine)
    conn = psycopg2.connect(dbname='matt', user='amelia', host='/var/run/postgresql')
    print(f'Table Name: {table}')
    print(f'{table_df.info()}')
    print(f'{table_df.describe()}')
    print(f'{table_df.head()}')

