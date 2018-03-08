
import psycopg2
import mysql.connector
import pandas as pd
from sqlalchemy import create_engine
import os

trackwell_user = os.environ.get('TRACKWELL_USER')
trackwell_password = os.environ.get('TRACKWELL_PASSWORD')
trackwell_port = os.environ.get('TRACKWELL_PORT')
trackwell_host = os.environ.get('TRACKWELL_HOST')
trackwell_db = os.environ.get('TRACKWELL_DB')
psql_user = os.environ.get('PSQL_USER')
psql_password = os.environ.get('PSQL_PASSWORD')

print('attempting to connect to trackwell')
sql_cnx = mysql.connector.connect(user=trackwell_user, password=trackwell_password, port=trackwell_port,
                              host=trackwell_host,
                              database=trackwell_db)


#tables = ["preset", "preset_array", "protocol", "protocol_array", "reminder", "scale_option", "statistic", "map_protocol_day_entry", "user", "entry", "comment"]

tables = ['user']

print('attempting to read tables')
for table in tables:
    #read table from trackwell to dataframe
    print(f'reading table {table}')
    df = pd.read_sql(f'SELECT * FROM {table}', con=sql_cnx)
    #write dataframe to local psql database
    print(f'writing table {table} with {df.info()}')
    engine = create_engine(f'postgresql+psycopg2://{psql_user}:{psql_password}@localhost:5432/matt',echo=False)
    df.to_sql(name='user_table', con=engine, if_exists = 'replace', index=False)
    print(f'finished writing table {table}')


sql_cnx.close()
