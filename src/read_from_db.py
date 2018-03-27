import psycopg2
import pandas as pd
import os
from sqlalchemy import create_engine

psql_user = os.environ.get('PSQL_USER')
psql_password = os.environ.get('PSQL_PASSWORD')
psql_db_name = os.environ.get('PSQL_TRACKWELL')

class ReadFromDB:
    def __init__(self, credentials):
        self.__engine = create_engine(f'postgresql+psycopg2://{credentials}',echo=False)

    def query_for_df(self, query_str):
        return pd.read_sql_query(query_str,con=self.__engine)
