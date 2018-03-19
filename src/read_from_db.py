import psycopg2
import pandas as pd
import os
from sqlalchemy import create_engine

psql_user = os.environ.get('PSQL_USER')
psql_password = os.environ.get('PSQL_PASSWORD')

def query_to_dataframe(query_string):
    '''
    Input: a string containing a sql query
    Return: pandas dataframe containing the results of the query
    '''
    engine = create_engine(f'postgresql+psycopg2://{psql_user}:{psql_password}@localhost:5432/trackwell',echo=False)
    table_df=pd.read_sql_query(query_string,con=engine)
    return table_df


''' still need to reorganize so I can do this:
class DB(object):
    def __init__(self, **credentials):
        self._connect = partial(pymysql.connect, **credentials)

    def query(self, q_str, params):
        with self._connect as conn:
            with conn.cursor() as cur:
                cur.execute(q_str, params)
                return cur.fetchall()

# now for usage

test_credentials = {
    # use credentials to a fake database
}

test_db = DB(**test_credentials)
test_db.query(write_query, list_of_fake_params)
results = test_db.query(read_query)
assert results = what_the_results_should_be
'''
