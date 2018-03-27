'''
The primary unit test file for my trackwell capstone.
'''

import time
import unittest
from src.read_from_db import ReadFromDB
import pandas as pd
import os

SLOW_TEST_THRESHOLD = 0.1
psql_user = os.environ.get('PSQL_USER')
psql_password = os.environ.get('PSQL_PASSWORD')
psql_db_name = os.environ.get('PSQL_TEST')

class TestTrackwell(unittest.TestCase):
    def setUp(self):
        self._started_at = time.time()

    def tearDown(self):
        elapsed = time.time() - self._started_at
        if elapsed > SLOW_TEST_THRESHOLD:
            print(f'{self.id()}: {round(elapsed,2)}s')

    def test_db_connection(self):
        db_conn = ReadFromDB(f'{psql_user}:{psql_password}@{psql_db_name}')
        df_test = db_conn.query_for_df(f'SELECT * FROM user_table;')
        df_expected = pd.DataFrame(data = {'_id': ['1', '2', '3']})
        self.assertEqual(df_expected.columns, df_test.columns)
        self.assertEqual(df_expected['_id'][1], df_test['_id'][1])
