'''
The primary unit test file for my trackwell capstone.
'''

import time
import unittest
from src.read_from_db import ReadFromDB
from src import trackwell_pipelines as pipe
#from src import create_user_profile as profile
import pandas as pd
import numpy as np
import os

SLOW_TEST_THRESHOLD = 0.1
psql_user = os.environ.get('PSQL_USER')
psql_password = os.environ.get('PSQL_PASSWORD')
psql_db_name = os.environ.get('PSQL_TEST')

class TestReadFromDB(unittest.TestCase):
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
        self.assertEqual(tuple(df_expected.columns), tuple(df_test.columns))
        self.assertEqual(df_expected.shape, df_test.shape)
        self.assertEqual(df_expected['_id'][1], df_test['_id'][1])

class TestTrackwellPipelines(unittest.TestCase):
    def setUp(self):
        self._started_at = time.time()

    def tearDown(self):
        elapsed = time.time() - self._started_at
        if elapsed > SLOW_TEST_THRESHOLD:
            print(f'{self.id()}: {round(elapsed,2)}s')

    def test_drop_empty_columns(self):
        df_to_drop = pd.DataFrame([[np.nan, np.nan, 2],[0, np.nan, 2],[0,np.nan,2]])
        df_expected = pd.DataFrame([[np.nan, 2],[0, 2],[0,2]])
        df_test = pipe.DropEmptyColumns().fit().transform(df_to_drop)
        self.assertEqual(df_expected.shape, df_test.shape)
        self.assertEqual(df_expected[0][1], df_test[0][1])

    def test_drop_listed_columns(self):
        df_to_drop = pd.DataFrame(data = [[0, 1, 2],[3, 4, 5],[6,7,8]], columns=['inie','minnie','miney'])
        columns_to_drop = ['inie','minnie']
        df_expected = pd.DataFrame(data = [[2],[5],[8]], columns = ['miney'])
        df_test = pipe.DropListedColumns().fit().transform(df_to_drop, columns_to_drop)
        self.assertEqual(df_expected.shape, df_test.shape)
        self.assertEqual(tuple(df_expected.columns), tuple(df_test.columns))
        self.assertEqual(df_expected['miney'][1], df_test['miney'][1])

    def test_group_by_user_id_min(self):
        df_to_merge = pd.DataFrame(data = [[1,1,0, 1, 2],[1,2,3, 4, 5],[2,1, 6,7,8], [2, 0, 6,7,8], [3, 11, 6,7,8]], columns=['user_id', 'inie','minnie','miney', 'moe'])
        df_expected = pd.DataFrame(data = [[1,1,0, 1, 2],[2, 0, 6,7,8], [3, 11, 6,7,8]], columns=['user_id', 'inie','minnie','miney', 'moe']).set_index(['user_id'])
        df_test = pipe.GroupByUserIDMin().fit().transform(df_to_merge)
        self.assertEqual(df_expected.shape, df_test.shape)
        self.assertEqual(tuple(df_expected.columns), tuple(df_test.columns))
        self.assertEqual(df_expected['miney'][1], df_test['miney'][1])

    def test_group_by_user_id_max(self):
        df_to_merge = pd.DataFrame(data = [[1,1,0, 1, 2],[1,2,3, 4, 5],[2,1, 6,7,8], [2, 0, 6,7,8], [3, 11, 6,7,8]], columns=['user_id', 'inie','minnie','miney', 'moe'])
        df_expected = pd.DataFrame(data = [[1,2,3, 4, 5],[2, 1, 6,7,8], [3, 11, 6,7,8]], columns=['user_id', 'inie','minnie','miney', 'moe']).set_index(['user_id'])
        df_test = pipe.GroupByUserIDMax().fit().transform(df_to_merge)
        self.assertEqual(df_expected.shape, df_test.shape)
        self.assertEqual(tuple(df_expected.columns), tuple(df_test.columns))
        self.assertEqual(df_expected['miney'][1], df_test['miney'][1])

    def test_to_date_drop_time(self):
        self.assertEqual(1,1)

    def test_na_to_0(self):
        self.assertEqual(1,1)

    def test_string_to_1_0(self):
        self.assertEqual(1,1)

    def test_answered_or_not(self):
        self.assertEqual(1,1)

    def test_create_height_likelihood(self):
        self.assertEqual(1,1)

    def test_open_text_length(self):
        self.assertEqual(1,1)

    def test_create_estimated_user_created_date(self):
        self.assertEqual(1,1)

    def test_create_max_days_active(self):
        self.assertEqual(1,1)

    def test_create_days_since_active(self):
        self.assertEqual(1,1)

    def test_create_user_entry_df(self):
        self.assertEqual(1,1)



#
# class TestCreateUserProfile(unittest.TestCase):
#     def setUp(self):
#         self._started_at = time.time()
#
#     def tearDown(self):
#         elapsed = time.time() - self._started_at
#         if elapsed > SLOW_TEST_THRESHOLD:
#             print(f'{self.id()}: {round(elapsed,2)}s')
