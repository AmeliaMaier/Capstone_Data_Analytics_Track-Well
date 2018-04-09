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
        df_dates = pd.DataFrame(data = [[pd.to_datetime('2017-11-30 15:16:45.433502912'), pd.to_datetime('2018-11-30 05:16:45.433502912')],[pd.to_datetime('2018-01-02 16:24:13.433502912'), pd.to_datetime('2017-06-03 01:16:45.433502912')]], columns=['dates1', 'dates2'], dtype=object)
        df_expected = pd.DataFrame(data = [[pd.to_datetime('2017-11-30'), pd.to_datetime('2018-11-30')],[pd.to_datetime('2018-01-02'), pd.to_datetime('2017-06-03')]], columns=['dates1','dates2'])
        df_test = pipe.ToDateDropTime().fit().transform(df_dates, ['dates1','dates2'])
        self.assertEqual(df_expected.shape, df_test.shape)
        self.assertEqual(tuple(df_expected.columns), tuple(df_test.columns))
        self.assertEqual(df_expected['dates1'][1], df_test['dates1'][1])

    def test_na_to_0(self):
        df_to_change = pd.DataFrame(data = [[np.nan, np.nan, 2],[0, np.nan, 2],[0,np.nan,np.nan]], columns=['inie','minnie','miney'])
        columns_to_change = ['inie','minnie']
        df_expected = pd.DataFrame(data = [[0, 0, 2],[0, 0, 2],[0,0,np.nan]], columns = ['inie','minnie','miney'])
        df_test = pipe.NAto0().fit().transform(df_to_change, columns_to_change)
        self.assertEqual(df_expected.shape, df_test.shape)
        self.assertEqual(tuple(df_expected.columns), tuple(df_test.columns))
        self.assertEqual(df_expected['miney'][1], df_test['miney'][1])
        self.assertEqual(df_expected['inie'][1], df_test['inie'][1])

    def test_string_to_1_0(self):
        df_to_change = pd.DataFrame(data = [['red', 2, 2],['blue', 1, 2],['green', 4, 5]], columns=['inie','minnie','miney'])
        columns_to_change = ['inie']
        df_expected = pd.DataFrame(data = [[0, 2, 2],[1, 1, 2],['green', 4, 5]], columns = ['inie','minnie','miney'])
        df_test = pipe.StringsTo1_0().fit().transform(df_to_change, columns_to_change, ('red', 'blue'))
        self.assertEqual(df_expected.shape, df_test.shape)
        self.assertEqual(tuple(df_expected.columns), tuple(df_test.columns))
        self.assertEqual(df_expected['inie'][2], df_test['inie'][2])
        self.assertEqual(df_expected['inie'][1], df_test['inie'][1])

    def test_answered_or_not(self):
        df_to_change = pd.DataFrame(data = [[np.nan, np.nan, 2],[0, np.nan, 2],[0,np.nan,np.nan]], columns=['inie','minnie','miney'])
        columns_to_change = ['inie','minnie']
        df_expected = pd.DataFrame(data = [[np.nan, np.nan, 2, 0, 0],[0, np.nan, 2, 1, 0],[0,np.nan, np.nan, 1, 0]], columns = ['inie','minnie','miney', 'inie_answered','minnie_answered'])
        df_test = pipe.AnsweredOrNot().fit().transform(df_to_change, columns_to_change)
        self.assertEqual(df_expected.shape, df_test.shape)
        self.assertEqual(tuple(df_expected.columns), tuple(df_test.columns))
        self.assertEqual(df_expected['miney'][1], df_test['miney'][1])
        self.assertEqual(df_expected['inie'][1], df_test['inie'][1])

    def test_create_height_likelihood(self):
        df_to_change = pd.DataFrame(data = [[np.nan, 1, 2],[0, 4, 5],[177+165/2,7,8]], columns=['inie','minnie','miney'])
        columns_to_change = ['inie']
        df_expected = pd.DataFrame(data = [[0, 1, 2, 0],[0, 4, 5, 0],[177+165/2,7,8, .02]], columns = ['inie','minnie','miney', 'height_likelihood'])
        df_test = pipe.CreateHeightLikelihood().fit().transform(df_to_change, columns_to_change)
        self.assertEqual(df_expected.shape, df_test.shape)
        self.assertEqual(tuple(df_expected.columns), tuple(df_test.columns))
        self.assertEqual(df_expected['miney'][1], df_test['miney'][1])
        self.assertEqual(df_expected['height_likelihood'][0], round(df_test['height_likelihood'][0],2))
        self.assertEqual(df_expected['height_likelihood'][1], round(df_test['height_likelihood'][1],2))
        self.assertEqual(df_expected['height_likelihood'][2], round(df_test['height_likelihood'][2],2))
        self.assertEqual(df_expected['inie'][0], df_test['inie'][0])

    def test_open_text_length(self):
        df_to_change = pd.DataFrame(data = [['123', 'this is text']], columns=['inie','minnie'])
        columns_to_change = ['minnie']
        df_expected = pd.DataFrame(data = [['123', 'this is text', 12]], columns = ['inie','minnie', 'minnie_len'])
        df_test = pipe.OpenTextLength().fit().transform(df_to_change, columns_to_change)
        self.assertEqual(df_expected.shape, df_test.shape)
        self.assertEqual(tuple(df_expected.columns), tuple(df_test.columns))
        self.assertEqual(df_expected['minnie'][0], df_test['minnie'][0])
        self.assertEqual(df_expected['minnie_len'][0], df_test['minnie_len'][0])
        self.assertEqual(df_expected['inie'][0], df_test['inie'][0])

    def test_create_estimated_user_created_date(self):
        df_to_change = pd.DataFrame(data = [[pd.to_datetime('2018-11-30'), pd.to_datetime('2018-11-30')],[pd.to_datetime('2018-01-02'), pd.to_datetime('2017-06-03')], [pd.to_datetime('2018-01-02'), pd.to_datetime('2018-06-03')]], columns=['dates1','dates2'])
        columns = ['dates1','dates2']
        df_expected = pd.DataFrame(data = [[pd.to_datetime('2018-11-30'), pd.to_datetime('2018-11-30'), pd.to_datetime('2018-11-30')],[pd.to_datetime('2018-01-02'), pd.to_datetime('2017-06-03'), pd.to_datetime('2017-06-03')], [pd.to_datetime('2018-01-02'), pd.to_datetime('2018-06-03'), pd.to_datetime('2018-01-02')]], columns=['dates1','dates2', 'estimated_user_created_date'])
        df_test = pipe.CreateEstimatedUserCreatedDate().fit().transform(df_to_change, columns)
        self.assertEqual(df_expected.shape, df_test.shape)
        self.assertEqual(tuple(df_expected.columns), tuple(df_test.columns))
        self.assertEqual(df_expected['dates1'][0], df_test['dates1'][0])
        self.assertEqual(df_expected['dates2'][1], df_test['dates2'][1])
        self.assertEqual(df_expected['estimated_user_created_date'][0], df_test['estimated_user_created_date'][0])
        self.assertEqual(df_expected['estimated_user_created_date'][1], df_test['estimated_user_created_date'][1])
        self.assertEqual(df_expected['estimated_user_created_date'][2], df_test['estimated_user_created_date'][2])

    def test_create_max_days_active(self):
        df_to_change = pd.DataFrame(data = [[pd.to_datetime('2018-11-30'), pd.to_datetime('2018-11-30')],[pd.to_datetime('2018-01-02'), pd.to_datetime('2017-06-03')], [pd.to_datetime('2018-01-02'), pd.to_datetime('2018-06-03')]], columns=['dates1','dates2'])
        columns = ['dates1','dates2']
        df_expected = pd.DataFrame(data = [[pd.to_datetime('2018-11-30'), pd.to_datetime('2018-11-30'), 1],[pd.to_datetime('2018-01-02'), pd.to_datetime('2017-06-03'), 213], [pd.to_datetime('2018-01-02'), pd.to_datetime('2018-06-03'), -152]], columns=['dates1','dates2', 'max_active_days'])
        df_test = pipe.CreateMaxDaysActive().fit().transform(df_to_change, columns)
        self.assertEqual(df_expected.shape, df_test.shape)
        self.assertEqual(tuple(df_expected.columns), tuple(df_test.columns))
        self.assertEqual(df_expected['dates1'][0], df_test['dates1'][0])
        self.assertEqual(df_expected['dates2'][1], df_test['dates2'][1])
        self.assertEqual(df_expected['max_active_days'][0], df_test['max_active_days'][0])
        self.assertEqual(df_expected['max_active_days'][1], df_test['max_active_days'][1])
        self.assertEqual(df_expected['max_active_days'][2], df_test['max_active_days'][2])

    def test_create_days_since_active(self):
        df_to_change = pd.DataFrame(data = [[pd.to_datetime('2018-11-30'), pd.to_datetime('2018-11-30')],[pd.to_datetime('2018-01-02'), pd.to_datetime('2017-06-03')], [pd.to_datetime('2018-01-02'), pd.to_datetime('2018-06-03')]], columns=['dates1','dates2'])
        columns = ['dates1','dates2']
        df_expected = pd.DataFrame(data = [[pd.to_datetime('2018-11-30'), pd.to_datetime('2018-11-30'), -272],[pd.to_datetime('2018-01-02'), pd.to_datetime('2017-06-03'), 60], [pd.to_datetime('2018-01-02'), pd.to_datetime('2018-06-03'), -92]], columns=['dates1','dates2', 'days_since_active'])
        df_test = pipe.CreateDaysSinceActive().fit().transform(df_to_change, columns)
        self.assertEqual(df_expected.shape, df_test.shape)
        self.assertEqual(tuple(df_expected.columns), tuple(df_test.columns))
        self.assertEqual(df_expected['dates1'][0], df_test['dates1'][0])
        self.assertEqual(df_expected['dates2'][1], df_test['dates2'][1])
        self.assertEqual(df_expected['days_since_active'][0], df_test['days_since_active'][0])
        self.assertEqual(df_expected['days_since_active'][1], df_test['days_since_active'][1])
        self.assertEqual(df_expected['days_since_active'][2], df_test['days_since_active'][2])


    def test_create_user_entry_df(self):
        user_df = pd.DataFrame(data = [[1,2],[3,4]], columns = ['_id', 'other'])
        entry_df = pd.DataFrame(data = [[1,'wheee',55,45],[3,'little',56,46], [3,'barne',57,47]], columns = ['chosen_user','_id','preset_array', 'other'])

        user_df_expected = pd.DataFrame(data = [[1,2],[3,4]], columns = ['user_id', 'other'])
        entry_df_expected = pd.DataFrame(data = [[1,'wheee',55,45],[2,'little',56,46], [2,'barne',57,47]], columns = ['user_id','entry_id','preset_array_id', 'other'])

        user_entry_df_expected = pd.DataFrame(data = [[1,2,'wheee',55,45],[3,4,'little',56,46],[3,4,'barne',57,47]], columns = ['user_id', 'user_other','entry_id','preset_array_id', 'entry_other'])

        user_df_test, entry_df_test, user_entry_df_test = pipe.CreateUserEntryDF().fit().transform(user_df, entry_df)

        self.assertEqual(user_df_expected.shape, user_df_test.shape)
        self.assertEqual(tuple(user_df_expected.columns), tuple(user_df_test.columns))
        self.assertEqual(user_df_expected['user_id'][0], user_df_test['user_id'][0])
        self.assertEqual(user_df_expected['other'][1], user_df_test['other'][1])

        self.assertEqual(entry_df_expected.shape, entry_df_test.shape)
        self.assertEqual(tuple(entry_df_expected.columns), tuple(entry_df_test.columns))
        self.assertEqual(entry_df_expected['user_id'][0], entry_df_test['user_id'][0])
        self.assertEqual(entry_df_expected['entry_id'][1], entry_df_test['entry_id'][1])
        self.assertEqual(entry_df_expected['preset_array_id'][0], entry_df_test['preset_array_id'][0])
        self.assertEqual(entry_df_expected['other'][1], entry_df_test['other'][1])

        self.assertEqual(user_entry_df_expected.shape, user_entry_df_test.shape)
        self.assertEqual(tuple(user_entry_df_expected.columns), tuple(user_entry_df_test.columns))
        self.assertEqual(user_entry_df_expected['user_id'][0], user_entry_df_test['user_id'][0])
        self.assertEqual(user_entry_df_expected['user_id'][1], user_entry_df_test['user_id'][1])
        self.assertEqual(user_entry_df_expected['user_id'][2], user_entry_df_test['user_id'][2])
        self.assertEqual(user_entry_df_expected['entry_id'][1], user_entry_df_test['entry_id'][1])
        self.assertEqual(user_entry_df_expected['preset_array_id'][0], user_entry_df_test['preset_array_id'][0])
        self.assertEqual(user_entry_df_expected['entry_other'][1], user_entry_df_test['entry_other'][1])


#
# class TestCreateUserProfile(unittest.TestCase):
#     def setUp(self):
#         self._started_at = time.time()
#
#     def tearDown(self):
#         elapsed = time.time() - self._started_at
#         if elapsed > SLOW_TEST_THRESHOLD:
#             print(f'{self.id()}: {round(elapsed,2)}s')
