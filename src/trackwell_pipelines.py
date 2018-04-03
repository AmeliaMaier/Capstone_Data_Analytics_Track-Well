import pandas as pd

DATA_PULL_DATE = pd.to_datetime('03/03/2018')

class DropEmptyColumns:
    '''drops all columns that are entirely empty'''
    def __init__(self):
        self

    def fit(self, *args, **kwargs):
        return self

    def transform(self, df, **transform_params):
        return df.dropna(axis=1,how='all')

class DropListedColumns:
    '''drops all columns that are entirely empty'''
    def __init__(self):
        self

    def fit(self, *args, **kwargs):
        return self

    def transform(self, df, column_names, **transform_params):
        return df.drop(column_names, axis=1)

class MergeByUserID:
    def __init__(self):
        self

    def fit(self, *args, **kwargs):
        return self

    def transform(self, columns, df, **transform_params):
        return df.groupby('user_id').min()

class GroupByUserIDMin:
    def __init__(self):
        self

    def fit(self, *args, **kwargs):
        return self

    def transform(self, df, **transform_params):
        return df.groupby('user_id').min()

class GroupByUserIDMin:
    def __init__(self):
        self

    def fit(self, *args, **kwargs):
        return self

    def transform(self, df, **transform_params):
        return df.groupby('user_id').max()

class ToDateDropTime:
    '''takes one or more date time column and drops the time portion'''
    def __init__(self):
        self

    def fit(self, *args, **kwargs):
        return self

    def transform(self, columns, **transform_params):
        columns = columns.dt.date
        return pd.to_datetime(columns)

class NAto0:
    '''takes one or more date time column and fills NA with zero'''
    def __init__(self):
        self

    def fit(self, *args, **kwargs):
        return self

    def transform(self, df, column_names, **transform_params):
        for column_name in column_names:
            df[column_name] = df[column_name].fillna(0)
        return df

class StringsTo1_0:
    '''takes a column and changes it to 0 and 1, the strings and column name must be provided'''
    def __init__(self):
        self

    def fit(self, *args, **kwargs):
        return self

    def transform(self, df, column_name, string_tuple, **transform_params):
        '''
            Input: df - the dataframe being worked with
            column_name: the string column name for the column to be changed
            string_tuple: the two strings to be replaced with ints
        '''
        return df[column_name].replace((string_tuple[0],string_tuple[1]), (0,1))

class AnsweredOrNot:
    '''takes a list of columns and changes all missing values to 0 and all filled values to 1'''
    def __init__(self):
        self

    def fit(self, *args, **kwargs):
        return self

    def transform(self, df, column_names, **transform_params):
        for column_name in column_names:
            df[f'{column_name}_answered'] = df[column_name]
            df[f'{column_name}_answered'] = df[f'{column_name}_answered'].fillna(-1)
            df[f'{column_name}_answered'] =  np.where(user_profile_df[f'{column_name}_answered']==-1,0,1)
        return df

class CreateHeightLikelihood:
    '''creates the general likelyhood of the provided height'''
    #so as to not have to pull sensus data, useing http://www.usablestats.com/lessons/normal for average heights and distributions
    #adult male heights are on average 70 inches  (5'10) with a standard deviation of 4 inches. Adult women are on average a bit shorter and less variable in height with a mean height of 65  inches (5'5) and standard deviation of 3.5 inches
    # male_average_height = 177.8
    # male_sd_height = 10.16
    # female_average_height = 165.1
    # female_sd_height = 8.89
    def __init__(self):
        self

    def fit(self, *args, **kwargs):
        return self

    def transform(self, df, **transform_params):
        adult_avg_height = 177+165/2
        adult_sd_height = 20
        adult_norm = stats.norm(adult_avg_height, adult_sd_height)
        df['height_cm'] = df['height_cm'].fillna(0)
        df['height_likelihood'] = df['height_cm'].apply(lambda x: adult_norm.pdf(x))
        return df

class OpenTextLength:
    '''takes a list of columns and changes an open text field to the char count of the field'''
    def __init__(self):
        self

    def fit(self, *args, **kwargs):
        return self

    def transform(self, df, column_names, **transform_params):
        for column_name in column_names:
            df[f'{column_name}_len'] = df[column_name].str.len()
            df[f'{column_name}_len'] = df[f'{column_name}_len'].fillna(0)
        return df

class CreateEstimatedUserCreatedDate:
    '''replaces the user_created_date with the estimated_user_created_date. Some users have entries assigned to dates before they signed up for the service. Because this field is being used to determine how long they have been providing data, the earliest date associated with them between the user_table and the entry table is used as an estimated created date.'''
    def __init__(self):
        self

    def fit(self, *args, **kwargs):
        return self

    def transform(self, columns, **transform_params):
        return np.argmin(columns, axis=0)

class CreateMaxDaysActive:
    '''creates the max days active based on the estimated created date and the most recent entry_created_day. Lowest value allowed is 1.'''
    def __init__(self):
        self

    def fit(self, *args, **kwargs):
        return self

    def transform(self, columns, **transform_params):
        columns['max_active_days'] = columns['entry_created_date'] - columns['estimated_created_date']
        columns['max_active_days'] = columns['max_active_days'].fillna(1)
        columns['max_active_days'] = columns['max_active_days'].dt.ceil('1D')
        columns['max_active_days'] = columns['max_active_days'].dt.days.astype(int)
        return columns

class CreateDaysSinceActive:
    '''creates the days since active based on the last day they logged data compared to the date the data was pulled. Lowest value allowed is 0.'''
    def __init__(self):
        self

    def fit(self, *args, **kwargs):
        return self

    def transform(self, columns, **transform_params):
        columns['days_since_active'] = DATA_PULL_DATE - np.maximum(columns['entry_created_date'],columns['estimated_created_date'])
        columns['days_since_active'] = columns['days_since_active'].dt.days.astype(int)
        return columns

class CreateUserEntryDF:
    '''creates the user_entry df and returns all three data frames with matching column names'''
    def __init__(self):
        self

    def fit(self, *args, **kwargs):
        return self

    def transform(self, user_df, entry_df, **transform_params):
        table_dataframes = [user_df.copy(), entry_df.copy()]

        for column in table_dataframes[0].columns:
            if column == '_id':
                table_dataframes[0].rename(index=str, columns={column: "user_id"}, inplace=True)
                user_df.rename(index=str, columns={column: "user_id"}, inplace=True)
            else:
                table_dataframes[0].rename(index=str, columns={column: f"user_{column}"}, inplace=True)
        #change column names for entry table
        for column in table_dataframes[1].columns:
            if column == 'chosen_user':
                table_dataframes[1].rename(index=str, columns={column: "user_id"}, inplace=True)
                entry_df.rename(index=str, columns={column: "user_id"}, inplace=True)
            elif column == '_id':
                table_dataframes[1].rename(index=str, columns={column: "entry_id"}, inplace=True)
                entry_df.rename(index=str, columns={column: "entry_id"}, inplace=True)
            elif column == 'preset_array':
                table_dataframes[1].rename(index=str, columns={column: "preset_array_id"}, inplace=True)
                entry_df.rename(index=str, columns={column: "preset_array_id"}, inplace=True)
            else:
                table_dataframes[1].rename(index=str, columns={column: f"entry_{column}"}, inplace=True)
        #return the two dataframes merged together on user_id
        user_entry_df = table_dataframes[0].merge(table_dataframes[1],how='left',on = 'user_id')
        return user_df, entry_df, user_entry_df
