import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def csv_to_df(path):
    return pd.read_csv(path, ",")


def corr_heatmap_with_values(df, title):
    corr = df.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(10,10))
    cmap = sns.color_palette('coolwarm')
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, square=True, linewidths=.5,
                yticklabels=True, annot=True, fmt='.2f', cbar_kws={'shrink':.5})
    plt.title(title, fontsize=20)
    plt.xticks(rotation=60, fontsize=11, horizontalalignment='right')
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()
    plt.show()


def correlation_maps_lasso(user_profile):
    corr_heatmap_with_values(user_profile[['user_active_yn','dup_protocol_started','caffeine_yn','dup_protocol_finished', 'married_yn', 'usual_activity_len', 'alcohol_yn', 'bio_sex_answered', 'bio_sex', 'menstruation_yn', 'blood_type_answered', 'usual_diet_len', 'usual_medications_len', 'usual_conditions_len']], "User Active YN\nLasso Based Correlation Heat Map")

    corr_heatmap_with_values(user_profile[['user_activity_score', 'dup_protocol_started','dup_protocol_finished', 'usual_activity_len', 'alcohol_yn', 'menstruation_yn', 'usual_medications_len', 'usual_conditions_len','caffeine_yn','bio_sex', 'smoke_yn', 'pregnant_yn']], 'User Activity Score\nLasso Based Correlation Heat Map')

    corr_heatmap_with_values(user_profile[['user_activity_cnt', 'usual_activity_len','pregnant_answered', 'caffeine_yn', 'usual_diet_len', 'smoke_answered', 'pregnant_yn', 'dup_protocol_started','usual_medications_len','usual_conditions_len', 'alcohol_yn', 'bio_sex']], 'User Activity Count\nLasso Based Correlation Heat Map')

    corr_heatmap_with_values(user_profile[['days_active', 'pregnant_answered','caffeine_yn', 'alcohol_answered', 'married_answered', 'menstruation_answered', 'usual_diet_len', 'caffeine_answered','usual_activity_len','blood_type_answered', 'married_yn', 'dup_protocol_started',]], 'Days Active\nLasso Based Correlation Heat Map')

    corr_heatmap_with_values(user_profile[['days_inactive', 'height_likelihood','blood_type_answered', 'caffeine_yn', 'dup_protocol_started', 'pregnant_yn', 'smoke_yn', 'dup_protocol_active','smoke_answered','menstruation_yn', 'dup_protocol_finished', 'bio_sex_answered']], 'Days Inactive\nLasso Based Correlation Heat Map')


def signup_rate_hist_monthly():
    #limit by date because the user table was pulled down a few days after all the others and has some extra users in it (people who signed up between 3/3/18 and the second pull
    user_signup_hist_query = """
    SELECT _id AS user_id, created_date AS signup_datetime
        FROM user_table
        WHERE  created_date <= '03/03/2018';
    """
    user_signup_df = query_to_dataframe(user_signup_hist_query)
    user_signup_df['signup_date'] = user_signup_df['signup_datetime'].dt.date
    #user_signup_df['user_id'] = user_signup_df.user_id.str.replace('x', '.').astype(float)
    user_signup_df['signup_date'] =  pd.to_datetime(user_signup_df['signup_date'])
    print(user_signup_df.info())
    print(user_signup_df.head())
    #user_signup_df.hist()
    user_signup_df[['signup_date','user_id']].groupby([user_signup_df["signup_date"].dt.year, user_signup_df["signup_date"].dt.month]).count().plot(kind="bar")
    plt.show()
