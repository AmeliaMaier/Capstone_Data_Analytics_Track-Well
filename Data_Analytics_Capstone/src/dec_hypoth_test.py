import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

def csv_to_df(path):
    return pd.read_csv(path, ",")


def bootstrap(sample_array, resample=10000):
    '''Implement a bootstrap function to randomly draw with replacement from a given sample. The function should take a sample as a numpy ndarray and the number of resamples as an integer (default: 10000). The function should return a list of numpy ndarray objects, each ndarray is one bootstrap sample.'''
    samples = []
    sample_array=sample_array.ravel()
    for num in range(resample):
        samples.append(np.random.choice(sample_array, len(sample_array)))
    return samples

def bootstrap_ci(sample,name, stat_function=np.mean, iterations=1000, ci=95):
    '''Implement a bootstrap_ci function to calculate the confidence interval of any sample statistic (in this case the mean). The function should take a sample, a function that computes the sample statistic, the number of resamples (default: 10000), and the confidence interval (default :95%). The function should return the lower and upper bounds of the confidence interval and the bootstrap distribution of the test statistic.'''
    sample_lst = bootstrap(sample, iterations)
    results = []
    for sample_set in sample_lst:
        results.append(stat_function(sample_set))
    results.sort()
    results = np.array(results)

    '''this does the same as the np.percentile under it'''
    #results_cut = results[int((len(results)*((1-(ci/100))/2))):int(len(results)-(len(results)*((1-(ci/100))/2)))]
    plot_histogram_with_normal(results, name)
    return (np.percentile(results, q=[(100-ci)/2,100-(100-ci)/2]), np.mean(results))

def plot_histogram_with_normal(data, name):
    '''simplest form of a histogram'''
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    x_range = np.linspace(stats.norm.ppf(0.0001, data.mean(), data.std()),stats.norm.ppf(0.9999, data.mean(), data.std()), 100)
    ax.hist(data, normed=True)
    normal = stats.norm.pdf(x_range, data.mean(), data.std())
    normal_line =ax.plot(x_range, normal, label='normal pmf', color='r')
    ax.set_title(f"{name}\nmean:{data.mean():.2f} std:{data.std():.2f}\nKS Test P value:{stats.kstest(data, 'norm',args=[data.mean(), data.std()])[1]:.2f}")
    ax.set_xlabel(name)
    plt.show()

def dec_vs_other_months(hypothesis_df):
    #hypothesis_df['estimated_created_date'] = hypothesis_df['estimated_created_date'].astype('datetime64[ns]')
    hypothesis_df_Dec = hypothesis_df.loc[hypothesis_df['estimated_created_date'].dt.month == 12]
    hypothesis_df_not_Dec = hypothesis_df.loc[hypothesis_df['estimated_created_date'].dt.month < 12]
    dec_active = hypothesis_df_Dec.loc[hypothesis_df_Dec['user_active_yn']==1]
    other_active = hypothesis_df_not_Dec.loc[hypothesis_df_not_Dec['user_active_yn']==1]


    # plt.hist([dec_active['user_activity_score'],other_active['user_activity_score']], 10, histtype='bar', color=['blue','green'], label=['December','All Other Months'])
    # plt.ylabel('User Count')
    # plt.xlabel('User Activity Score')
    # plt.title('User Activity Scores\nDecember vs All Other Months')
    # plt.legend()
    # plt.show()
    #
    # plt.hist([hypothesis_df_Dec['user_active_yn'],hypothesis_df_not_Dec['user_active_yn']], 10, histtype='bar', color=['blue','green'], label=['December','All Other Months'])
    # plt.ylabel('User Count')
    # plt.xlabel('User Active YN')
    # plt.title('User Active YN\nDecember vs All Other Months')
    # plt.legend()
    # plt.show()


    bootstrap_ci(dec_active['user_activity_score'], 'December User Activity Score')
    bootstrap_ci(other_active['user_activity_score'], 'All Other Months User Activity Score')
    bootstrap_ci(hypothesis_df_Dec['user_active_yn'], 'December User Active YN')
    bootstrap_ci(hypothesis_df_not_Dec['user_active_yn'], 'All Other Months User Active YN')

    #dec sample for H on user_active_score
    print("Running ttest on user_active score for Dec vs all other months")
    print(f"Dec Average: {dec_active['user_activity_score'].mean():.2f}")
    print(f'Other Average: {other_active["user_activity_score"].mean():.2f}')
    other_sample = other_active.sample(len(dec_active.index))
    user_active_score_ttest = stats.ttest_ind(dec_active['user_activity_score'], other_sample['user_activity_score'])
    print(user_active_score_ttest)
    #Ttest_indResult(statistic=-5.9016620625301828, pvalue=4.0248341180765505e-09)

    #dec sample for H on user activity yn
    print("Running ttest on user active yn for Dec vs all other months")
    print(f"Dec Average: {hypothesis_df_Dec['user_active_yn'].mean():.2f}")
    print(f'Other Average: {hypothesis_df_not_Dec["user_active_yn"].mean():.2f}')
    hypothesis_df_Dec_sample = hypothesis_df_Dec.sample(len(hypothesis_df_not_Dec.index))
    user_active_ttest = stats.ttest_ind(hypothesis_df_Dec_sample['user_active_yn'], hypothesis_df_not_Dec['user_active_yn'])
    print(user_active_ttest)
    #Ttest_indResult(statistic=-12.497492110151295, pvalue=6.2818586376725302e-35)


    '''
    Running ttest on user_active score for Dec vs all other months
    Dec Average: 0.8813206096033261
    Other Average: 2.769237275134839
    Ttest_indResult(statistic=-14.63079024707694, pvalue=8.6358543621434763e-47)

    Running ttest on user active yn for Dec vs all other months
    Dec Average: 0.30687830687830686
    Other Average: 0.5045742434904996
    Ttest_indResult(statistic=-10.82884489661955, pvalue=8.3293441015901507e-27)

    '''
if __name__ == '__main__':
    user_profile_df = csv_to_df('user_profile.csv')
    user_profile_df['estimated_created_date'] = user_profile_df['estimated_created_date'].astype('datetime64[ns]')
    dec_vs_other_months(user_profile_df[['user_id','user_active_yn','user_activity_score', 'user_activity_cnt', 'days_active', 'days_inactive', 'estimated_created_date']])
