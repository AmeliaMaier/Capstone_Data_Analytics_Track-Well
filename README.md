# Trackwell
## Data Science Capstone Project

My capstone is using data from the website Trackwell. This site was created to allow users to upload 'protocols' and 'hypothesis tests' for themselves and others to use with the hopes of attracting enough people and collecting enough data to allow for conclusions to be drawn on the 'protocols' or experiments being run. For now, Trackwell needs to know if it is possible to predict which users will provide the most data and/or when a user might stop using the service.

### Questions
The original questions that the owner of Trackwell asked me were:
- Can we develop a score for user participation so I can rank users and find those who are most involved?
- Can we predict whether a user will be more or less involved in the future?
- Can we develop a threshold below which a user isn't inputting enough data for us to do any analysis?

The questions I am answering in this section are:

- Is there a statistical difference in activity level, days active, and/or amount of data provided by users that signed up due to a large push from a fitness blogger?
- Can we develop a score for user participation so I can rank users and find those who are most involved?
  * I am using assigning an activity score per user based on the number of data points they have provided and the number of days they have been active on the service.
- Can we predict whether a user will be more or less involved in the future?
  * Ultimatly, the final answer for this question will probably be a time series model. For now, I will see what I can do with logistic and linear modeling.
  * I am starting with:
    - Can we predict if a user will be involved at all based on their profile data?
    - If they will be involved, can we prodict how involved based on their profile data?
  * If I have time, I will also look at:
    - Based on past involvement or profile data, can we predict how long a user will remain active?
    - Based on past involvement or profile data, can we predict how long a user will go inactive or which will go inactive?
### Exploratory Data Analysis 
There are no pre-defined site involvment levels, tracking of how long a user has been active/inactive or other metrix tracking so those features will have to be created from existing data.
- Total days active: (last date a user entered data on the site) - (the day the user signed up on the site)
- Days since last activity: 03/03/2018 - (last date a user entered data on the site)
- Estimated total data points: (number of unique, non-null data points associated with their profile data) + (number of followup entries)
- User activity score: (Estimated total data points) / (Total days active)
- User active yes/no: yes if user activity score is above 0

There appears to have been a manual data load into the database at somepoint as there are some users with 0-1 Total days active but 2000 data points. In those cases, the first date with an entry assigned to it will take the place of theire sign up date. Profile data was also manipulated to determine if the data entered or the the fact that anything was entered was more important to predicting future activity:
- usual_activity_len: the character count of the entry in the usual_activity field
- dup_protocol_started: changed to a yes/no instead of actual protocol hashes
- dup_protocol_finished: changed to a yes/no instead of actual protocol hashes
- usual_medications_len: the character count of the entry in the usual_medications field
- married_answered: changed to yes/no for if the queston was answered
- menstruation_answered: changed to yes/no for if the queston was answered
- bio_sex_answered: changed to yes/no for if the queston was answered
- blood_type_answered: changed to yes/no for if the queston was answered
- pregnant_answered: changed to yes/no for if the queston was answered
- caffeine_answered: changed to yes/no for if the queston was answered
- alcohol_answered: changed to yes/no for if the queston was answered
- smoke_answered: changed to yes/no for if the queston was answered
- usual_diet_len: the character count of the entry in the usual_diet field
- usual_conditions_len: the character count of the entry in the usual_conditions field
- dup_protocol_active: changed to a yes/no instead of actual protocol hashes
- height_likelihood: the probability that the height reported exists in the adult population
  * This was determined with a very forgiving normal distribution based on the average of the mean heights for men and women and adding together the standard distribution for the two groups. It is not meant to predict if they were accurate but to simply show where heights provided were not feasible (40 cm for example).
  
Because these features are based on existing features, there are likely to be highly correlated sets in the full feature map. I intend to pick only some of the features, avoiding correlated groups, for the final models.

Correlation Heatmap: ![Alt](images/profile_corr_map.png) 

After going back and fixing for the dataload, the timebased correlations became more obviouse:

Total Number of Days Active By Signup Month: ![Total Number of Days Active By Signup Month](images/monthly_days_active.png) 

Days Since Last Activity By Signup Month: ![Days Since Last Activity By Signup Month](images/monthly_days_inactive.png) 

And the thought that the blogger's push for January activity (people signing up in December to start in January) was helpful in increasinng the percentage of involvement becomes less likely, though it does seem to have pushed a higher volume of people to the site:

Activity Score By Signup Month: ![Activity Score By Signup Month](images/monthly_activity.png) 

Signups By Month: ![Signups By Month](images/EDA_Signups_Per_Month-after_start_fix.png)

## Hypothesis Test
__Question:__ Is there a statistical difference in activity level, days active, and/or amount of data provided by users that signed up due to a large push from a fitness blogger?

>>>__H<sub>0</sub>:__ There is no difference in involvment metrics for Dec when compared to the rest of the year.

>>>__H<sub>1</sub>:__ There is a difference in involment metrics for Dec when compared to the rest of the year.

>>>__alpha:__ .025 = (.05/2)

I used a two population t-test for each metric to determine if there was a statistical differance. Due to the time sequence effect on Days Active and Days Inactive, I decided not to include them in the testinng as a difference between Dec and the rest of the year could simply be due to people signing up in Dec having less time to be active or inactive on the service.

Running ttest on User Activity Scores for Dec vs all other months

>>>Dec Average: 0.8813206096033261

>>>Year Average without Dec: 2.769237275134839

>>>Ttest_indResult(statistic=-14.595829266547389, pvalue=1.391951976598514e-46)


Running ttest on Users Active YN for Dec vs all other months

>>>Dec Average: 0.30687830687830686

>>>Year Average without Dec: 0.5045742434904996

>>>Ttest_indResult(statistic=-11.124419257482321, pvalue=3.613288170575678e-28)

__Conclusion:__ In metrics that have been normalized for total time on the system, there does appear to be a statisticaly significant difference in users who signed up in December and all other users. Unfortunantly for Trackwell, this segment of their users appear to be less likely to be active at all on the site and, if they are active, to provide less data per day than other users. 

