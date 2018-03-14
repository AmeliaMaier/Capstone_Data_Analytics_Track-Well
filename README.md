# Capstone
## Data Science Capstone Project

### Trackwell
My capstone is using data from the website Trackwell. This site was created to allow users to upload 'protocols' and 'hypothesis tests' for themselves and others to use with the hopes of attracting enough people and collecting enough data to allow for conclusions to be drawn on the 'protocols' or experiments being run.

### Questions
The original questions that the owner of Trackwell asked me where:
- Can we develop a score for user participation so I can rank users and find those who are most involved?
- Can we predict whether a user will be more or less involved in the future?
- Can we develop a threshold below which a user isn't inputting enough data for us to do any analysis?

The questions I am answering in this section are:

- Is there a statistical difference in activity level, days active, and/or amount of data provided by users that signed up due to a large push from a fitness blogger?
- Can we develop a score for user participation so I can rank users and find those who are most involved?
  * I am using the (last date a user entered data on the site) - (the day the user signed up on the site) to determine how many days a user is/was active
  * I am using the (number of unique, non-null data points associated with their profile data) + (number of followup entries) to estimate the total data points provided by one user. This doesn't include location, name or social media information as that data is removed from the anonymized data set I am working with.
  * I am dividing the number of data points provided by a user by their total days active to create a user activity score that can be compared with other users.
- Can we predict whether a user will be more or less involved in the future?
  * I am starting with:
    - Can we predict if a user will be involved at all based on their profile data?
    - If they will be involved, can we prodict how involved based on their profile data?
  * If I have time, I will also look at:
    - Based on past involvement or profile data, can we predict how long a user will remain active?
### Exploratory Data Analysis 
There are no pre-defined site involvment levels, tracking of how long a user has been active/inactive or other metrix tracking so those features will have to be created from existing data.
- Total days active: (last date a user entered data on the site) - (the day the user signed up on the site)
- Days since last activity: (last date a user entered data on the site) - 03/03/2018
- Estimated total data points: (number of unique, non-null data points associated with their profile data) + (number of followup entries)
- User activity score: (Estimated total data points) / (Total days active)

There appears to have been a manual data load into the database at somepoint as there are some users with 0-1 Total days active but 2000 data points. In those cases, the first date with an entry assigned to it will take the place of theire sign up date. Profile data was also manipulated to determine if the data entered or the the fact that anything was entered was more important:
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
  * This was determined with a very forgiving normal distribution based on the average of the mean heights for men and women and adding together the standard distribution for the two groups. It is not meant to predict if they were accurate but to simply show where heights provided weren't feasible (40 cm for example).
  
Because these features are based on existing features, there are likely to be highly correlated sets in the full feature map. I intend to pick only some of the features, avoiding correlated groups, for the final models.
