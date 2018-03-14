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
- 
    
