'need to add in user_table data'
'need to add in average count data'


CREATE TABLE data_value_per_user AS(
  SELECT entry.chosen_user,
          (COUNT(DISTINCT(entry.usual_activity))+
              COUNT(DISTINCT(entry.pregnant_yn))+
              COUNT(DISTINCT(entry.text))+
              COUNT(DISTINCT(entry.image))+
              COUNT(DISTINCT(entry.preset_array_protocol))+
              COUNT(DISTINCT(entry.age_yrs))+
              COUNT(DISTINCT(entry.bio_sex))+
              COUNT(DISTINCT(entry.preset_just))+
              COUNT(DISTINCT(entry.location_lng))+
              COUNT(DISTINCT(entry.preset_array_text))+
              COUNT(DISTINCT(entry.smoke_yn))+
              COUNT(DISTINCT(entry.location_lat))+
              COUNT(DISTINCT(entry.location))+
              COUNT(DISTINCT(entry.usual_conditions))+
              COUNT(DISTINCT(entry.usual_medications))+
              COUNT(DISTINCT(entry.blood_type))+
              COUNT(DISTINCT(entry.image_name))+
              COUNT(DISTINCT(entry.preset_array_amount))+
              COUNT(DISTINCT(entry.alcohol_yn))+
              COUNT(DISTINCT(entry.married_yn))+
              COUNT(DISTINCT(entry.caffeine_yn))+
              COUNT(DISTINCT(entry.protocol_list))+
              COUNT(DISTINCT(entry.height_cm))+
              COUNT(DISTINCT(entry.menstruation_yn))+
              COUNT(DISTINCT(entry.preset_array))+
              COUNT(DISTINCT(entry.location_address))+
              COUNT(DISTINCT(entry.usual_diet))+
              COUNT(DISTINCT(entry.modified_date))+
              COUNT(DISTINCT(entry.chosen_datetime))+
              COUNT(DISTINCT(entry.created_date)))
              AS user_interaction_low,
          (COUNT(entry.usual_activity)+
              COUNT(entry.pregnant_yn)+
              COUNT(entry.text)+
              COUNT(entry.image)+
              COUNT(entry.preset_array_protocol)+
              COUNT(entry.age_yrs)+
              COUNT(entry.bio_sex)+
              COUNT(entry.preset_just)+
              COUNT(entry.location_lng)+
              COUNT(entry.preset_array_text)+
              COUNT(entry.smoke_yn)+
              COUNT(entry.location_lat)+
              COUNT(entry.location)+
              COUNT(entry.usual_conditions)+
              COUNT(entry.blood_type)+
              COUNT(entry.usual_medications)+
              COUNT(entry.image_name)+
              COUNT(entry.preset_array_amount)+
              COUNT(entry.alcohol_yn)+
              COUNT(entry.married_yn)+
              COUNT(entry.caffeine_yn)+
              COUNT(entry.protocol_list)+
              COUNT(entry.height_cm)+
              COUNT(entry.menstruation_yn)+
              COUNT(entry.preset_array)+
              COUNT(entry.location_address)+
              COUNT(entry.usual_diet)+
              COUNT(entry.modified_date)+
              COUNT(entry.chosen_datetime)+
              COUNT(entry.created_date))
              AS user_interaction_high,
            COUNT(DISTINCT(entry.usual_activity)) AS usual_activity_udp,
            COUNT(DISTINCT(entry.pregnant_yn)) AS pregnant_yn_udp,
            COUNT(DISTINCT(entry.text)) AS text_udp,
            COUNT(DISTINCT(entry.image)) AS image_udp,
            COUNT(DISTINCT(entry.preset_array_protocol)) AS pa_protocal_udp,
            COUNT(DISTINCT(entry.age_yrs)) AS age_yrs_udp,
            COUNT(DISTINCT(entry.bio_sex)) AS bio_sex_udp,
            COUNT(DISTINCT(entry.preset_just))  AS preset_just_udp,
            COUNT(DISTINCT(entry.location_lng)) AS loc_lng_udp,
            COUNT(DISTINCT(entry.preset_array_text)) AS pa_text_udp,
            COUNT(DISTINCT(entry.smoke_yn)) AS smoke_yn_udp,
            COUNT(DISTINCT(entry.location_lat)) AS location_lat_udp,
            COUNT(DISTINCT(entry.location)) AS location_udp,
            COUNT(DISTINCT(entry.usual_conditions)) AS usual_conditions_udp,
            COUNT(DISTINCT(entry.blood_type)) AS blood_type_udp,
            COUNT(DISTINCT(entry.usual_medications)) AS usual_medications_udp,
            COUNT(DISTINCT(entry.image_name)) AS image_name_udp,
            COUNT(DISTINCT(entry.preset_array_amount)) AS pa_amount_udp,
            COUNT(DISTINCT(entry.alcohol_yn)) AS alcohol_yn_udp,
            COUNT(DISTINCT(entry.married_yn)) AS married_yn_udp,
            COUNT(DISTINCT(entry.caffeine_yn)) AS caffeine_yn_udp,
            COUNT(DISTINCT(entry.protocol_list)) AS protocol_list_udp,
            COUNT(DISTINCT(entry.height_cm)) AS height_cm_udp,
            COUNT(DISTINCT(entry.menstruation_yn)) AS menstruation_yn_udp,
            COUNT(DISTINCT(entry.preset_array)) AS pa_udp,
            COUNT(DISTINCT(entry.location_address)) AS location_address_udp,
            COUNT(DISTINCT(entry.usual_diet)) AS usual_diet_udp,
            COUNT(DISTINCT(entry.modified_date)) AS modified_date_udp,
            COUNT(DISTINCT(entry.chosen_datetime)) AS chosen_datetime_udp,
            COUNT(DISTINCT(entry.created_date)) AS created_date_udp
    FROM entry
    JOIN (SELECT _id, tutorial_plan_done, fb_id, poll_yn, dup_protocol_active, onboarding_yn 'need more here' FROM user_table WHERE  created_date <= '03/03/2018')
    ON entry.chosen_user == user_t._id
    GROUP BY entry.chosen_user);
