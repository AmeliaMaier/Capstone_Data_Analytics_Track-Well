import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

def csv_to_df(path):
    return pd.read_csv(path, ",")

def logistic_regression(df):
    logreg = LogisticRegression()
    columns = ['dup_protocol_started','caffeine_yn', 'married_yn', 'usual_activity_len', 'alcohol_yn', 'bio_sex_answered', 'bio_sex', 'menstruation_yn', 'blood_type_answered', 'usual_diet_len', 'usual_medications_len', 'usual_conditions_len']
    #X=df[['dup_protocol_started','caffeine_yn', 'married_yn', 'usual_activity_len', 'alcohol_yn', 'bio_sex_answered', 'bio_sex', 'menstruation_yn', 'blood_type_answered', 'usual_diet_len', 'usual_medications_len', 'usual_conditions_len']]
    #X=df[['dup_protocol_started','caffeine_yn', 'married_yn', 'usual_activity_len', 'bio_sex_answered', 'menstruation_yn', 'blood_type_answered', 'usual_diet_len', 'usual_medications_len']]
    #X=stand(df[['dup_protocol_started','caffeine_yn', 'married_yn', 'usual_activity_len', 'bio_sex_answered', 'menstruation_yn', 'blood_type_answered', 'usual_diet_len', 'usual_medications_len']])
    X=stand(df[['dup_protocol_started','caffeine_yn', 'married_yn', 'usual_activity_len', 'alcohol_yn', 'bio_sex_answered', 'bio_sex', 'menstruation_yn', 'blood_type_answered', 'usual_diet_len', 'usual_medications_len', 'usual_conditions_len']])

    y=df['user_active_yn']
    logit_model=sm.Logit(y,X)
    result=logit_model.fit()
    print(result.summary())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    print("With train test fix")
    print(f'Mean accuracy of logistic regression classifier on test set: {logreg.score(X_test, y_test)}')
    print(confusion_matrix(y_test, y_pred))
    print("tn, fp\nfn, tp")
    print('\n\n')

    coef_list = pd.DataFrame(data = {'features':columns, 'estimatedCoefficients':logreg.coef_.ravel()})
    coef_list['sort'] = coef_list.estimatedCoefficients.abs()
    print(coef_list.sort_values(by='sort', ascending=False))
    print(f'Estimated intercept: {logreg.intercept_}')

    print("With KFold cross validation")
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    modelCV = LogisticRegression()
    scoring = 'accuracy'
    results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
    print(f"10-fold cross validation average accuracy: {results.mean():.2f}")
    scoring = 'recall'
    results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
    print(f"10-fold cross validation average recall: {results.mean():.2f}")
    scoring = 'precision'
    results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
    print(f"10-fold cross validation average precision: {results.mean():.2f}")
    print(classification_report(y_test, y_pred))

    print_roc_curve(y_test, logreg, X_test, 'Standard Logistic')
    plt.scatter(logreg.predict_proba(X_test)[:,1], y_pred)
    plt.title("Active YN Probability\nStandard Logistic")
    plt.xlabel('Predicted Probability')
    plt.ylabel('Actual Probability')
    plt.show()
    cost_benefit = [[100,-10],[0,0]]
    plot_profit_curve(logreg, cost_benefit, X_train, X_test, y_train, y_test)


def logistic_balanced(df):
    logreg = LogisticRegression()
    columns = ['dup_protocol_started','caffeine_yn', 'married_yn', 'usual_activity_len', 'alcohol_yn', 'bio_sex_answered', 'bio_sex', 'menstruation_yn', 'blood_type_answered', 'usual_diet_len', 'usual_medications_len', 'usual_conditions_len']
    #X=df[['dup_protocol_started','caffeine_yn', 'married_yn', 'usual_activity_len', 'alcohol_yn', 'bio_sex_answered', 'bio_sex', 'menstruation_yn', 'blood_type_answered', 'usual_diet_len', 'usual_medications_len', 'usual_conditions_len']]
    #X=df[['dup_protocol_started','caffeine_yn', 'married_yn', 'usual_activity_len', 'bio_sex_answered', 'menstruation_yn', 'blood_type_answered', 'usual_diet_len', 'usual_medications_len']]
    #X=stand(df[['dup_protocol_started','caffeine_yn', 'married_yn', 'usual_activity_len', 'bio_sex_answered', 'menstruation_yn', 'blood_type_answered', 'usual_diet_len', 'usual_medications_len']])
    X=stand(df[['dup_protocol_started','caffeine_yn', 'married_yn', 'usual_activity_len', 'alcohol_yn', 'bio_sex_answered', 'bio_sex', 'menstruation_yn', 'blood_type_answered', 'usual_diet_len', 'usual_medications_len', 'usual_conditions_len']])
    y=df['user_active_yn']

    # logit_model=sm.Logit(y,X)
    # result=logit_model.fit()
    # print(result.summary())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    logreg = LogisticRegression(class_weight='balanced')
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    print("With train test fix")
    print(f'Mean accuracy of logistic regression classifier on test set: {logreg.score(X_test, y_test)}')
    print(confusion_matrix(y_test, y_pred))
    print("tn, fp\nfn, tp")
    print('\n\n')

    coef_list = pd.DataFrame(data = {'features':columns, 'estimatedCoefficients':logreg.coef_.ravel()})
    coef_list['sort'] = coef_list.estimatedCoefficients.abs()
    print(coef_list.sort_values(by='sort', ascending=False))
    print(f'Estimated intercept: {logreg.intercept_}')

    print("With KFold cross validation")
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    modelCV = LogisticRegression()
    scoring = 'accuracy'
    results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
    print(f"10-fold cross validation average accuracy: {results.mean():.2f}")
    scoring = 'recall'
    results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
    print(f"10-fold cross validation average recall: {results.mean():.2f}")
    scoring = 'precision'
    results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
    print(f"10-fold cross validation average precision: {results.mean():.2f}")
    print(classification_report(y_test, y_pred))

    print_roc_curve(y_test, logreg, X_test, 'Balanced without December')
    plt.scatter(logreg.predict_proba(X_test)[:,1], y_pred)
    plt.title("Active YN Probability\nBalanced without December")
    plt.xlabel('Predicted Probability')
    plt.ylabel('Actual Probability')
    plt.show()
    cost_benefit = [[100,-10],[0,0]]
    plot_profit_curve(logreg, cost_benefit, X_train, X_test, y_train, y_test)

def stand(df):
    scaler = StandardScaler()
    scaler.fit(df)
    return scaler.transform(df)


def lasso_fit_and_print(target_name, predictor, target, predvar_columns):
    pred_train, pred_test, tar_train, tar_test = train_test_split(predictor, target,test_size=.3)
    model=LassoLarsCV(cv=10, precompute=False).fit(pred_train,tar_train)
    m_log_alphas = -np.log10(model.alphas_)
    ax = plt.gca()
    plt.plot(m_log_alphas, model.coef_path_.T)
    plt.axvline(-np.log10(model.alpha_), color='k',label='alpha CV')
    plt.ylabel('Regression Coefficients')
    plt.legend()
    plt.xlabel('-log(alpha)')
    plt.title(f'Regression Coefficients Progression for Lasso Paths\n{target_name}')
    plt.show()
    # print variable names and regression coefficients
    var_imp = pd.DataFrame(data = {'predictors':list(predvar_columns.values),'coefficients':model.coef_})
    var_imp['sort'] = var_imp.coefficients.abs()
    print(f"{target_name} Coefficients")
    print(var_imp.sort_values(by='sort', ascending=False))

def lasso_attempt(data_clean):
    #creating predictors and targets
    predvar_all = data_clean.drop(['user_active_yn','user_activity_score', 'user_activity_cnt', 'days_active', 'days_inactive'], axis=1)
    target1 = data_clean['user_active_yn']
    active_users = data_clean.loc[data_clean['user_active_yn']==1]
    target2 = active_users['user_activity_score']
    target3 = active_users['user_activity_cnt']
    target4 = active_users['days_active']
    target5 = active_users['days_inactive']
    predvar_active = active_users.drop(['user_active_yn','user_activity_score', 'user_activity_cnt', 'days_active', 'days_inactive'], axis=1)
    scaler = StandardScaler()
    scaler.fit(predvar_all)
    predictors_all=scaler.transform(predvar_all)
    scaler.fit(predvar_active)
    predictors_active = scaler.transform(predvar_active)
    lasso_fit_and_print("User Active YN", predictors_all, target1, predvar_all.columns)
    lasso_fit_and_print("User Activity Score", predictors_active, target2, predvar_active.columns)
    lasso_fit_and_print("User Activity Count", predictors_active, target3, predvar_active.columns)
    lasso_fit_and_print("Days Active", predictors_active, target4, predvar_active.columns)
    lasso_fit_and_print("Days Inactive", predictors_active, target5, predvar_active.columns)


def profit_curve(cost_benefit, predicted_probs, y_true):
    thresholds = np.sort(predicted_probs)
    thresholds = np.append(thresholds, [1.0])
    expected_profits = []
    for threshold in thresholds:
        confusion_matrix = standard_confusion_matrix(y_true, predicted_probs >= threshold)
        expected_profits.append(np.sum(confusion_matrix * cost_benefit)/np.sum(confusion_matrix))
    return np.array([thresholds, expected_profits])

def plot_profit_curve(model, cost_benefit, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predicted_probs = model.predict_proba(X_test)
    profits = profit_curve(cost_benefit, predicted_probs[:,1], y_test)
    percentages = profits[0]
    profits = profits[1]
    plt.plot(percentages, profits, label='profit')
    plt.title("Profit Curve")
    plt.xlabel("Percentage of test instances (decreasing by score)")
    plt.ylabel("Profit")
    plt.legend(loc='best')
    plt.show()

def print_roc_curve(y_test, logreg, X_test, title):
    logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC\n{title}')
    plt.legend(loc="lower right")
    #plt.savefig('Log_ROC')
    plt.show()

def standard_confusion_matrix(y_true, y_predict):
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    true_positives = np.sum(np.logical_and(y_predict==1,y_true==1))
    false_positives = np.sum(np.logical_and(y_predict==1,y_true==0))
    false_negatives = np.sum(np.logical_and(y_predict==0,y_true==1))
    true_negatives = np.sum(np.logical_and(y_predict==0,y_true==0))
    confusion_matrix = np.array([[true_positives,false_positives],[false_negatives,true_negatives]])
    return confusion_matrix

if __name__ == '__main__':
    user_profile_df = csv_to_df('user_profile.csv')
    user_profile_df['estimated_created_date'] = user_profile_df['estimated_created_date'].astype('datetime64[ns]')
