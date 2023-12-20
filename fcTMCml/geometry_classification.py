



X_train, X_test, y_train, y_test = train_test_split(curr_X, y_truth, test_size=0.2, random_state=42)
#print(y_test)
hyperparams = rc_optimization(X_train, X_test, y_train, y_test)
print(hyperparams)

clf = RidgeClassifier(**hyperparams)
avg_score, ppv, sensitivity, f_score, clf = k_folds(clf, curr_X, y_truth, True)
coeffs = clf.coef_

print(run_ident+" RR (TPE): ", avg_score, ppv, sensitivity, f_score, coeffs)

acc_dict[run_ident+" RR (TPE): "] = {"score": avg_score, "ppv": ppv, "sensitivity": sensitivity, "f_score":f_score, "coeffs":coeffs}

#print(y_test)
hyperparams = rfc_optimization(X_train, X_test, y_train, y_test)
print(hyperparams)

clf = RandomForestClassifier(**hyperparams)
avg_score, ppv, sensitivity, f_score, clf = k_folds(clf, curr_X, y_truth, True)
coeffs = clf.feature_importances_

print(run_ident+" RFC (TPE): ", avg_score, ppv, sensitivity, f_score, coeffs)

acc_dict[run_ident+" RFC (TPE): "] = {"score": avg_score, "ppv": ppv, "sensitivity": sensitivity, "f_score":f_score, "coeffs":coeffs}    