#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from pandas import DataFrame
from pandas import concat
from scipy import sparse
import matplotlib.pyplot as plt

from config import config_dict

from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.multiclass import OutputCodeClassifier
from sklearn.multiclass import OneVsRestClassifier

from sklearn.feature_selection import SelectKBest, chi2, f_classif

from Metrics import Metrics


class Classifier(Metrics):

    def __init__(self, X, y, people, df_features, feature_names, conf_dict):

        self.X = X
        self.y = np.array(y)
        self.feature_names = feature_names

        self.people = people.reset_index(drop=True)
        self.X_df = df_features
        self.y_df = y.reset_index(drop=True)

        self.app_list = config_dict["app_list"]
        self.labels_numeric = {name: i for i, name in enumerate(self.app_list)}
        self.n_classes = len(self.labels_numeric)

        self.clf_name = config_dict["classifier"]

#        self.feature_selection = config_dict["feature_selection"] #True/False
#        self.num_features = config_dict["num_features"]
#        self.one_vs_all_type = config_dict["one_vs_all_type"]
        self.feature_selection = conf_dict["feature_selection"] #True/False
        self.num_features = conf_dict["num_features"]
        self.one_vs_all_type = conf_dict["one_vs_all_type"]

        self.chosen_feature_names = None
        self.chosen_features_all_folds = []


        self.clf_dict = {}
        #self.clf_dict["one_vs_all"] = OneVsRestClassifier(SVC(kernel='rbf', C=1000, gamma=0.001))

        self.clf_dict["output_code"] = OutputCodeClassifier(SVC(kernel='rbf', C=1000, gamma=0.001), code_size=2, random_state=0)

        params_rf = {'n_estimators': 100, 'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 10, 'random_state': 0}
        #params_rf = {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 3, 'max_features': 'auto', 'max_depth': 20, 'random_state': 0}
        self.clf_dict["rf"] = RandomForestClassifier(**params_rf)
#        self.clf_dict["svm"] = SVC(kernel='rbf', C=1000, gamma=0.001)
#        self.clf_dict["svm"] = SVC(kernel='linear', C=1, gamma=0.001)
        params_svm = {'C': 10, 'degree': 2, 'gamma': 'scale', 'kernel': 'sigmoid'}
        self.clf_dict["svm"] = SVC(**params_svm)

        #Naive Bayes classifier is a general term which refers to conditional independence of each of the features in the model, while Multinomial Naive Bayes classifier is a specific instance of a Naive Bayes classifier which uses a multinomial distribution for each of the features.
        self.clf_dict["nb"] = MultinomialNB(alpha=0.00001)
        self.clf_dict["gnb"] = GaussianNB(var_smoothing=0.05)

        self.clf_dict["knn"] = KNeighborsClassifier(n_neighbors = 8)

        params_dt = {'criterion': 'gini', 'max_depth': 20, 'max_features': 'auto', 'min_samples_leaf': 2, 'min_samples_split': 2, 'random_state': 42, 'splitter': 'best'}
        self.clf_dict["dt"] = DecisionTreeClassifier(**params_dt)

        self.clf_dict["one_vs_all"] = OneVsRestClassifier(self.clf_dict[conf_dict["one_vs_all_type"]])

        self.fs_dict = {}
        self.fs_dict["selectKbest_chi2"] = SelectKBest(chi2, k=self.num_features)
        self.fs_dict["selectKbest_fclassif"] =  SelectKBest(f_classif, k=self.num_features)



    def do_feature_selection(self, X_train, y_train, X_test):

        fs = self.fs_dict["selectKbest_fclassif"]

        X_train = fs.fit_transform(X_train, y_train)

        X_test = fs.transform(X_test)

        self.chosen_feature_names = [self.feature_names[i] for i in fs.get_support(indices=True)]
        self.chosen_features_all_folds.append(self.chosen_feature_names)
        #print("Chosen Features by Feature selection: ", self.chosen_feature_names)
        self.chosen_features_df = DataFrame(data=X_train.toarray(), columns=self.chosen_feature_names)
        self.y_train_show = y_train

        return X_train, X_test


    def do_hyperparameter_tuning(self):

        #Make X sparse matrix, y series
        if isinstance(self.X_train, DataFrame):
            X_to_fit = sparse.csr_matrix(self.X_train)
        if isinstance(self.y_train, DataFrame):
            y_to_fit = self.y_train[0]

        #print(X_to_fit, y_to_fit)

        from sklearn.model_selection import GridSearchCV

        if (self.clf_name == "one_vs_all" and self.one_vs_all_type == "svm"):
            parameters = {'kernel': ["linear", 'rbf', "poly", "sigmoid"],
                           'gamma': ["scale", "auto", 1e-3, 1e-4, 1e-5],
                           'degree': [2,3,4],
                         'C': [1, 10, 100, 500, 1000, 5000, 10000],
                         }
            #parameters = {'kernel':('linear', 'rbf'), 'C':[0.1, 1, 10, 100]}
            est = SVC()
            self.clf = GridSearchCV(est, parameters, cv=3, scoring="f1_macro", n_jobs=-1)
            self.clf.fit(X_to_fit, y_to_fit)

        elif (self.clf_name == "one_vs_all" and self.one_vs_all_type == "rf"):

            from sklearn.model_selection import RandomizedSearchCV
            # Number of trees in random forest
            n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
            # Number of features to consider at every split
            max_features = ['auto', 'sqrt']
            # Maximum number of levels in tree
            max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
            max_depth.append(None)
            # Minimum number of samples required to split a node
            min_samples_split = [2, 5, 10]
            # Minimum number of samples required at each leaf node
            min_samples_leaf = [1, 2, 4]

            # Create the random grid
            parameters = {
                         'n_estimators': n_estimators,
                         'max_features': max_features,
                         'max_depth': max_depth,
                         'min_samples_split': min_samples_split,
                         'min_samples_leaf': min_samples_leaf,
                           }

            est = RandomForestClassifier()
#            self.clf = RandomizedSearchCV(estimator = est,
#                                          param_distributions = parameters,
#                                          n_iter = 100,
#                                          cv = 3,
#                                          scoring="f1_macro",
#                                          verbose=2,
#                                          random_state=42,
#                                          n_jobs = -1)
            self.clf = GridSearchCV(est, parameters, cv=3, scoring="f1_macro", n_jobs=-1)
            self.clf.fit(X_to_fit, y_to_fit)


        elif (self.clf_name == "one_vs_all" and self.one_vs_all_type == "dt"):
            parameters = {
                    'criterion': ['gini', 'entropy'],
                    "splitter": ["best", "random"],
                    "max_depth": [int(x) for x in np.linspace(10, 110, num = 11)] + [None],
                    "min_samples_split": [1,2,3,4,5,10],
                    "min_samples_leaf": [1,2,4],
                    "max_features": ["auto", "sqrt"],
                    "random_state": [42],
                         }
            #parameters = {'kernel':('linear', 'rbf'), 'C':[0.1, 1, 10, 100]}
            est = DecisionTreeClassifier()
            self.clf = GridSearchCV(est, parameters, cv=3, scoring="f1_macro", n_jobs=-1)
            self.clf.fit(X_to_fit, y_to_fit)


        elif (self.clf_name == "one_vs_all" and self.one_vs_all_type == "nb"):
            parameters = {'alpha': [1e-06, 0.00001, 0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 1.5, 2, 3, 10, 50],
                         }
            est = MultinomialNB()
            self.clf = GridSearchCV(est, parameters, cv=3, scoring="f1_macro", n_jobs=-1)
            self.clf.fit(X_to_fit, y_to_fit)

        elif (self.clf_name == "one_vs_all" and self.one_vs_all_type == "knn"):
            parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                         }
            est = KNeighborsClassifier()
            self.clf = GridSearchCV(est, parameters, cv=3, scoring="f1_macro", n_jobs=-1)
            self.clf.fit(X_to_fit, y_to_fit)

        elif (self.clf_name == "one_vs_all" and self.one_vs_all_type == "gnb"):
            parameters = {'var_smoothing': [0.01, 0.03, 0.05, 0.07,  0.1, 0.5, 1, 2],
                         }
            est = GaussianNB()
            self.clf = GridSearchCV(est, parameters, cv=3, scoring="f1_macro", n_jobs=-1)
            self.clf.fit(X_to_fit.toarray(), np.array(y_to_fit))

        print("\nBest params from hyperparameter tuning: ", self.clf.best_params_," with score: ", self.clf.best_score_ ,"\n")




    def plot_feature_importance(self, feature_importances, title):

        if self.feature_selection:
            index = self.chosen_feature_names
            n = len(feature_importances)
            n=50
        else:
            index = self.feature_names
            n = 50
        self.df_feature_importances = DataFrame(data = feature_importances,
                                           index = index,
                                           columns=["importance"]) \
                                    .sort_values(by="importance", ascending=False)
        self.df_feature_importances[:n][::-1].plot(kind="barh",figsize=(16,8), grid=True)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.title(title, fontsize=24)
        plt.tight_layout()
        plt.savefig("Plots_metrics/feature_importance_" +title+ ".png")


    def plot_feature_importances_big(self):

        if self.clf_name == "rf":
            self.plot_feature_importance(self.model.feature_importances_, "random_forest")

        elif (self.clf_name == "one_vs_all" and self.one_vs_all_type == "rf"):
            for key,value in self.labels_numeric.items():
                self.plot_feature_importance(self.model.estimators_[value].feature_importances_, "one_vs_all_rf_"+key)

        elif self.clf_name == "svm":
            if self.model.get_params()["kernel"] == "linear":
                for i in range(int(self.n_classes * (self.n_classes-1)/2)):
                    self.plot_feature_importance(self.model.coef_[i].toarray().T, "svm_linear_kernel_" + str(i))

        elif self.clf_name == "one_vs_all" and self.one_vs_all_type == "svm":
            if self.model.get_params()["estimator__kernel"] == "linear":
                for key,value in self.labels_numeric.items():
                    self.plot_feature_importance(self.model.estimators_[value].coef_.toarray().T,  "one_vs_all_svm_"+key)



    def classify(self, X_train, y_train, X_test, y_test, fold=0):

        #Make X sparse matrix, y series
        if isinstance(X_train, DataFrame):
            X_train = sparse.csr_matrix(X_train)
        if isinstance(X_test, DataFrame):
            X_test = sparse.csr_matrix(X_test)
        if isinstance(y_train, DataFrame):
            y_train = y_train[0]
        if isinstance(y_test, DataFrame):
            y_test = y_test[0]

        print("Length train: ", len(y_train))
        print("Length test: ", len(y_test))

        if self.feature_selection:
            X_train, X_test = self.do_feature_selection(X_train, y_train, X_test)

        if (self.clf_name == "output_code") or (self.one_vs_all_type == "output_code"):
            X_train, X_test = X_train.toarray(), X_test.toarray()

        if (self.clf_name == "gnb") or (self.one_vs_all_type == "gnb"):
            X_train, X_test = X_train.toarray(), X_test.toarray()


        self.model.fit(X_train, y_train)
        self.y_pred = self.model.predict(X_test)

        print("Num features: ", X_train.shape[1])
        #Performance
        #print("Model score", self.model.score(X_test, y_test)) #same as accuracy

        print("Accuracy: %.2f%%" % (super().print_accuracy(y_test, self.y_pred)))

        #Plot feature importance if algorithm supports it
        #self.plot_feature_importances_big()

        #Metrics
        Metrics().print_cl_report(y_test, self.y_pred)
        #Metrics().conf_matrix(y_test, self.y_pred, title=str(fold), save=True)


        #Save some metrics
        self.acc = self.model.score(X_test, y_test)
        self.cl_report = Metrics().save_cl_report(y_test, self.y_pred)
        self.f1_score = Metrics().get_f1_score(y_test, self.y_pred)






    def k_fold_classify(self, n_splits):

        kf = KFold(n_splits=n_splits, shuffle=True)
        self.model = self.clf_dict[self.clf_name]
        if self.clf_name == "one_vs_all":
            print("\nOne vs. all classifier of type: ", self.one_vs_all_type)

        i=0
        for train_index, test_index in kf.split(self.X):
            i+=1
            print("\nSplit ", i)
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, self.y_test = self.y[train_index], self.y[test_index]

            self.classify(X_train, y_train, X_test, self.y_test, fold=i)



    def k_fold_per_user_classify(self):

        self.model = self.clf_dict[self.clf_name]


        self.X_train = DataFrame()
        self.y_train = DataFrame()

        self.X_test = DataFrame()
        self.y_test = DataFrame()

        for i in self.y_df.unique():
            if i == 0:
                #webex teams
                good_people = ["dena", "simone", "michela", "martino", "paolo"]
            elif i == 1:
                #msteams
                good_people = ["dena", "paolo", "maurizio", "gianluca"]
            elif i == 2:
                #skype
                good_people = ["antonio", "gianluca", "dena"]
            elif i == 3:
                #jitsi
                good_people = ["antonio", "dena"]
            elif i == 4:
                #zoom
                good_people = ["antonio"]
            elif i == 5:
                #google_meet
                good_people = ["antonio", "other"]


            self.X_0 = self.X_df.loc[self.y_df == i]
            self.y_0 = self.y_df.loc[self.y_df == i]
            self.people_0 = self.people.loc[self.y_df == i]
            #behaviour for label 0

            self.X_train = concat([self.X_train, self.X_0.loc[self.people.isin(good_people)]])
            self.X_test = concat([self.X_test, self.X_0.loc[~self.people.isin(good_people)]])

            self.y_train = concat([self.y_train, self.y_0.loc[self.people.isin(good_people)]])
            self.y_test = concat([self.y_test, self.y_0.loc[~self.people.isin(good_people)]])


        self.classify(self.X_train, self.y_train, self.X_test, self.y_test)









