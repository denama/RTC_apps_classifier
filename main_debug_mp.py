#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import json
import itertools
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(["tstat_plots.mplstyle"])

from Loader import Loader
from DataVectorizer import DataVectorizer
from Classifier import Classifier


from config import config_dict

import multiprocessing


def main_func(conf_dict):

    try:
        print(conf_dict)
        
        loader = Loader(conf_dict)
        df = loader.dff

        #Data vectorizer trials
        dv = DataVectorizer(df, conf_dict)
        df_classify = dv.make_df_classify()
        X,y, people = dv.create_Xy()

        #Classifier
        estimator = Classifier(X, y, people, dv.df_features, dv.feature_names, conf_dict)


        #Hyperparameter tuning according to scoring function of type of classifier
        #Usually mean accuracy on the given test data and labels
    #    estimator.do_hyperparameter_tuning()

        estimator.k_fold_per_user_classify()
    #    estimator.k_fold_classify(3)

        #Saving data
        d = {}
        for key,value in conf_dict.items():
            if key != "num_features":
                d[key] = value
        d["accuracy"] = estimator.acc
        d["cl_report"] = estimator.cl_report
        d["f1_macro"] = estimator.f1_score
        d["labels_numeric"] = dv.labels_numeric
        d["class_samples"] = dv.class_samples.to_dict()
        d["feature_list"] = estimator.feature_names
        d["feature_number"] = len(estimator.feature_names)
        d["chosen_feature_names"] = estimator.chosen_feature_names
        d["len_chosen_feature_names"] = len(estimator.chosen_feature_names)

        print("F1 macro:", d["f1_macro"], "\n")

        return d

    except:
        e = sys.exc_info()[0]
        print("Error: ", e)
        return None


#Loader trials
if __name__ == "__main__":

#     config_dict_grid = {
#      #seconds before: -1 to take all domains from the whole pcap
#      "seconds_before": [0, 5, 10, 15, 20, 25, 30],
#      "seconds_after": [0, 5, 10, 15, 20, 25, 30],
#      "use_as": [False, True],
#      "vectorizer_min_df": [0.01, 0.02, 0.03, 0.04, 0.06, 0.08],
#      "vectorizer_max_df": [0.5, 0.6, 0.7, 0.8],
#      "feature_selection" : [True, False],
#      "num_features": [2,4,5,6,8,10,15,20,30,40,50,100],
#      "domain_level": ["description", "second_level_domains"],
#      "one_vs_all_type": ["svm", "rf", "nb"],
#      }

    config_dict_grid = {
     #seconds before: -1 to take all domains from the whole pcap
     "seconds_before": [25],
     "seconds_after": [0],
     "use_as": [False],
     "vectorizer_min_df": [0.02],
     "vectorizer_max_df": [0.7],
     "feature_selection" : [True],
     "num_features": np.arange(1,41,1),
     "domain_level": ["description"],
     "one_vs_all_type": ["rf"],
     }


    keys, values = zip(*config_dict_grid.items())
    permutation_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]


    n_cores = 48
    pool = multiprocessing.Pool(processes=n_cores)
    result = pool.map(main_func, permutation_dicts)

    save_path = "./Output_data/feature_selection.json"
    with open(save_path, "w+") as f:
        for value in result:
            if value is not None:
                json.dump(value, f)
                f.write("\n")
