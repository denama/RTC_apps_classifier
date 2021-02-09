#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import json
import itertools
import numpy as np
import matplotlib.pyplot as plt
import os
plt.style.use(["tstat_plots.mplstyle"])

from Loader import Loader
from DataVectorizer import DataVectorizer
from Classifier import Classifier


from config import config_dict, config_dict_grid, output_name, n_cores

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

    
    keys, values = zip(*config_dict_grid.items())
    permutation_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    pool = multiprocessing.Pool(processes=n_cores)
    result = pool.map(main_func, permutation_dicts)
    
    save_folder = "Output_data"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    save_path = os.path.join(save_folder, f"{output_name}.json")
    with open(save_path, "w+") as f:
        for value in result:
            if value is not None:
                json.dump(value, f)
                f.write("\n")
