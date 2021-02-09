#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import json
import itertools
import matplotlib.pyplot as plt
plt.style.use(["tstat_plots.mplstyle"])

from Loader import Loader
from DataVectorizer import DataVectorizer
from Classifier import Classifier
import os


from config import config_dict, config_dict_grid, output_name, n_cores


#Loader trials
if __name__ == "__main__":



    keys, values = zip(*config_dict_grid.items())
    permutation_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    
    save_to_file = 1
    
    if save_to_file:
        save_folder = "Output_data"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_path = os.path.join(save_folder, f"{output_name}.json")
        save_file = open(save_path, "w+")
        
    
    i=0

    for conf_dict in permutation_dicts:

        print("\n", i)
        i+=1
        print(conf_dict)

        try:
            loader = Loader(conf_dict)
            df = loader.dff

            data_bla = loader.data_bla
#            all_domains = loader.all_domains
#            bad_domains = loader.bad_domains
#            d_domains = loader.get_per_app_domains_count()
            #loader.plot_pie_apps(which_apps="to_classify", title_addon=" - all")

            #Data vectorizer trials
            dv = DataVectorizer(df, conf_dict)
            df_classify = dv.make_df_classify()
            X,y, people = dv.create_Xy()

            #Classifier
            estimator = Classifier(X, y, people, dv.df_features, dv.feature_names, conf_dict)


            estimator.k_fold_per_user_classify()
        #    estimator.k_fold_classify(3)

            #Hyperparameter tuning according to scoring function f1_score
#            estimator.do_hyperparameter_tuning()


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

            print("F1 score macro: ", d["f1_macro"])

            if save_to_file:
                json.dump(d, save_file)
                save_file.write("\n")

        except:
            e = sys.exc_info()[0]
            print("Error: ", e)

    if save_to_file:
        save_file.close()


#Plot len description
#plt.figure(figsize=(12,5))
#df["len"].hist(bins=50, color="indianred")
#plt.xlabel("Number of domains visited during call", fontsize=24)
#plt.xticks(fontsize=24)
#plt.yticks(fontsize=24)
#plt.tight_layout()
#plt.savefig("len_domains.png")

#plt.scatter(df["len"], df["person"])
#plt.savefig("How_many_domains_per_person.png")
#plt.tight_layout()
