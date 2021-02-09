#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#if you don't put self, it's a class method - don't need an instance to access it
#def print_accuracy(y_true, y_pred)
#Metrics.print_accuracy
from config import config_dict

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

from pandas import DataFrame
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score



class Metrics:

    def __init__(self):

        self.app_list = config_dict["app_list"]
        self.labels_numeric = {name: i for i, name in enumerate(self.app_list)}


    def print_accuracy(self, y_true, y_pred):

        y_true = np.array(y_true)
        n_right = 0
        for i in range(len(y_pred)):
            if y_pred[i] == y_true[i]:
                n_right += 1
        return (n_right/float(len(y_true)) * 100)



    def conf_matrix(self, y_test, y_predict, title ='Confusion Matrix', save = False):

        dict_label = {y:x for x,y in self.labels_numeric.items()}
        labels = list(dict_label.keys())
        #labels_pre = pd.Series(y_predict).unique()
        #labels_test = pd.Series(y_test).unique()
        #labels = list( set(labels_pre).union(labels_test))
        cm_ = confusion_matrix(y_test,y_predict, labels = labels)

        plt.figure(figsize = (14,12))
        sub_label_name = [dict_label[k] for k in labels]
        cm_df = DataFrame(cm_, columns = sub_label_name, index = sub_label_name)
        cm_df["All"] = cm_df.sum( axis = 1)

        fig, ax = sns.heatmap(cm_df, annot = True, cbar=False, cmap = 'Greens', fmt = 'd').set_ylim(len(cm_), -0.5)
        #ax.tick_params(axis='both', which='major', pad=15)
        plt.yticks(rotation=0, fontsize=24)
        plt.xticks(fontsize=24)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        #plt.title(title)
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join("Plots_metrics", "cm_"+title+".png"), dpi = 300)
        else:
            plt.show()



    def print_cl_report(self, y_true, y_pred):

        #F1 = 2 * (precision * recall) / (precision + recall)
        from sklearn.metrics import classification_report
        cl_report = classification_report(y_true, y_pred,
                    target_names=self.labels_numeric.keys(),
                    zero_division=0)
        print(cl_report)


    def save_cl_report(self, y_true, y_pred):

        from sklearn.metrics import classification_report
        self.cl_report = classification_report(y_true, y_pred,
                         target_names=self.labels_numeric.keys(),
                         zero_division=0, output_dict=True)
        return self.cl_report



    def print_cl_report2(self, y_true, y_pred):

        precision_recall_fscore_support(y_true, y_pred, average=None, \
                                labels=list(self.labels_numeric.values()), \
                                zero_division=0)


    def get_f1_score(self, y_true, y_pred):

        f1score = f1_score(y_true, y_pred, labels=list(self.labels_numeric.values()), pos_label=1, average='macro', zero_division=0)
        return f1score

    def get_jaccard_score(self, y_true, y_pred):

        jaccard = jaccard_score(y_true, y_pred, labels=list(self.labels_numeric.values()), pos_label=1, average='macro', zero_division=0)
        return jaccard



