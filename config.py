#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np


config_dict = {
 "data_folder": "Data",
 "app_list": [
              "webex_teams",
              "msteams",
              "skype",
              "jitsi",
              "zoom",
              #"google_meet",
              ],
 "classifier": "one_vs_all",
 }


config_dict_grid = {
 #seconds before: -1 to take all domains from the whole pcap
 "seconds_before": [25],
 "seconds_after": [0],
 "use_as": [False],
 "vectorizer_min_df": [0.02],
 "vectorizer_max_df": [0.7],
 "feature_selection" : [True],
 "num_features": np.arange(1,5,1),
 "domain_level": ["description"],
 "one_vs_all_type": ["rf"],
 }

output_name = "bla"
n_cores = 48


